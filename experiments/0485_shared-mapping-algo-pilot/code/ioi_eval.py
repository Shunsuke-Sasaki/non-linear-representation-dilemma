#!/usr/bin/env python
"""Experiment 0485 (llm-jp/experiments#485), IOI part: shared non-linear map on Pythia-410m seeds.

Design: shared_mapp_diffrent_algo/ioi_design.md. Same research question as main_eval.py
(distributive law): does a RevNet phi shared by several source models, plus a per-model orthogonal
A_m, transfer to an unseen model, and does that separate a correct candidate from a wrong one?

Models: EleutherAI/pythia-410m-seed{1..5}, revisions step143000 ("trained") and step0 ("random").
Intervention: output of gpt_neox.layers[12] at the last prompt token; the first 512 dims of
phi(A_m h) are the high-level variable (as in scripts/das_llm.py with intervention_dim_proportion
0.5).
Candidates (both pick one of the base prompt's two names):
  pos    Z = "the IO is the second name of the opening clause" (the variable of the original
         Preprocess_IOI_Data / make_intervention_dataset_LLM)
  alpha  Z = "the IO name precedes the S name alphabetically" (wrong by design)
For both, the counterfactual answer is the base IO if Z(source) == Z(base), otherwise the base S.
Data: "default" = fahamu/ioi mecha_ioi_26m.parquet (as das_llm.py), "namesplit" = templates with
disjoint train / eval / test names (as Generate_LLM_Eval_Intervention_Data_NameSplit).

Stages (resumable):
  prepare                       download models and data, tokenise, build intervention pairs
  maps --task i --num-tasks n   train every n-th map from i (optionally --kind / --cand filter)
  adapt --task t                held-out evaluation for (fold, state) task t
  summarize                     tables
  smoke                         tests-sized end-to-end run with timings under results/ioi_smoke
  bench                         full-size timings and precision checks (needs smoke data) -> results/ioi_bench
"""
import argparse
import copy
import json
import os
import random
import re
import shutil
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "source"))

from das.RevNet import RevNet  # noqa: E402

STATES = {"trained": "step143000", "random": "step0"}
CANDIDATES = ["pos", "alpha"]
SPLITS = ["default", "namesplit"]
CONDITIONS = ["no_shared", "single_model", "shared_joint"]

FULL = dict(
    seeds=[1, 2, 3, 4, 5],
    model_name="EleutherAI/pythia-410m-seed{seed}",
    tokenizer_name="EleutherAI/pythia-410m-seed1",
    layer=12, hidden=1024, n_var_dims=512,
    revnet_blocks=8, revnet_width=64, revnet_depth=1,
    dataset_repo="fahamu/ioi", dataset_file="mecha_ioi_26m.parquet",
    default_rows=2_000_000,  # das_llm.py truncates to 2 * intervention_train_size
    namesplit_sentences=200_000,
    data_seed=20261107,
    map_train=1_000_000, dev=2048, test=8192, adapt_train=4096, adapt_dev=1024, cal_size=4096,
    map_batch=256, map_lr=1e-3, map_warmup=500, map_max_epochs=3, map_val_interval=100,
    map_patience=25, map_threshold=1e-3, map_sched_patience=10,
    adapt_sizes=[0, 256, 1024, 4096], adapt_updates=300, adapt_batch=64, adapt_lr=1e-3,
    adapt_val_interval=50, adapt_inits=["procrustes", "random_orthogonal"],
    eval_batch=256, matmul_precision="high",
)
SMOKE = dict(
    FULL,
    seeds=[1, 2, 3],
    default_rows=20_000, namesplit_sentences=20_000,
    map_train=2000, dev=128, test=256, adapt_train=128, adapt_dev=64, cal_size=256,
    map_max_epochs=1, map_val_interval=4, map_warmup=2,
    adapt_sizes=[0, 64], adapt_updates=6, adapt_val_interval=3,
)

# Templates and names of das/Dataset_Generation.py Generate_LLM_Eval_Intervention_Data_NameSplit.
BABA = [
    "Then, {B} and {A} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {B} and {A} had a lot of fun at the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {B} and {A} were working at the {PLACE}. {B} decided to give a {OBJECT} to {A}",
    "Then, {B} and {A} were thinking about going to the {PLACE}. {B} wanted to give a {OBJECT} to {A}",
    "Then, {B} and {A} had a long argument, and afterwards {B} said to {A}",
    "After {B} and {A} went to the {PLACE}, {B} gave a {OBJECT} to {A}",
    "When {B} and {A} got a {OBJECT} at the {PLACE}, {B} decided to give it to {A}",
    "When {B} and {A} got a {OBJECT} at the {PLACE}, {B} decided to give the {OBJECT} to {A}",
    "While {B} and {A} were working at the {PLACE}, {B} gave a {OBJECT} to {A}",
    "While {B} and {A} were commuting to the {PLACE}, {B} gave a {OBJECT} to {A}",
    "After the lunch, {B} and {A} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Afterwards, {B} and {A} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {B} and {A} had a long argument. Afterwards {B} said to {A}",
    "The {PLACE} {B} and {A} went to had a {OBJECT}. {B} gave it to {A}",
    "Friends {B} and {A} found a {OBJECT} at the {PLACE}. {B} gave it to {A}",
]
ABBA = [t.replace("{B} and {A}", "{A} and {B}", 1) for t in BABA]
NAMES = ['Faye', 'Yolanda', 'Fannie', 'Mildred', 'Doris', 'Melissa', 'Shannon', 'Natasha', 'Nellie', 'Stacey',
         'Sonia', 'Kelly', 'Marsha', 'Inez', 'Jacqueline', 'Violet', 'Miriam', 'Jerry', 'Brian', 'Stella',
         'Rachel', 'Pauline', 'Yvette', 'Rebecca', 'Marilyn', 'Diana', 'Chelsea', 'Kathryn', 'Lula', 'Anne',
         'Bernadette', 'Loretta', 'Kevin', 'David', 'Alyssa', 'Gwen', 'Amanda', 'Charlene', 'Irma', 'Megan',
         'Stacy', 'Virginia', 'Rosemarie', 'Jessica', 'Paula', 'Richard', 'Jackie', 'Teresa', 'Leona', 'Carol',
         'Juana', 'Kristi', 'Evelyn', 'Brandy', 'William', 'Deanna', 'Sarah', 'Eleanor', 'Maggie', 'Harriet',
         'Kari', 'Rosa', 'Angie', 'Kristina', 'Melanie', 'Mark', 'Martha', 'Lorraine', 'Agnes', 'Janet',
         'Geneva', 'Robin', 'Ella', 'Heidi', 'Suzanne', 'Wilma', 'Velma', 'Pat', 'Estelle', 'Erika', 'Alice',
         'Sophia', 'Anita', 'Alicia', 'Carmen', 'Lee', 'Grace', 'Susan', 'Sheila', 'Leigh', 'Mona', 'Kim',
         'Jody', 'Mattie', 'Ginger', 'Ethel', 'Theresa', 'Dianne', 'Marian', 'Jessie']
PLACES = ['store', 'theater', 'museum', 'shop', 'beach', 'school', 'mountain', 'desert', 'river', 'cafe',
          'library', 'cave', 'forest', 'market', 'zoo', 'park', 'home', 'lake', 'bar', 'gym']
OBJECTS = ['apple', 'raspberry', 'orange', 'grape', 'blueberry', 'coconut', 'banana', 'plum', 'kiwi', 'papaya',
           'cherry', 'pear', 'blackberry', 'pineapple', 'honeydew', 'cantaloupe', 'watermelon', 'mango', 'peach',
           'strawberry']


def stable_seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) % (2 ** 31)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")  # unique: parallel tasks may write the same file
    with open(tmp, "wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())  # a node failure after rename left truncated (16 MiB) files on Lustre (job 100041)
    tmp.replace(path)


##############################################################################
##  Data                                                                    ##
##############################################################################

def parse_sentences(sentences, tokenizer):
    """Preprocess_IOI_Data (das/Dataset_Generation.py) that also keeps the two names.

    Returns prompt strings and, per prompt, (tok_io, tok_s, io_is_second, io_alpha_first).
    """
    verify_cache = {}

    def verify(a, b):  # das/Dataset_Generation.py verify()
        key = (a, b)
        if key not in verify_cache:
            ta, tb = tokenizer.tokenize(" " + a), tokenizer.tokenize(" " + b)
            if "".join(tokenizer.tokenize(a)) == "".join(ta[1:]):
                verify_cache[key] = (False, ta[0] != tb[0])
            else:
                verify_cache[key] = (True, ta[0] != tb[0])
        return verify_cache[key]

    first_token = {}

    def tok(word):
        if word not in first_token:
            first_token[word] = tokenizer(word, add_special_tokens=False)["input_ids"][0]
        return first_token[word]

    prompts, meta = [], []
    for text in sentences:
        io = text.split(" ")[-1]
        cutoff = text[:-1 * len(io) - 1]
        m = re.search(r"\b([A-Z][a-z]+)\s+(?:gave|decided|wanted|said)\b", text)
        if m is None:
            continue
        s = m.group(1)
        io_is_second = text.find(io) > text.find(s)
        second, first = (io, s) if io_is_second else (s, io)
        space, differ = verify(second, first)
        if not differ or io == s:
            continue
        if space:
            t_io, t_s = tok(" " + io), tok(" " + s)
        else:
            cutoff = cutoff + " "
            t_io, t_s = tok(io), tok(s)
        prompts.append(cutoff)
        meta.append((t_io, t_s, io_is_second, io < s))
    return prompts, meta


def make_pool(prompts, meta, tokenizer):
    enc = tokenizer(prompts, add_special_tokens=True)["input_ids"]
    lengths = torch.tensor([len(e) for e in enc], dtype=torch.long)
    ids = torch.full((len(enc), int(lengths.max())), tokenizer.pad_token_id, dtype=torch.int32)
    for i, e in enumerate(enc):
        ids[i, :len(e)] = torch.tensor(e, dtype=torch.int32)
    m = torch.tensor([[a, b] for a, b, _, _ in meta], dtype=torch.long)
    return dict(ids=ids, lengths=lengths, tok_io=m[:, 0], tok_s=m[:, 1],
                pos=torch.tensor([x[2] for x in meta]), alpha=torch.tensor([x[3] for x in meta]))


def namesplit_sentences(names, n, rng):
    out = []
    for _ in range(n):
        a, b = rng.sample(names, 2)
        t = rng.choice(BABA) if rng.random() < 0.5 else rng.choice(ABBA)
        out.append(t.replace("{A}", a).replace("{B}", b)
                   .replace("{OBJECT}", rng.choice(OBJECTS)).replace("{PLACE}", rng.choice(PLACES)))
    return out


class Pairs:
    """Intervention pairs: indices into a pool (base and source prompts)."""

    def __init__(self, pool, base, src):
        self.pool, self.base, self.src = pool, base, src

    def __len__(self):
        return len(self.base)

    def subset(self, idx):
        return Pairs(self.pool, self.base[idx], self.src[idx])

    def labels(self, cand):
        p = self.pool
        same = p[cand][self.src] == p[cand][self.base]
        tok_io, tok_s = p["tok_io"][self.base], p["tok_s"][self.base]
        true = torch.where(same, tok_io, tok_s)
        false = torch.where(same, tok_s, tok_io)
        return true, false, ~same  # ~same: the counterfactual answer differs from the base answer


def batch_tokens(pool, idx, device):
    lengths = pool["lengths"][idx]
    width = int(lengths.max())
    ids = pool["ids"][idx, :width].long()
    mask = (torch.arange(width)[None, :] < lengths[:, None]).long()
    return ids.to(device), mask.to(device), (lengths - 1).to(device)


POSITIONS = ["last", "s2", "s1", "io"]


def token_positions(pool, which):
    """Per-prompt token index of the intervention site (cached in the pool), -1 where it cannot be located.

    last: the final prompt token (END). s1 / s2: first / second occurrence of the S name's token (rows where it
    occurs exactly twice). io: the IO name's token in the opening clause (rows where it occurs exactly once).
    The name tokens are the answer tokens of parse_sentences, so a name tokenised differently inside the
    sentence (e.g. sentence-initial, without a leading space) is not found and the row gets -1.
    """
    key = "_at_" + which
    if key not in pool:
        lengths = pool["lengths"]
        if which == "last":
            at = lengths - 1
        elif which in ("s1", "s2", "io"):
            ids = pool["ids"]
            tok = (pool["tok_io"] if which == "io" else pool["tok_s"]).to(ids.dtype)
            width = ids.shape[1]
            hit = (ids == tok[:, None]) & (torch.arange(width)[None, :] < lengths[:, None])
            count = hit.sum(1)
            first = hit.int().argmax(1)
            last = width - 1 - hit.flip(1).int().argmax(1)
            if which == "io":
                at = torch.where(count == 1, first, -1)
            else:
                at = torch.where(count == 2, first if which == "s1" else last, -1)
        else:
            raise ValueError(which)
        pool[key] = at
    return pool[key]


##############################################################################
##  Models and intervention                                                 ##
##############################################################################

class _Stop(Exception):
    pass


def layer_module(model, layer):
    return model.gpt_neox.layers[layer]


def _out_tensor(output):
    return output[0] if isinstance(output, tuple) else output


def _replace(output, new):
    return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new


@torch.no_grad()
def hidden_last(model, ids, mask, last, layer):
    """Layer output at the last prompt token (forward stops after the layer)."""
    store = {}

    def hook(module, inp, output):
        h = _out_tensor(output)
        store["h"] = h[torch.arange(h.shape[0], device=h.device), last].float()
        raise _Stop

    handle = layer_module(model, layer).register_forward_hook(hook)
    try:
        model.gpt_neox(input_ids=ids, attention_mask=mask, use_cache=False)
    except _Stop:
        pass
    finally:
        handle.remove()
    return store["h"]


def last_logits(model, ids, mask, last, layer=None, edit=None, at=None):
    """Logits at the last prompt token; edit(h) replaces the layer output at token `at` (default: last)."""
    handle = None
    if edit is not None:
        site = last if at is None else at

        def hook(module, inp, output):
            h = _out_tensor(output)
            rows = torch.arange(h.shape[0], device=h.device)
            new = h.clone()
            new[rows, site] = edit(h[rows, site].float()).to(h.dtype)
            return _replace(output, new)
        handle = layer_module(model, layer).register_forward_hook(hook)
    try:
        hs = model.gpt_neox(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    finally:
        if handle is not None:
            handle.remove()
    return model.get_output_embeddings()(hs[torch.arange(hs.shape[0], device=hs.device), last])


class IdentityMap(nn.Module):
    def forward(self, x):
        return x

    def inverse(self, x):
        return x


class FixedOrthogonal(nn.Module):
    def __init__(self, Q):
        super().__init__()
        self.register_buffer("Q", Q.detach().float().clone())

    def matrix(self):
        return self.Q

    def forward(self, h):
        return h @ self.Q

    def inverse(self, z):
        return z @ self.Q.T


class IdentityAlignment(nn.Module):
    def matrix(self):
        return None

    def forward(self, h):
        return h

    def inverse(self, z):
        return z


class TrainableOrthogonalAlignment(nn.Module):
    """Right-side Cayley update of an initial orthogonal matrix (main_eval.py, Joint_Alignment_PoC)."""

    def __init__(self, Q0):
        super().__init__()
        self.register_buffer("initial", Q0.detach().float().clone())
        self.skew = nn.Parameter(torch.zeros_like(self.initial))

    def matrix(self):
        skew = self.skew - self.skew.T
        eye = torch.eye(skew.shape[0], device=skew.device, dtype=skew.dtype)
        return self.initial @ torch.linalg.solve(eye + skew, eye - skew)

    def forward(self, h):
        return h @ self.matrix()

    def inverse(self, z):
        return z @ self.matrix().T

    def frozen(self):
        with torch.no_grad():
            return FixedOrthogonal(self.matrix())


def random_orthogonal(n, seed):
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn((n, n), generator=g))
    return q * torch.where(torch.diag(r) < 0, -1.0, 1.0)


def procrustes(H_ref, H_m):
    """Uncentred orthogonal Procrustes H_m Q ~ H_ref (das/Shared_Mapping.estimate_procrustes)."""
    U, _, Vt = torch.linalg.svd(H_m.double().T @ H_ref.double())
    Q = (U @ Vt).float()
    return Q, ((H_m @ Q - H_ref) ** 2).mean().item()


def intervened_logits(phi, member, pairs, cfg, device):
    model, align = member
    k = cfg["n_var_dims"]
    b_ids, b_mask, b_last = batch_tokens(pairs.pool, pairs.base, device)
    s_ids, s_mask, s_last = batch_tokens(pairs.pool, pairs.src, device)
    which = cfg.get("position", "last")  # results/ioi_eval (no "position" key) used the last token
    if which == "last":
        b_at, s_at = b_last, s_last
    else:
        at = token_positions(pairs.pool, which)
        b_at, s_at = at[pairs.base].to(device), at[pairs.src].to(device)
        if (b_at < 0).any() or (s_at < 0).any():
            raise ValueError(f"pairs without a {which} token; filter them first")
    z_src = phi(align(hidden_last(model, s_ids, s_mask, s_at, cfg["layer"])))

    def edit(h):
        z = phi(align(h))
        z = torch.cat([z_src[:, :k], z[:, k:]], dim=1)
        return align.inverse(phi.inverse(z))

    return last_logits(model, b_ids, b_mask, b_last, cfg["layer"], edit, at=b_at)


def pair_loss_acc(logits, true, false):
    true, false = true.to(logits.device), false.to(logits.device)
    loss = nn.functional.cross_entropy(logits, true, reduction="none")  # LLM_Criterion_CE (full vocab)
    rows = torch.arange(logits.shape[0], device=logits.device)
    correct = logits[rows, true] > logits[rows, false]
    return loss, correct


@torch.no_grad()
def evaluate(phi, member, pairs, cand, cfg, device):
    true, false, changed = pairs.labels(cand)
    losses, corrects = [], []
    for start in range(0, len(pairs), cfg["eval_batch"]):
        sl = slice(start, start + cfg["eval_batch"])
        sub = pairs.subset(torch.arange(len(pairs))[sl])
        loss, correct = pair_loss_acc(intervened_logits(phi, member, sub, cfg, device), true[sl], false[sl])
        losses.append(loss.cpu())
        corrects.append(correct.cpu())
    loss, correct = torch.cat(losses), torch.cat(corrects).float()
    return {"iia": correct.mean().item(), "loss": loss.mean().item(),
            "iia_label_changed": correct[changed].mean().item() if changed.any() else float("nan"),
            "iia_label_unchanged": correct[~changed].mean().item() if (~changed).any() else float("nan"),
            "label_changed_rate": changed.float().mean().item(), "n": len(pairs)}


@torch.no_grad()
def task_accuracy(model, pool, idx, cfg, device):
    correct = []
    for start in range(0, len(idx), cfg["eval_batch"]):
        sel = idx[start:start + cfg["eval_batch"]]
        ids, mask, last = batch_tokens(pool, sel, device)
        logits = last_logits(model, ids, mask, last)
        rows = torch.arange(len(sel), device=device)
        correct.append((logits[rows, pool["tok_io"][sel].to(device)] > logits[rows, pool["tok_s"][sel].to(device)]).cpu())
    return torch.cat(correct).float().mean().item()


##############################################################################
##  Experiment                                                              ##
##############################################################################

class Experiment:
    def __init__(self, root, cfg, device, offline=True):
        self.root, self.cfg, self.device, self.offline = Path(root), cfg, device, offline
        self._data, self._models = {}, {}

    def path(self, *parts):
        return self.root.joinpath(*parts)

    def write_config(self):
        p = self.path("config.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text(json.dumps(self.cfg, indent=2))

    def folds(self):
        seeds = self.cfg["seeds"]
        out = []
        for k, held in enumerate(seeds):
            sources = [s for s in seeds if s != held]
            out.append({"heldout": held, "sources": sources, "ref": sources[0]})
        return out

    # --- downloads and models
    def tokenizer(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.cfg["tokenizer_name"], revision=STATES["trained"],
                                            local_files_only=self.offline)
        tok.pad_token = tok.eos_token
        return tok

    def download_models(self):
        from huggingface_hub import list_repo_files, snapshot_download
        for seed in self.cfg["seeds"]:
            for rev in STATES.values():
                name = self.cfg["model_name"].format(seed=seed)
                files = list_repo_files(name, revision=rev)
                weights = "*.safetensors" if any(f.endswith(".safetensors") for f in files) else "*.bin"
                snapshot_download(name, revision=rev, allow_patterns=["*.json", "*.txt", weights])
                log(f"downloaded {name}@{rev}")

    def get_model(self, state, seed):
        key = (state, seed)
        if key not in self._models:
            from transformers import AutoModelForCausalLM
            name = self.cfg["model_name"].format(seed=seed)
            model = AutoModelForCausalLM.from_pretrained(name, revision=STATES[state], local_files_only=self.offline)
            model = model.float().to(self.device).eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self._models[key] = model
        return self._models[key]

    def drop_models(self):
        self._models.clear()
        torch.cuda.empty_cache()

    # --- data
    def make_data(self):
        c = self.cfg
        tok = None
        for split in SPLITS:
            p = self.path("data", f"{split}.pt")
            if p.exists():
                continue
            tok = tok or self.tokenizer()
            rng = random.Random(stable_seed("sentences", split, c["data_seed"]))
            t0 = time.time()
            if split == "default":
                import pyarrow.parquet as pq
                from huggingface_hub import hf_hub_download
                f = hf_hub_download(repo_id=c["dataset_repo"], filename=c["dataset_file"], repo_type="dataset")
                sentences = []  # first default_rows rows, as das_llm.py (data.iloc[:2 * train_size])
                for rb in pq.ParquetFile(f).iter_batches(batch_size=100_000, columns=["ioi_sentences"]):
                    sentences += rb.column(0).to_pylist()
                    if len(sentences) >= c["default_rows"]:
                        break
                sentences = sentences[:c["default_rows"]]
                prompts, meta = parse_sentences(sentences, tok)
                order = list(range(len(prompts)))
                rng.shuffle(order)
                n1, n2 = int(len(order) * 0.8), int(len(order) * 0.9)
                parts = {"train": order[:n1], "eval": order[n1:n2], "test": order[n2:]}
                pools = {k: make_pool([prompts[i] for i in v], [meta[i] for i in v], tok) for k, v in parts.items()}
            else:
                n = c["namesplit_sentences"]
                lengths = {"train": int(n / 10 * 8), "eval": int(n / 10), "test": int(n / 10)}
                a, b = round(len(NAMES) / 10 * 6), round(len(NAMES) / 10 * 8)
                names = {"train": NAMES[:a], "eval": NAMES[a:b], "test": NAMES[b:]}
                pools = {}
                for k in ["train", "eval", "test"]:
                    prompts, meta = parse_sentences(namesplit_sentences(names[k], lengths[k], rng), tok)
                    pools[k] = make_pool(prompts, meta, tok)
            g = torch.Generator().manual_seed(stable_seed("pairs", split, c["data_seed"]))

            def pairs(pool_name, n):
                size = len(pools[pool_name]["lengths"])
                return {"pool": pool_name, "base": torch.randint(size, (n,), generator=g),
                        "src": torch.randint(size, (n,), generator=g)}

            sets = {"train": pairs("train", c["map_train"]), "dev": pairs("eval", c["dev"]),
                    "adapt_train": pairs("eval", c["adapt_train"]), "adapt_dev": pairs("eval", c["adapt_dev"]),
                    "test": pairs("test", c["test"]),
                    "cal": {"pool": "train", "base": torch.randperm(len(pools["train"]["lengths"]), generator=g)[:c["cal_size"]]}}
            atomic_save({"pools": pools, "sets": sets}, p)
            log(f"data {split}: pools {[len(v['lengths']) for v in pools.values()]}, "
                f"max length {max(int(v['lengths'].max()) for v in pools.values())}, {time.time() - t0:.0f}s")

    def data(self, split):
        if split not in self._data:
            self._data[split] = torch.load(self.path("data", f"{split}.pt"), weights_only=False)
        return self._data[split]

    def pairs(self, split, name):
        d = self.data(split)
        s = d["sets"][name]
        return Pairs(d["pools"][s["pool"]], s["base"], s["src"])

    def calibration(self, state, seed, split):
        p = self.path("calibration", f"{state}_seed{seed}_{split}.pt")
        if p.exists():
            return torch.load(p, map_location=self.device, weights_only=False)
        d = self.data(split)
        pool, idx = d["pools"]["train"], d["sets"]["cal"]["base"]
        model = self.get_model(state, seed)
        which = self.cfg.get("position", "last")
        if which != "last":
            idx = idx[token_positions(pool, which)[idx] >= 0]
        hs = []
        for start in range(0, len(idx), self.cfg["eval_batch"]):
            sel = idx[start:start + self.cfg["eval_batch"]]
            ids, mask, last = batch_tokens(pool, sel, self.device)
            at = last if which == "last" else token_positions(pool, which)[sel].to(self.device)
            hs.append(hidden_last(model, ids, mask, at, self.cfg["layer"]))
        H = torch.cat(hs)
        atomic_save(H.cpu(), p)
        return H

    def procrustes_init(self, state, ref, seed, split):
        return procrustes(self.calibration(state, ref, split), self.calibration(state, seed, split))

    # --- maps
    def new_phi(self, seed):
        torch.manual_seed(seed)
        c = self.cfg
        return RevNet(number_blocks=c["revnet_blocks"], in_features=c["hidden"], hidden_size=c["revnet_width"],
                      depth=c["revnet_depth"]).to(self.device)

    def map_path(self, kind, state, split, cand, idx):
        return self.path("maps", state, split, cand, f"{kind}_{idx}.pt")

    def map_tasks(self):
        out = []
        for kind in ["individual", "joint"]:
            for state in STATES:
                for cand in CANDIDATES:
                    for split in SPLITS:
                        for i in range(len(self.cfg["seeds"])):
                            out.append((kind, state, cand, split, i))
        return out

    def load_phi(self, kind, state, split, cand, idx):
        ckpt = torch.load(self.map_path(kind, state, split, cand, idx), map_location=self.device, weights_only=False)
        phi = self.new_phi(0)
        phi.load_state_dict(ckpt["phi"])
        phi.eval()
        for p in phi.parameters():
            p.requires_grad_(False)
        return phi

    def train_map(self, kind, state, cand, split, idx):
        c = self.cfg
        out = self.map_path(kind, state, split, cand, idx)
        if out.exists():
            log(f"skip {out}")
            return
        if kind == "individual":
            seeds, label = [c["seeds"][idx]], f"seed{c['seeds'][idx]}"
        else:
            fold = self.folds()[idx]
            seeds, label = fold["sources"], f"fold{idx} (held-out seed {fold['heldout']})"
        ref = seeds[0]
        members, trainable = [], []
        for s in seeds:
            if s == ref:
                align = IdentityAlignment()
            else:
                Q, _ = self.procrustes_init(state, ref, s, split)
                align = TrainableOrthogonalAlignment(Q).to(self.device)
                trainable.append(align)
            members.append((self.get_model(state, s), align))
        restart = c.get("map_restart", 0)  # 0 keeps the seeds of the earlier runs
        phi = self.new_phi(stable_seed("phi", kind, state, cand, split, idx, *([restart] if restart else [])))
        clip, margin = c.get("map_clip", 0), c.get("map_rollback")  # both off unless set (phase 7 of ioi_ioname)
        params = list(phi.parameters()) + [p for a in trainable for p in a.parameters()]
        optimizer = optim.Adam(params, lr=c["map_lr"])
        plateau = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=c["map_sched_patience"])
        train, dev = self.pairs(split, "train"), self.pairs(split, "dev")
        per_member = c["map_batch"] // len(members)
        updates_per_epoch = len(train) // (per_member * len(members))
        max_updates = updates_per_epoch * c["map_max_epochs"]

        state_path = out.with_suffix(".resume.pt")
        st = dict(update=0, best_loss=float("inf"), best=None, bad=0, history=[], rollbacks=0, lr_scale=1.0)
        r = None
        if state_path.exists():
            try:
                r = torch.load(state_path, map_location=self.device, weights_only=False)
            except RuntimeError as e:  # truncated by a node failure: keep it for inspection and start over
                bad = state_path.with_suffix(f".corrupt{int(time.time())}")
                state_path.rename(bad)
                log(f"unreadable resume file ({e.__class__.__name__}); moved to {bad.name}, starting from update 0")
        if r is not None:
            phi.load_state_dict(r["phi"])
            for a, s in zip(trainable, r["aligns"]):
                a.load_state_dict(s)
            optimizer.load_state_dict(r["optimizer"])
            plateau.load_state_dict(r["plateau"])
            st = r["st"]
            log(f"resume {out} at update {st['update']}")

        def dev_eval():
            phi.eval()
            res = [evaluate(phi, m, dev, cand, c, self.device) for m in members]
            phi.train()
            return res

        log(f"map {kind} {state} {cand} {split} {label}: {len(members)} members, {per_member}/member/update, "
            f"max {max_updates} updates")
        t0 = time.time()
        phi.train()
        while st["update"] < max_updates and st["bad"] < c["map_patience"]:
            u = st["update"]
            epoch, pos = divmod(u, updates_per_epoch)
            perm = torch.randperm(len(train), generator=torch.Generator().manual_seed(
                stable_seed("order", kind, state, cand, split, idx, epoch)))
            sel = perm[pos * per_member * len(members):(pos + 1) * per_member * len(members)]
            if u < c["map_warmup"]:  # linear warm-up (das_llm.py lr_warmup_steps), then ReduceLROnPlateau
                for grp in optimizer.param_groups:
                    grp["lr"] = c["map_lr"] * st.get("lr_scale", 1.0) * (u + 1) / c["map_warmup"]
            optimizer.zero_grad()
            total = 0.0
            for j, m in enumerate(members):
                sub = train.subset(sel[j * per_member:(j + 1) * per_member])
                true, false, _ = sub.labels(cand)
                loss, _ = pair_loss_acc(intervened_logits(phi, m, sub, c, self.device), true, false)
                loss = loss.mean() / len(members)
                loss.backward()
                total += loss.item()
            if clip:
                torch.nn.utils.clip_grad_norm_(params, clip)
            optimizer.step()
            st["update"] += 1
            if st["update"] % c["map_val_interval"] == 0 or st["update"] == max_updates:
                res = dev_eval()
                dev_loss = sum(r["loss"] for r in res) / len(res)
                diverged = margin is not None and st["best"] is not None and not dev_loss <= st["best_loss"] + margin
                if diverged:  # back to the best point, half the learning rate, fresh Adam moments
                    phi.load_state_dict(st["best"]["phi"])
                    for a, s_ in zip(trainable, st["best"]["align_states"]):
                        a.load_state_dict(s_)
                    optimizer.state.clear()
                    st["rollbacks"] += 1
                    st["lr_scale"] *= 0.5
                    for grp in optimizer.param_groups:
                        grp["lr"] *= 0.5
                    log(f"  update {st['update']}: dev_loss {dev_loss:.4f} > best {st['best_loss']:.4f} + {margin}; "
                        f"rollback {st['rollbacks']} to update {st['best']['update']}, lr {optimizer.param_groups[0]['lr']:.2e}")
                    if st["rollbacks"] > c.get("map_max_rollbacks", 4):
                        log("  too many rollbacks; stopping")
                        st["bad"] = c["map_patience"]
                elif st["update"] > c["map_warmup"]:
                    plateau.step(dev_loss)
                if diverged:
                    pass  # neither a new best nor a bad check
                elif dev_loss < st["best_loss"]:
                    st["bad"] = 0 if st["best_loss"] - dev_loss > c["map_threshold"] else st["bad"] + 1
                    st["best_loss"] = dev_loss
                    st["best"] = {"phi": copy.deepcopy(phi.state_dict()),
                                  "aligns": [a.matrix().detach().cpu() for a in trainable],
                                  "align_states": [copy.deepcopy(a.state_dict()) for a in trainable],
                                  "update": st["update"]}
                else:
                    st["bad"] += 1
                st["history"].append({"update": st["update"], "train_loss": total, "dev_loss": dev_loss,
                                      "dev_iia": [r["iia"] for r in res], "lr": optimizer.param_groups[0]["lr"],
                                      "seconds": time.time() - t0})
                log(f"  update {st['update']}: train_loss={total:.4f} dev_loss={dev_loss:.4f} "
                    f"dev_iia={[round(r['iia'], 4) for r in res]} bad={st['bad']} ({time.time() - t0:.0f}s)")
                atomic_save({"phi": phi.state_dict(), "aligns": [a.state_dict() for a in trainable],
                             "optimizer": optimizer.state_dict(),
                             "plateau": plateau.state_dict(), "st": st}, state_path)
        if st["best"] is None:
            raise RuntimeError("no checkpoint was selected")
        phi.load_state_dict(st["best"]["phi"])
        phi.eval()
        test = self.pairs(split, "test")
        members_best = [(members[0][0], IdentityAlignment())] + [
            (m[0], FixedOrthogonal(Q.to(self.device))) for m, Q in zip(members[1:], st["best"]["aligns"])]
        tests = [evaluate(phi, m, test, cand, c, self.device) for m in members_best]
        atomic_save({"phi": st["best"]["phi"], "alignments": {str(s): (None if i == 0 else st["best"]["aligns"][i - 1])
                                                            for i, s in enumerate(seeds)},
                     "info": {"kind": kind, "state": state, "cand": cand, "split": split, "idx": idx, "seeds": seeds,
                              "best_update": st["best"]["update"], "updates": st["update"],
                              "best_dev_loss": st["best_loss"], "rollbacks": st.get("rollbacks", 0),
                              "restart": restart, "map_lr": c["map_lr"], "map_clip": clip, "map_rollback": margin,
                              "seconds": time.time() - t0, "history": st["history"]},
                     "test": tests}, out)
        state_path.unlink(missing_ok=True)
        log(f"saved {out}: best update {st['best']['update']}, source test IIA {[round(t['iia'], 4) for t in tests]}")

    # --- adaptation
    def adapt(self, phi, model, Q0, train, dev, cand, n, seed):
        c = self.cfg
        align = TrainableOrthogonalAlignment(Q0).to(self.device)
        if n == 0:
            return align.frozen(), 0, evaluate(phi, (model, align.frozen()), dev, cand, c, self.device)["loss"], 0.0
        optimizer = optim.Adam(align.parameters(), lr=c["adapt_lr"])
        best = (align.matrix().detach().clone(), 0,
                evaluate(phi, (model, align.frozen()), dev, cand, c, self.device)["loss"])
        g = torch.Generator().manual_seed(seed)
        for update in range(1, c["adapt_updates"] + 1):
            sub = train.subset(torch.randint(n, (c["adapt_batch"],), generator=g))
            true, false, _ = sub.labels(cand)
            optimizer.zero_grad()
            loss, _ = pair_loss_acc(intervened_logits(phi, (model, align), sub, c, self.device), true, false)
            loss.mean().backward()
            optimizer.step()
            if update % c["adapt_val_interval"] == 0 or update == c["adapt_updates"]:
                d = evaluate(phi, (model, align.frozen()), dev, cand, c, self.device)["loss"]
                if d < best[2]:
                    best = (align.matrix().detach().clone(), update, d)
        final = FixedOrthogonal(best[0])
        moved = torch.linalg.norm(best[0] - Q0.to(best[0].device)).item()
        return final, best[1], best[2], moved

    def adapt_tasks(self):
        return [(k, state) for k in range(len(self.cfg["seeds"])) for state in STATES]

    def run_adapt(self, task):
        c = self.cfg
        fold_index, state = self.adapt_tasks()[task]
        fold = self.folds()[fold_index]
        held, ref = fold["heldout"], fold["ref"]
        out = self.path("adapt", f"fold{fold_index}_{state}.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        done = {json.loads(x)["key"] for x in out.read_text().splitlines()} if out.exists() else set()
        target = self.get_model(state, held)
        t_start = time.time()
        with out.open("a") as fh:
            def write(row):
                fh.write(json.dumps(row) + "\n")
                fh.flush()

            for split in SPLITS:
                test = self.pairs(split, "test")
                a_train, a_dev = self.pairs(split, "adapt_train"), self.pairs(split, "adapt_dev")
                Qp, cost = self.procrustes_init(state, ref, held, split)
                inits = {"procrustes": Qp,
                         "random_orthogonal": random_orthogonal(c["hidden"], stable_seed("rand_orth", fold_index, state, split)).to(self.device)}
                for cand in CANDIDATES:
                    common = dict(fold=fold_index, heldout=held, ref=ref, state=state, split=split, cand=cand)
                    key = f"{state}|{split}|{cand}|individual_self"
                    if key not in done:
                        phi = self.load_phi("individual", state, split, cand, c["seeds"].index(held))
                        write(dict(evaluate(phi, (target, IdentityAlignment()), test, cand, c, self.device),
                                   key=key, condition="individual_self", adapt_init=None, adapt_examples=None, **common))
                    phis = {"no_shared": IdentityMap(),
                            "single_model": self.load_phi("individual", state, split, cand, c["seeds"].index(ref)),
                            "shared_joint": self.load_phi("joint", state, split, cand, fold_index)}
                    for cond, phi in phis.items():
                        for ainit, Q0 in inits.items():
                            for n in c["adapt_sizes"]:
                                key = f"{state}|{split}|{cand}|{cond}|{ainit}|n{n}"
                                if key in done:
                                    continue
                                t0 = time.time()
                                align, best_update, dev_loss, moved = self.adapt(
                                    phi, target, Q0, a_train, a_dev, cand, n,
                                    stable_seed("adapt", fold_index, state, split, cand, cond, ainit, n))
                                res = evaluate(phi, (target, align), test, cand, c, self.device)
                                write(dict(res, key=key, condition=cond, adapt_init=ainit, adapt_examples=n,
                                           adapt_best_update=best_update, adapt_dev_loss=dev_loss,
                                           distance_from_init=moved,
                                           procrustes_cost=cost if ainit == "procrustes" else None,
                                           seconds=time.time() - t0, **common))
                    log(f"fold {fold_index} {state} {split} {cand} done ({time.time() - t_start:.0f}s so far)")

    # --- summary
    def summarize(self):
        import pandas as pd

        def md(frame, index=True):
            return "```\n" + frame.to_string(index=index) + "\n```"

        c = self.cfg
        lines = ["# 0485 IOI evaluation summary", ""]
        acc, base = [], []
        for split in SPLITS:
            d = self.data(split)
            test = self.pairs(split, "test")
            for cand in CANDIDATES:
                _, _, changed = test.labels(cand)
                base.append({"split": split, "cand": cand, "label_changed_rate": changed.float().mean().item()})
            t_pos, _, _ = test.labels("pos")
            t_alpha, _, _ = test.labels("alpha")
            base.append({"split": split, "cand": "pos=alpha agreement",
                         "label_changed_rate": (t_pos == t_alpha).float().mean().item()})
            pool = d["pools"]["test"]
            idx = test.base.unique()[:c["test"]]
            for state in STATES:
                for seed in c["seeds"]:
                    acc.append({"split": split, "state": state, "seed": seed,
                                "task_accuracy": task_accuracy(self.get_model(state, seed), pool, idx, c, self.device)})
                self.drop_models()
        lines += ["## Model task accuracy on the test prompts (IO logit > S logit)", "",
                  md(pd.DataFrame(acc).pivot_table(index="seed", columns=["split", "state"], values="task_accuracy")), ""]
        lines += ["## Test pairs: rate of changed labels, and agreement of the two candidates' labels", "",
                  md(pd.DataFrame(base), index=False), ""]

        maps = []
        for p in sorted(self.path("maps").rglob("*.pt")):
            if p.name.endswith(".resume.pt"):
                continue
            ck = torch.load(p, map_location="cpu", weights_only=False)
            i = ck["info"]
            maps.append({"kind": i["kind"], "state": i["state"], "split": i["split"], "cand": i["cand"], "idx": i["idx"],
                         "source_test_iia_mean": float(np.mean([t["iia"] for t in ck["test"]])),
                         "best_update": i["best_update"], "updates": i["updates"], "minutes": i["seconds"] / 60})
        if maps:
            m = pd.DataFrame(maps)
            m.to_csv(self.path("maps.csv"), index=False)
            lines += ["## Maps: source test IIA (mean over members), mean over seeds / folds", "",
                      md(m.groupby(["state", "split", "cand", "kind"])[["source_test_iia_mean", "best_update", "minutes"]]
                         .mean().round(4)), ""]

        rows = []
        for f in sorted(self.path("adapt").glob("*.jsonl")):
            rows += [json.loads(x) for x in f.read_text().splitlines()]
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(self.path("adapt_results.csv"), index=False)
            ind = df[df.condition == "individual_self"].groupby(["state", "split", "cand"])["iia"].agg(["mean", "min", "max"])
            lines += ["## Individual map on the held-out model itself (reference)", "", md(ind.round(4)), ""]
            main = df[df.condition != "individual_self"]
            g = main.groupby(["state", "split", "cand", "adapt_init", "condition", "adapt_examples"])
            lines += ["## Held-out IIA, mean over folds", "", md(g["iia"].mean().unstack("adapt_examples").round(4)), ""]
            rng = (g["iia"].min().unstack("adapt_examples").round(3).astype(str) + "–"
                   + g["iia"].max().unstack("adapt_examples").round(3).astype(str))
            lines += ["## Held-out IIA, min–max over folds", "", md(rng), ""]
            nmax = max(c["adapt_sizes"])
            ch = main[main.adapt_examples == nmax].groupby(["state", "split", "cand", "adapt_init", "condition"])[
                ["iia_label_changed", "iia_label_unchanged"]].mean()
            lines += [f"## IIA split by whether the label changes (n={nmax})", "", md(ch.round(4)), ""]
            mv = g["distance_from_init"].mean().unstack("adapt_examples").round(3)
            lines += ["## Target A_m: Frobenius distance moved from its initial value", "", md(mv), ""]
        self.path("summary.md").write_text("\n".join(lines) + "\n")
        log(f"wrote {self.path('summary.md')}")


def run_stage(exp, stage, task, num_tasks, kind=None, cand=None):
    if stage == "prepare":
        exp.write_config()
        exp.download_models()
        exp.make_data()
    elif stage == "maps":
        tasks = [t for t in exp.map_tasks() if (kind is None or t[0] == kind) and (cand is None or t[2] == cand)]
        for t in tasks[task::num_tasks]:
            exp.train_map(*t)
            exp.drop_models()
    elif stage == "adapt":
        exp.run_adapt(task)
    elif stage == "summarize":
        exp.summarize()
    else:
        raise ValueError(stage)


def bench(device, out_dir, n_updates=100, n_warm=10):
    """Full-size per-update / eval timings and numerical checks for each matmul precision.

    Uses the smoke data and models (seeds 1-3); the 4-member joint map reuses seed 1 as the 4th member,
    which costs the same as a distinct model. Batch sizes are the FULL ones.
    """
    exp = Experiment(ROOT / "results" / "ioi_smoke", FULL, device, offline=True)
    c = FULL
    train = exp.pairs("default", "train")
    model = exp.get_model("trained", 1)
    models4 = [model, exp.get_model("trained", 2), exp.get_model("trained", 3), model]
    g = torch.Generator().manual_seed(0)

    def timed(step, n):
        for _ in range(n_warm):
            step()
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(n):
            step()
        torch.cuda.synchronize()
        return (time.time() - t) / n

    def sample(n):
        return train.subset(torch.randint(len(train), (n,), generator=g))

    results = {}
    for precision in ["high", "highest"]:
        torch.set_float32_matmul_precision(precision)
        torch.cuda.reset_peak_memory_stats()
        r = {}

        # individual map: phi + identity alignment, batch 256
        phi = exp.new_phi(1)
        opt = optim.Adam(phi.parameters(), lr=1e-4)

        def step_individual():
            sub = sample(c["map_batch"])
            true, false, _ = sub.labels("pos")
            opt.zero_grad()
            loss, _ = pair_loss_acc(intervened_logits(phi, (model, IdentityAlignment()), sub, c, device), true, false)
            loss.mean().backward()
            opt.step()
        r["map_individual_s_per_update"] = timed(step_individual, n_updates)

        # joint map: 4 members (3 trainable Cayley alignments), 64 per member
        phi = exp.new_phi(2)
        aligns = [IdentityAlignment()] + [TrainableOrthogonalAlignment(random_orthogonal(c["hidden"], s)).to(device)
                                          for s in (1, 2, 3)]
        opt = optim.Adam(list(phi.parameters()) + [p for a in aligns[1:] for p in a.parameters()], lr=1e-4)
        per = c["map_batch"] // 4

        def step_joint():
            opt.zero_grad()
            for m, a in zip(models4, aligns):
                sub = sample(per)
                true, false, _ = sub.labels("pos")
                loss, _ = pair_loss_acc(intervened_logits(phi, (m, a), sub, c, device), true, false)
                (loss.mean() / 4).backward()
            opt.step()
        r["map_joint_s_per_update"] = timed(step_joint, n_updates)

        # adaptation: frozen phi, one trainable alignment, batch 64
        phi.eval()
        for p in phi.parameters():
            p.requires_grad_(False)
        align = TrainableOrthogonalAlignment(random_orthogonal(c["hidden"], 4)).to(device)
        opt = optim.Adam(align.parameters(), lr=1e-3)

        def step_adapt():
            sub = sample(c["adapt_batch"])
            true, false, _ = sub.labels("pos")
            opt.zero_grad()
            loss, _ = pair_loss_acc(intervened_logits(phi, (model, align), sub, c, device), true, false)
            loss.mean().backward()
            opt.step()
        r["adapt_s_per_update"] = timed(step_adapt, n_updates)

        # evaluation throughput (pairs per second), scaled to the FULL test size
        ev = train.subset(torch.arange(min(len(train), 2048)))
        frozen = align.frozen()
        evaluate(phi, (model, frozen), ev.subset(torch.arange(256)), "pos", c, device)
        torch.cuda.synchronize()
        t = time.time()
        evaluate(phi, (model, frozen), ev, "pos", c, device)
        torch.cuda.synchronize()
        r["eval_s_per_pair"] = (time.time() - t) / len(ev)
        r["eval_test_s"] = r["eval_s_per_pair"] * c["test"]

        # numerical checks: no-op intervention (src = base) through a random RevNet + rotation
        same = Pairs(ev.pool, ev.base, ev.base)
        rot = FixedOrthogonal(random_orthogonal(c["hidden"], 3).to(device))
        phi0 = exp.new_phi(7).eval()
        diffs, flips = [], []
        with torch.no_grad():
            for start in range(0, len(same), c["eval_batch"]):
                sub = same.subset(torch.arange(len(same))[start:start + c["eval_batch"]])
                ids, mask, last = batch_tokens(sub.pool, sub.base, device)
                plain = last_logits(model, ids, mask, last)
                noop = intervened_logits(phi0, (model, rot), sub, c, device)
                rows = torch.arange(len(sub), device=device)
                tio, ts = sub.pool["tok_io"][sub.base].to(device), sub.pool["tok_s"][sub.base].to(device)
                d_plain = plain[rows, tio] - plain[rows, ts]
                d_noop = noop[rows, tio] - noop[rows, ts]
                diffs.append((plain - noop).abs().max().item())
                flips.append(((d_plain > 0) != (d_noop > 0)).cpu())
            a = TrainableOrthogonalAlignment(random_orthogonal(c["hidden"], 1)).to(device)
            a.skew.normal_(0, 0.01)
            h = torch.randn(256, c["hidden"], device=device) * 3
            r["cayley_inverse_max_abs_err"] = (a.inverse(a(h)) - h).abs().max().item()
            r["revnet_inverse_max_abs_err"] = (phi0.inverse(phi0(h)) - h).abs().max().item()
        r["noop_max_abs_dlogit"] = max(diffs)
        r["noop_io_vs_s_flip_rate"] = torch.cat(flips).float().mean().item()
        r["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2**30
        results[precision] = r
        log(f"bench {precision}: " + json.dumps({k: round(v, 5) for k, v in r.items()}))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bench.json").write_text(json.dumps(results, indent=2))
    log(f"wrote {out_dir / 'bench.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "maps", "adapt", "summarize", "smoke", "bench"])
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=1)
    parser.add_argument("--kind", choices=["individual", "joint"], default=None)
    parser.add_argument("--cand", choices=CANDIDATES, default=None)
    parser.add_argument("--root", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this on a GPU node")
    device = torch.device("cuda")
    torch.set_num_threads(max(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), 1))
    log(f"stage={args.stage} task={args.task}/{args.num_tasks} kind={args.kind} cand={args.cand} "
        f"device={torch.cuda.get_device_name(device)}")

    if args.stage == "bench":
        bench(device, Path(args.root or ROOT / "results" / "ioi_bench"))
        return

    if args.stage == "smoke":
        torch.set_float32_matmul_precision(SMOKE["matmul_precision"])
        root = Path(args.root or ROOT / "results" / "ioi_smoke")
        if root.exists():
            shutil.rmtree(root)
        exp = Experiment(root, SMOKE, device, offline=False)
        timings = {}
        t = time.time(); run_stage(exp, "prepare", 0, 1); timings["prepare"] = time.time() - t
        exp.offline = True
        for kind in ["individual", "joint"]:
            tasks = [x for x in exp.map_tasks() if x[0] == kind]
            t = time.time(); run_stage(exp, "maps", 0, 1, kind=kind)
            timings[f"maps_{kind}_per_map"] = (time.time() - t) / len(tasks)
        t = time.time()
        for k in range(len(exp.adapt_tasks())):
            run_stage(exp, "adapt", k, 1)
            exp.drop_models()
        timings["adapt_per_task"] = (time.time() - t) / len(exp.adapt_tasks())
        run_stage(exp, "summarize", 0, 1)
        n_rows = sum(len(f.read_text().splitlines()) for f in exp.path("adapt").glob("*.jsonl"))
        (root / "timings.json").write_text(json.dumps(timings, indent=2))
        log(f"smoke finished: {n_rows} adaptation rows, timings {json.dumps({k: round(v, 1) for k, v in timings.items()})}")
        log(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        return

    root = Path(args.root or ROOT / "results" / "ioi_eval")
    cfg = json.loads((root / "config.json").read_text()) if (root / "config.json").exists() else FULL
    torch.set_float32_matmul_precision(cfg["matmul_precision"])
    exp = Experiment(root, cfg, device, offline=args.stage != "prepare")
    run_stage(exp, args.stage, args.task, args.num_tasks, args.kind, args.cand)
    log(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
