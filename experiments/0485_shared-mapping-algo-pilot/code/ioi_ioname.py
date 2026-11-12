#!/usr/bin/env python
"""Experiment 0485, IOI phase 5: the IO-name candidate against C_pos, under one 4-way criterion.

Candidates (same intervention site, same answer when nothing is intervened on):
  pos   Z = whether the IO is the second name of the opening clause (C_pos of ioi_eval). Counterfactual answer:
        the base IO if Z(source) == Z(base), otherwise the base S.
  name  Z = the IO name itself. Counterfactual answer: the source IO, a name that does not occur in the base prompt.
Criterion: argmax over the first tokens {base IO, base S, source IO, source S} (chance 25%); also the argmax over
the whole vocabulary. The training loss is the full-vocabulary cross entropy of ioi_eval (unchanged).
Pairs: the ioi_eval pairs restricted to those whose four first tokens are distinct (so the two candidates always
predict different tokens) and whose prompts do not end with a separate space token (parse_sentences "space" rows,
whose answer tokens have no leading space).
phi complexity: identity (no training) < rotation (orthogonal, das/Rotation_Model.py) < RevNet with 1, 2, 4, 8
blocks (width 64, depth 1; the ioi_eval map is revnet8). Same layer output at END, same 512 intervened dims.
Model: Pythia-410m seed 1, trained (step143000) at layers LAYERS, random (step0) at layer 12 as the control.
Splits: default and namesplit (the evaluation names of namesplit are not seen in training).

Stages:
  smoke5                 checks on results/ioi_smoke data -> results/ioi_ioname_smoke
  identity               identity phi (512 dims) and full END patching (1024 dims), no training -> eval/identity.jsonl
  maps --task i --num-tasks n [--layers 12 ...]   every n-th map of map_tasks(layers), then its 4-way evaluation
Results: results/ioi_ioname (maps/L{layer}/{phi}/..., eval/*.jsonl). Data: results/ioi_eval/data (read only).

Phase 6 (transfer of C_name; layer 12, END, revnet8, trained seeds 1-5, the folds of ioi_eval):
  smoke6                 checks on results/ioi_smoke -> results/ioi_ioname_transfer_smoke
  tmaps --task i --num-tasks n   C_name joint maps (5 folds x 2 splits) and individual maps (5 seeds x 2 splits)
  tadapt --task k        held-out adaptation of fold k (A_m only, Procrustes init): shared_joint and no_shared,
                         n in T_SIZES, T_REPEATS repeats (new minibatch seed and n-subset), and individual_self;
                         4-way evaluation of both candidates -> results/ioi_ioname_transfer/adapt/fold{k}.jsonl
Phase 7 (phase 6 maps diverged: loss jumped to ~10.4 and the IIA of every source fell to ~.5; no clipping):
  both candidates retrained on the disjoint pairs with lr 3e-4, gradient clipping 1.0, and a rollback to the best
  point (half lr, fresh Adam moments) when the dev loss exceeds the best by 0.5; --restarts phi initialisations each,
  the one with the lowest source dev loss is used (held-out data are not looked at). Maps whose 2-way source test
  IIA is >= S_PASS on every source are flagged (joint_passed / indiv_passed) in the adaptation rows.
  smoke7                 checks on results/ioi_smoke -> results/ioi_ioname_stable_smoke
  smaps --task i --num-tasks n [--restarts r]   every n-th map of smap_tasks -> results/ioi_ioname_stable/r{r}/
  sadapt --task k [--restarts r]                selection (selection.json) and the phase-6 adaptation of fold k
  Phase 6 note: C_pos reuses the joint / individual maps of results/ioi_eval (trained on all pairs; the C_name maps are trained
  on the disjoint pairs). Adaptation and test pairs are the disjoint ones for both candidates.
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch

import ioi_eval as ie
import ioi_diag as idg
from das.RevNet import RevNet
from das.Rotation_Model import Rotation

ROOT = ie.ROOT
EVAL_ROOT = ROOT / "results" / "ioi_eval"
SMOKE_ROOT = ROOT / "results" / "ioi_smoke"
OUT = ROOT / "results" / "ioi_ioname"
CANDS = ["pos", "name"]
TRAINED_PHIS = ["rotation", "revnet1", "revnet2", "revnet4", "revnet8"]
LAYERS = [6, 12, 18]
RANDOM_LAYERS = [12]
SEED = 1
FOUR = ["base_io", "base_s", "src_io", "src_s"]
log = ie.log


class NamePairs(ie.Pairs):
    def subset(self, idx):
        return NamePairs(self.pool, self.base[idx], self.src[idx])

    def four(self):
        p = self.pool
        return torch.stack([p["tok_io"][self.base], p["tok_s"][self.base],
                            p["tok_io"][self.src], p["tok_s"][self.src]], 1)

    def labels(self, cand):
        if cand == "name":
            true, false = self.pool["tok_io"][self.src], self.pool["tok_io"][self.base]
            return true, false, torch.ones(len(self), dtype=torch.bool)
        return super().labels(cand)


def disjoint(pairs, space_id):
    """Pairs whose four first tokens are distinct and whose prompts do not end with a lone space token."""
    p = NamePairs(pairs.pool, pairs.base, pairs.src)
    f = p.four()
    distinct = torch.ones(len(p), dtype=torch.bool)
    for i in range(4):
        for j in range(i + 1, 4):
            distinct &= f[:, i] != f[:, j]
    ids, lengths = p.pool["ids"], p.pool["lengths"]
    last = ids[torch.arange(len(lengths)), lengths - 1].long()
    ok = distinct & (last[p.base] != space_id) & (last[p.src] != space_id)
    return p.subset(torch.nonzero(ok).squeeze(1))


class NameExperiment(idg.DiagExperiment):
    def __init__(self, root, cfg, device, data_root, models=None):
        super().__init__(root, cfg, device, data_root, offline=True)
        if models is not None:
            self._models = models  # shared between the per-(layer, phi) experiments
        self._pairs, self._space = {}, None

    def space_id(self):
        if self._space is None:
            self._space = self.tokenizer()(" ", add_special_tokens=False)["input_ids"][-1]
        return self._space

    def pairs(self, split, name):
        key = (split, name)
        if key not in self._pairs:
            self._pairs[key] = disjoint(super().pairs(split, name), self.space_id())
        return self._pairs[key]

    def new_phi(self, seed):
        torch.manual_seed(seed)
        c, kind = self.cfg, self.cfg["phi"]
        if kind == "identity":
            return ie.IdentityMap()
        if kind == "rotation":
            return Rotation(c["hidden"]).to(self.device)
        if kind.startswith("revnet"):
            return RevNet(number_blocks=int(kind[len("revnet"):]), in_features=c["hidden"],
                          hidden_size=c["revnet_width"], depth=c["revnet_depth"]).to(self.device)
        raise ValueError(kind)


def name_exp(cfg, data_root, out, layer, phi, device, models=None, tag=None):
    cfg = dict(cfg, seeds=[SEED], layer=layer, phi=phi, position="last")
    exp = NameExperiment(Path(out) / "maps" / f"L{layer}" / (tag or phi), cfg, device, data_root, models)
    exp.write_config()
    return exp


@torch.no_grad()
def eval4(phi, member, pairs, cfg, device):
    """4-way and vocabulary IIA of both candidates, and which of the four names the intervened model picks."""
    four = pairs.four()
    zsame = pairs.pool["pos"][pairs.src] == pairs.pool["pos"][pairs.base]
    want = {"pos": torch.where(zsame, 0, 1), "name": torch.full((len(pairs),), 2)}
    picks, tops, ces = [], [], {c: [] for c in CANDS}
    for start in range(0, len(pairs), cfg["eval_batch"]):
        sl = slice(start, start + cfg["eval_batch"])
        sub = pairs.subset(torch.arange(len(pairs))[sl])
        logits = ie.intervened_logits(phi, member, sub, cfg, device)
        f = four[sl].to(device)
        picks.append(logits.gather(1, f).argmax(1).cpu())
        hit = logits.argmax(1)[:, None] == f  # which of the four is the vocabulary argmax (-1: none)
        tops.append(torch.where(hit.any(1), hit.float().argmax(1), -1).cpu())
        for c in CANDS:
            ces[c].append(torch.nn.functional.cross_entropy(
                logits, f.gather(1, want[c][sl].to(device)[:, None]).squeeze(1), reduction="none").cpu())
    pick, top = torch.cat(picks), torch.cat(tops)
    out = {"n": len(pairs), "zsame_rate": zsame.float().mean().item(),
           "top_outside_four": (top < 0).float().mean().item()}
    for c in CANDS:
        hit = pick == want[c]
        out[f"{c}_iia4"] = hit.float().mean().item()
        out[f"{c}_iia_vocab"] = (top == want[c]).float().mean().item()
        out[f"{c}_loss"] = torch.cat(ces[c]).mean().item()
    out["pos_iia4_zdiff"] = (pick[~zsame] == 1).float().mean().item()  # the pairs where C_pos changes the answer
    out["pos_iia4_zsame"] = (pick[zsame] == 0).float().mean().item()
    for i, nm in enumerate(FOUR):
        out[f"pick_{nm}"] = (pick == i).float().mean().item()
        out[f"pick_{nm}_zsame"] = (pick[zsame] == i).float().mean().item()
        out[f"pick_{nm}_zdiff"] = (pick[~zsame] == i).float().mean().item()
    return out


def map_tasks(layers):
    out = [("trained", L, phi, split, cand) for L in layers for phi in TRAINED_PHIS
           for split in ie.SPLITS for cand in CANDS]
    out += [("random", L, phi, split, cand) for L in RANDOM_LAYERS if L in layers for phi in TRAINED_PHIS
            for split in ie.SPLITS for cand in CANDS]
    return out


def done_keys(path):
    return {json.loads(x)["key"] for x in path.read_text().splitlines()} if path.exists() else set()


def write_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def run_maps(cfg, data_root, out, tasks, eval_path, device):
    models = {}
    done = done_keys(eval_path)
    for state, layer, phi_kind, split, cand in tasks:
        key = f"{state}|L{layer}|{phi_kind}|{split}|{cand}"
        if key in done:
            log(f"skip {key}")
            continue
        exp = name_exp(cfg, data_root, out, layer, phi_kind, device, models)
        exp.train_map("individual", state, cand, split, 0)
        ckpt = torch.load(exp.map_path("individual", state, split, cand, 0), map_location="cpu", weights_only=False)
        phi = exp.load_phi("individual", state, split, cand, 0)
        t0 = time.time()
        res = eval4(phi, (exp.get_model(state, SEED), ie.IdentityAlignment()), exp.pairs(split, "test"), exp.cfg, device)
        info = ckpt["info"]
        write_row(eval_path, dict(res, key=key, state=state, seed=SEED, layer=layer, phi=phi_kind, split=split,
                                  trained_on=cand, best_update=info["best_update"], updates=info["updates"],
                                  map_seconds=info["seconds"], eval_seconds=time.time() - t0,
                                  n_train=len(exp.pairs(split, "train"))))
        log(f"{key}: pos iia4 {res['pos_iia4']:.4f}, name iia4 {res['name_iia4']:.4f} "
            f"(best update {info['best_update']}/{info['updates']}, {info['seconds']:.0f}s)")


def run_identity(cfg, data_root, out, layers, states, path, device):
    """Identity phi on the first 512 dims, full END patching (1024 dims), and the clean 2-way accuracy."""
    models, done = {}, done_keys(path)
    for state in states:
        for layer in layers:
            for k in (cfg["n_var_dims"], cfg["hidden"]):
                exp = name_exp(dict(cfg, n_var_dims=k), data_root, out, layer, "identity", device, models,
                               tag=f"identity_k{k}")
                for split in ie.SPLITS:
                    key = f"{state}|L{layer}|identity_k{k}|{split}"
                    if key in done:
                        continue
                    test = exp.pairs(split, "test")
                    model = exp.get_model(state, SEED)
                    res = eval4(ie.IdentityMap(), (model, ie.IdentityAlignment()), test, exp.cfg, device)
                    clean = ie.task_accuracy(model, test.pool, test.base, exp.cfg, device)
                    write_row(path, dict(res, key=key, state=state, seed=SEED, layer=layer, phi="identity",
                                         n_var_dims=k, split=split, clean_acc2=clean))
                    log(f"{key}: pos iia4 {res['pos_iia4']:.4f}, name iia4 {res['name_iia4']:.4f}, clean {clean:.4f}")


def time_updates(exp, n=10, warm=2):
    """Seconds per training update of a fresh phi (batch map_batch, C_name labels, no dev evaluation)."""
    phi = exp.new_phi(0)
    opt = torch.optim.Adam(phi.parameters(), lr=exp.cfg["map_lr"])
    train, member = exp.pairs("default", "train"), (exp.get_model("trained", SEED), ie.IdentityAlignment())
    g = torch.Generator().manual_seed(0)
    for i in range(n + warm):
        if i == warm:
            torch.cuda.synchronize()
            t0 = time.time()
        sub = train.subset(torch.randint(len(train), (exp.cfg["map_batch"],), generator=g))
        true, false, _ = sub.labels("name")
        opt.zero_grad()
        loss, _ = ie.pair_loss_acc(ie.intervened_logits(phi, member, sub, exp.cfg, exp.device), true, false)
        loss.mean().backward()
        opt.step()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


def smoke5(device):
    torch.set_float32_matmul_precision(ie.SMOKE["matmul_precision"])
    out = ROOT / "results" / "ioi_ioname_smoke"
    if out.exists():
        shutil.rmtree(out)
    cfg = dict(ie.SMOKE)
    t = time.time()
    exp = name_exp(cfg, SMOKE_ROOT, out, 12, "revnet1", device)
    space = exp.space_id()
    for split in ie.SPLITS:
        for name in ["train", "dev", "test"]:
            orig, p = ie.Experiment.pairs(exp, split, name), exp.pairs(split, name)
            f = p.four()
            assert all((f[:, i] != f[:, j]).all() for i in range(4) for j in range(i + 1, 4))
            last = p.pool["ids"][torch.arange(len(p.pool["lengths"])), p.pool["lengths"] - 1]
            assert (last[p.base] != space).all() and (last[p.src] != space).all()
            true, false, changed = p.labels("name")
            assert torch.equal(true, p.pool["tok_io"][p.src]) and changed.all()
            t_pos, f_pos, _ = p.labels("pos")
            t_ref, f_ref, _ = ie.Pairs.labels(p, "pos")
            assert torch.equal(t_pos, t_ref) and torch.equal(f_pos, f_ref)
            log(f"{split}/{name}: kept {len(p)}/{len(orig)} pairs ({len(p) / len(orig):.3f}), "
                f"Z_pos same {(p.pool['pos'][p.src] == p.pool['pos'][p.base]).float().mean():.3f}")
    log(f"check A (disjoint pairs, labels) passed ({time.time() - t:.0f}s)")

    t = time.time()
    last_layer = 23
    full = name_exp(dict(cfg, n_var_dims=cfg["hidden"]), SMOKE_ROOT, out, last_layer, "identity", device,
                    tag="identity_full")
    test = full.pairs("default", "test")
    res = eval4(ie.IdentityMap(), (full.get_model("trained", SEED), ie.IdentityAlignment()), test, full.cfg, device)
    log(f"full END patching at L{last_layer}: name iia4 {res['name_iia4']:.4f}, pos iia4 {res['pos_iia4']:.4f}, "
        f"picks {[round(res['pick_' + x], 3) for x in FOUR]}")
    assert res["name_iia4"] > 0.7, res["name_iia4"]  # the output becomes the source's own output
    log(f"check B (full patching at the last layer gives the source IO) passed ({time.time() - t:.0f}s)")

    t = time.time()
    path = out / "eval" / "maps.jsonl"
    tasks = [("trained", 12, phi, "default", cand) for phi in ["rotation", "revnet1"] for cand in CANDS]
    tasks += [("random", 12, "revnet1", "namesplit", "name"), ("trained", 18, "revnet8", "namesplit", "name")]
    run_maps(cfg, SMOKE_ROOT, out, tasks, path, device)
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert len(rows) == len(tasks), len(rows)
    for state, layer, phi_kind, split, cand in tasks:
        e = name_exp(cfg, SMOKE_ROOT, out, layer, phi_kind, device)
        ckpt = torch.load(e.map_path("individual", state, split, cand, 0), map_location="cpu", weights_only=False)
        again = ie.evaluate(e.load_phi("individual", state, split, cand, 0), (e.get_model(state, SEED),
                            ie.IdentityAlignment()), e.pairs(split, "test"), cand, e.cfg, device)
        assert abs(again["iia"] - ckpt["test"][0]["iia"]) < 0.02, (again["iia"], ckpt["test"][0]["iia"])
        r = next(r for r in rows if r["key"] == f"{state}|L{layer}|{phi_kind}|{split}|{cand}")
        log(f"{r['key']}: reload IIA {again['iia']:.4f} = saved {ckpt['test'][0]['iia']:.4f}")
    run_maps(cfg, SMOKE_ROOT, out, tasks, path, device)  # resume: nothing new
    assert len(path.read_text().splitlines()) == len(tasks)
    log(f"check C (map training per phi, reload, resume) passed ({time.time() - t:.0f}s)")

    t = time.time()
    run_identity(cfg, SMOKE_ROOT, out, [12], ["trained"], out / "eval" / "identity.jsonl", device)
    assert len((out / "eval" / "identity.jsonl").read_text().splitlines()) == 4
    log(f"check D (identity stage) passed ({time.time() - t:.0f}s)")

    t = time.time()
    for layer in [6, 18]:
        for phi_kind in ["rotation", "revnet8"]:
            e = name_exp(cfg, SMOKE_ROOT, out, layer, phi_kind, device, tag=f"timing_{phi_kind}")
            log(f"timing L{layer} {phi_kind}: {time_updates(e):.3f} s/update (batch {cfg['map_batch']})")
    log(f"check E (timings) done ({time.time() - t:.0f}s)")
    log(f"smoke5 passed; peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


T_OUT = ROOT / "results" / "ioi_ioname_transfer"
T_SIZES = [0, 256, 1024, 4096]
T_REPEATS = 3
T_CONDS = ["shared_joint", "no_shared"]


def transfer_exp(cfg, data_root, out, device, models=None):
    cfg = dict(cfg, layer=12, phi="revnet8", position="last")
    assert cfg["revnet_blocks"] == 8
    exp = NameExperiment(out, cfg, device, data_root, models)
    exp.write_config()
    return exp


def tmap_tasks(exp):
    n = len(exp.cfg["seeds"])
    return ([("joint", split, k) for split in ie.SPLITS for k in range(n)]
            + [("individual", split, i) for split in ie.SPLITS for i in range(n)])


def t_phi(exp, pos_exp, kind, split, cand, idx):
    return (pos_exp if cand == "pos" else exp).load_phi(kind, "trained", split, cand, idx)


def run_tadapt(exp, get_phi, fold_index, splits, sizes, repeats, path, device, extra=None):
    """get_phi(kind, split, cand, idx) -> frozen phi. extra(split, cand, fold_index) -> dict added to every row."""
    c = exp.cfg
    fold = exp.folds()[fold_index]
    held, ref = fold["heldout"], fold["ref"]
    target = exp.get_model("trained", held)
    done = done_keys(path)
    t_start = time.time()
    for split in splits:
        test, a_train, a_dev = exp.pairs(split, "test"), exp.pairs(split, "adapt_train"), exp.pairs(split, "adapt_dev")
        Qp, cost = exp.procrustes_init("trained", ref, held, split)
        for cand in CANDS:
            common = dict(fold=fold_index, heldout=held, ref=ref, sources=fold["sources"], split=split, trained_on=cand,
                          **(extra(split, cand, fold_index) if extra else {}))
            key = f"{split}|{cand}|individual_self"
            if key not in done:
                phi = get_phi("individual", split, cand, c["seeds"].index(held))
                write_row(path, dict(eval4(phi, (target, ie.IdentityAlignment()), test, c, device), key=key,
                                     condition="individual_self", adapt_examples=None, repeat=0, **common))
            phis = {"shared_joint": get_phi("joint", split, cand, fold_index), "no_shared": ie.IdentityMap()}
            for cond in T_CONDS:
                phi = phis[cond]
                for n in sizes:
                    for r in range(1 if n == 0 else repeats):
                        key = f"{split}|{cand}|{cond}|n{n}|r{r}"
                        if key in done:
                            continue
                        t0 = time.time()
                        n_eff = min(n, len(a_train))
                        perm = torch.randperm(len(a_train), generator=torch.Generator().manual_seed(
                            ie.stable_seed("tsubset", fold_index, split, cand, cond, n, r)))
                        align, best_update, dev_loss, moved = exp.adapt(
                            phi, target, Qp, a_train.subset(perm), a_dev, cand, n_eff,
                            ie.stable_seed("tadapt", fold_index, split, cand, cond, n, r))
                        res = eval4(phi, (target, align), test, c, device)
                        write_row(path, dict(res, key=key, condition=cond, adapt_examples=n, adapt_used=n_eff,
                                             repeat=r, adapt_best_update=best_update, adapt_dev_loss=dev_loss,
                                             distance_from_init=moved, procrustes_cost=cost,
                                             seconds=time.time() - t0, **common))
                log(f"fold {fold_index} {split} {cand} done ({time.time() - t_start:.0f}s so far)")


def smoke6(device):
    torch.set_float32_matmul_precision(ie.SMOKE["matmul_precision"])
    out = ROOT / "results" / "ioi_ioname_transfer_smoke"
    if out.exists():
        shutil.rmtree(out)
    cfg = dict(ie.SMOKE)
    exp = transfer_exp(cfg, SMOKE_ROOT, out, device)
    pos_exp = ie.Experiment(SMOKE_ROOT, cfg, device, offline=True)
    t = time.time()
    tasks = tmap_tasks(exp)
    assert len(tasks) == 4 * len(cfg["seeds"]), tasks
    for kind, split, idx in [("joint", "default", 0), ("individual", "default", 0)]:
        exp.train_map(kind, "trained", "name", split, idx)
    log(f"check A (C_name joint and individual maps) passed ({time.time() - t:.0f}s)")

    t = time.time()
    path = out / "adapt" / "fold0.jsonl"
    sizes = [0, cfg["adapt_sizes"][-1]]
    run_tadapt(exp, lambda *a: t_phi(exp, pos_exp, *a), 0, ["default"], sizes, 2, path, device)
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    per_cand = 1 + len(T_CONDS) * (1 + 2)
    assert len(rows) == 2 * per_cand, len(rows)
    orig = {json.loads(x)["key"]: json.loads(x) for x in (SMOKE_ROOT / "adapt" / "fold0_trained.jsonl").read_text().splitlines()}
    want = orig["trained|default|pos|shared_joint|procrustes|n0"]["procrustes_cost"]
    got = next(r for r in rows if r["condition"] == "shared_joint")["procrustes_cost"]
    assert abs(got - want) <= 1e-4 * max(abs(want), 1.0), (got, want)
    log(f"Procrustes cost {got:.6f} = ioi_eval {want:.6f}")
    held = exp.folds()[0]["heldout"]
    phi = t_phi(exp, pos_exp, "individual", "default", "pos", cfg["seeds"].index(held))
    again = ie.evaluate(phi, (exp.get_model("trained", held), ie.IdentityAlignment()),
                        ie.Experiment.pairs(exp, "default", "test"), "pos", cfg, device)
    assert abs(again["iia"] - orig["trained|default|pos|individual_self"]["iia"]) < 0.02
    log(f"C_pos individual_self (2-way, all pairs) {again['iia']:.4f} = ioi_eval "
        f"{orig['trained|default|pos|individual_self']['iia']:.4f}")
    for r in rows:
        log(f"{r['key']}: pos iia4 {r['pos_iia4']:.3f}, name iia4 {r['name_iia4']:.3f}, {r.get('seconds', 0):.1f}s")
    run_tadapt(exp, lambda *a: t_phi(exp, pos_exp, *a), 0, ["default"], sizes, 2, path, device)
    assert len(path.read_text().splitlines()) == 2 * per_cand
    log(f"check B (adaptation rows, reused C_pos maps, Procrustes, resume) passed ({time.time() - t:.0f}s)")
    log(f"smoke6 passed; peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


# --- Phase 7: both candidates retrained with stabilised map training, best of S_RESTARTS by source dev loss
S_OUT = ROOT / "results" / "ioi_ioname_stable"
S_RESTARTS = 3
S_PASS = 0.95  # a map counts as fitted when its 2-way test IIA is >= S_PASS on every source model
S_TRAIN = dict(map_lr=3e-4, map_clip=1.0, map_rollback=0.5, map_max_rollbacks=4)


def stable_exps(cfg, data_root, out, device, restarts=S_RESTARTS, train=S_TRAIN):  # one root per restart
    exps = []
    for r in range(restarts):
        models = exps[0]._models if exps else None
        exps.append(transfer_exp(dict(cfg, **train, map_restart=r), data_root, Path(out) / f"r{r}", device, models))
    return exps


def smap_tasks(exp, restarts=S_RESTARTS):
    return [(r, cand, kind, split, idx) for r in range(restarts) for cand in CANDS
            for kind, split, idx in tmap_tasks(exp)]


def s_select(exps, items):
    """items: (cand, kind, split, idx). Choose the restart with the lowest source dev loss (no held-out data)."""
    out = {}
    for cand, kind, split, idx in items:
        runs = []
        for e in exps:
            ck = torch.load(e.map_path(kind, "trained", split, cand, idx), map_location="cpu", weights_only=False)
            i = ck["info"]
            runs.append(dict(restart=i["restart"], best_dev_loss=i["best_dev_loss"], best_update=i["best_update"],
                             updates=i["updates"], rollbacks=i["rollbacks"], seeds=i["seeds"],
                             test_iia=[t["iia"] for t in ck["test"]]))
        best = min(runs, key=lambda x: x["best_dev_loss"])
        out[(cand, kind, split, idx)] = dict(cand=cand, kind=kind, split=split, idx=idx, chosen=best["restart"],
                                             min_source_iia=min(best["test_iia"]),
                                             passed=min(best["test_iia"]) >= S_PASS, runs=runs)
    return out


def run_sadapt(exps, sel, fold_index, splits, sizes, repeats, path, device):
    exp = exps[0]
    held_idx = exp.cfg["seeds"].index(exp.folds()[fold_index]["heldout"])

    def get_phi(kind, split, cand, idx):
        return exps[sel[(cand, kind, split, idx)]["chosen"]].load_phi(kind, "trained", split, cand, idx)

    def extra(split, cand, k):
        j, i = sel[(cand, "joint", split, k)], sel[(cand, "individual", split, held_idx)]
        return dict(joint_restart=j["chosen"], joint_passed=j["passed"], joint_min_source_iia=j["min_source_iia"],
                    indiv_restart=i["chosen"], indiv_passed=i["passed"], indiv_source_iia=i["min_source_iia"])
    run_tadapt(exp, get_phi, fold_index, splits, sizes, repeats, path, device, extra)


def smoke7(device):
    torch.set_float32_matmul_precision(ie.SMOKE["matmul_precision"])
    out = ROOT / "results" / "ioi_ioname_stable_smoke"
    if out.exists():
        shutil.rmtree(out)
    cfg = dict(ie.SMOKE)
    item = ("individual", "trained", "name", "default", 0)

    t = time.time()  # A: without the new keys, train_map behaves as in phase 6
    e6 = transfer_exp(cfg, SMOKE_ROOT, out / "phase6", device)
    e6.train_map(*item)
    new = torch.load(e6.map_path("individual", "trained", "default", "name", 0), map_location="cpu", weights_only=False)
    old_path = ROOT / "results" / "ioi_ioname_transfer_smoke" / "maps" / "trained" / "default" / "name" / "individual_0.pt"
    old = torch.load(old_path, map_location="cpu", weights_only=False)
    log(f"best update {new['info']['best_update']} (phase 6 smoke: {old['info']['best_update']})")
    assert abs(new["test"][0]["iia"] - old["test"][0]["iia"]) < 0.02, (new["test"][0]["iia"], old["test"][0]["iia"])
    assert new["info"]["rollbacks"] == 0 and new["info"]["map_clip"] == 0 and new["info"]["map_rollback"] is None
    log(f"check A (defaults unchanged: best update {new['info']['best_update']}, test IIA {new['test'][0]['iia']:.4f} "
        f"vs phase 6 {old['test'][0]['iia']:.4f}) passed ({time.time() - t:.0f}s)")

    t = time.time()  # B: a negative margin forces a rollback at every check after the first best
    eb = transfer_exp(dict(cfg, **S_TRAIN, map_rollback=-100.0, map_max_rollbacks=2, map_max_epochs=4), SMOKE_ROOT,
                      out / "rollback", device, e6._models)
    eb.train_map("joint", "trained", "name", "default", 0)  # joint: the source alignments are restored too
    ck = torch.load(eb.map_path("joint", "trained", "default", "name", 0), map_location="cpu", weights_only=False)
    assert ck["info"]["rollbacks"] == 3, ck["info"]["rollbacks"]
    lrs = [h["lr"] for h in ck["info"]["history"]]
    assert abs(lrs[-1] - S_TRAIN["map_lr"] / 8) < 1e-12 or lrs[-1] < S_TRAIN["map_lr"] / 8, lrs
    assert ck["info"]["best_update"] == ck["info"]["history"][0]["update"], ck["info"]
    log(f"check B (rollback: 3 rollbacks, lr {lrs}, stopped at update {ck['info']['updates']}) passed "
        f"({time.time() - t:.0f}s)")

    t = time.time()  # C: restarts, selection, adaptation rows, resume
    exps = stable_exps(cfg, SMOKE_ROOT, out, device, restarts=2)
    exps[0]._models = e6._models
    for e in exps[1:]:
        e._models = e6._models
    items = [(cand, kind, "default", 0) for cand in CANDS for kind in ["joint", "individual"]]
    for e in exps:
        for cand, kind, split, idx in items:
            e.train_map(kind, "trained", cand, split, idx)
    p0 = torch.load(exps[0].map_path("joint", "trained", "default", "name", 0), map_location="cpu", weights_only=False)
    p1 = torch.load(exps[1].map_path("joint", "trained", "default", "name", 0), map_location="cpu", weights_only=False)
    assert p0["info"]["restart"] == 0 and p1["info"]["restart"] == 1
    k0 = next(iter(p0["phi"]))
    assert not torch.equal(p0["phi"][k0], p1["phi"][k0]), "restarts start from the same phi"
    sel = s_select(exps, items)
    for v in sel.values():
        log(f"select {v['cand']} {v['kind']}: restart {v['chosen']}, dev losses "
            f"{[round(r['best_dev_loss'], 4) for r in v['runs']]}, source IIA {[r['test_iia'] for r in v['runs']]}")
        assert v["chosen"] == min(v["runs"], key=lambda r: r["best_dev_loss"])["restart"]
    path = out / "adapt" / "fold0.jsonl"
    sizes = [0, cfg["adapt_sizes"][-1]]
    run_sadapt(exps, sel, 0, ["default"], sizes, 2, path, device)
    per_cand = 1 + len(T_CONDS) * (1 + 2)
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert len(rows) == 2 * per_cand and all("joint_passed" in r for r in rows), len(rows)
    for r in rows:
        log(f"{r['key']}: pos iia4 {r['pos_iia4']:.3f}, name iia4 {r['name_iia4']:.3f}, joint restart {r['joint_restart']}")
    run_sadapt(exps, sel, 0, ["default"], sizes, 2, path, device)
    assert len(path.read_text().splitlines()) == 2 * per_cand
    log(f"check C (restarts differ, selection by dev loss, adaptation rows, resume) passed ({time.time() - t:.0f}s)")
    log(f"smoke7 passed; peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


def write_selection(sel, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(list(sel.values()), indent=1))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["smoke5", "identity", "maps", "smoke6", "tmaps", "tadapt", "smoke7", "smaps", "sadapt"])
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=1)
    parser.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    parser.add_argument("--restarts", type=int, default=S_RESTARTS)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this on a GPU node")
    device = torch.device("cuda")
    torch.set_num_threads(max(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), 1))
    log(f"stage={args.stage} task={args.task}/{args.num_tasks} layers={args.layers} "
        f"device={torch.cuda.get_device_name(device)}")
    if args.stage == "smoke5":
        smoke5(device)
    elif args.stage == "identity":
        cfg = idg.eval_cfg()
        run_identity(cfg, EVAL_ROOT, OUT, args.layers, ["trained", "random"], OUT / "eval" / "identity.jsonl", device)
    elif args.stage == "smoke6":
        smoke6(device)
    elif args.stage == "smoke7":
        smoke7(device)
    elif args.stage in ("smaps", "sadapt"):
        exps = stable_exps(idg.eval_cfg(), EVAL_ROOT, S_OUT, device, args.restarts)
        if args.stage == "smaps":
            tasks = smap_tasks(exps[0], args.restarts)[args.task::args.num_tasks]
            log(f"{len(tasks)} maps: {tasks}")
            for r, cand, kind, split, idx in tasks:
                exps[r].train_map(kind, "trained", cand, split, idx)
        else:
            sel = s_select(exps, [(cand, kind, split, idx) for cand in CANDS for kind, split, idx in tmap_tasks(exps[0])])
            write_selection(sel, S_OUT / "selection.json")
            run_sadapt(exps, sel, args.task, ie.SPLITS, T_SIZES, T_REPEATS, S_OUT / "adapt" / f"fold{args.task}.jsonl",
                       device)
    elif args.stage in ("tmaps", "tadapt"):
        cfg = idg.eval_cfg()
        exp = transfer_exp(cfg, EVAL_ROOT, T_OUT, device)
        if args.stage == "tmaps":
            tasks = tmap_tasks(exp)[args.task::args.num_tasks]
            log(f"{len(tasks)} maps: {tasks}")
            for kind, split, idx in tasks:
                exp.train_map(kind, "trained", "name", split, idx)
        else:
            pos_exp = ie.Experiment(EVAL_ROOT, cfg, device, offline=True)
            run_tadapt(exp, lambda *a: t_phi(exp, pos_exp, *a), args.task, ie.SPLITS, T_SIZES, T_REPEATS,
                       T_OUT / "adapt" / f"fold{args.task}.jsonl", device)
    else:
        cfg = idg.eval_cfg()
        tasks = map_tasks(args.layers)[args.task::args.num_tasks]
        log(f"{len(tasks)} maps: {tasks}")
        run_maps(cfg, EVAL_ROOT, OUT, tasks, OUT / "eval" / f"maps_task{args.task}of{args.num_tasks}.jsonl", device)
    log(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
