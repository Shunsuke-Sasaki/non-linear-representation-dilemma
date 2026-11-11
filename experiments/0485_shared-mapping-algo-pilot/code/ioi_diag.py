#!/usr/bin/env python
"""Experiment 0485, IOI follow-up diagnostics on the Pythia-410m seeds (phase 1).

Questions (after results/ioi_eval):
  1. Where does seed 3 carry the position variable?  -> activation patching sweep over all layers and the
     END / S2 / S1 / IO tokens (full-vector patching from the source prompt, no map), all trained seeds.
  2. Is seed 4's large Procrustes residual a difference of shape or of scale?  -> layer-12 END statistics and
     pairwise raw / relative / centred-and-normalised Procrustes residuals and linear CKA.
  3. Preparation for "replace seeds 3 and 4 among the sources": download seeds 6-9 (step143000), measure their
     IOI accuracy, and train their individual C_pos map (layer 12, END, default split) to pick good seeds.

Stages:
  smoke                 checks on results/ioi_smoke (token positions, patching vs phi, an S2-position map)
  diag                  questions 1-2 and the accuracy of seeds 1-9 -> results/ioi_diag/diag
  extra-maps --task i   individual C_pos map of seed 6+i -> results/ioi_diag/extra_seeds
Phase 2 (after the phase-1 results):
  smoke2                checks of the phase-2 code paths on results/ioi_smoke
  loc-maps --task 0-5   individual C_pos maps at other layers / positions (LOC_MAPS) -> results/ioi_diag/loc
  replace-maps --task 0-11   joint maps with seeds 3, 4 replaced by 6, 7 among the sources -> results/ioi_diag/replace34
  replace-adapt --task 0-2   shared_joint adaptation on held-out seeds 1, 2, 5 (same protocol as ioi_eval)
Phase 3 (does seed 4 partly run the algorithm?):
  smoke3                checks of the phase-3 code paths on results/ioi_smoke
  seed4 --task 0        logit-margin patching + linear probes, trained 1, 3, 4 / random 1, 4 -> results/ioi_diag/seed4
  seed4 --task 1        IIA restricted to pairs the held-out model answers correctly (held-out 4 and 1)
Reuses ioi_eval.py (data in results/ioi_eval/data, same pairs, same map training).
"""
import argparse
import csv
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import torch

import ioi_eval as ie

ROOT = ie.ROOT
EVAL_ROOT = ROOT / "results" / "ioi_eval"
SMOKE_ROOT = ROOT / "results" / "ioi_smoke"
EXTRA_SEEDS = [6, 7, 8, 9]
log = ie.log


class DiagExperiment(ie.Experiment):
    """ioi_eval.Experiment that reads the data of another root and can override the folds."""

    def __init__(self, root, cfg, device, data_root, offline=True, folds=None):
        super().__init__(root, cfg, device, offline)
        self.data_root, self._folds = Path(data_root), folds

    def data(self, split):
        if split not in self._data:
            self._data[split] = torch.load(self.data_root / "data" / f"{split}.pt", weights_only=False)
        return self._data[split]

    def folds(self):
        return self._folds if self._folds is not None else super().folds()

    def pairs(self, split, name):
        p = super().pairs(split, name)
        which = self.cfg.get("position", "last")
        if which == "last":
            return p
        at = ie.token_positions(p.pool, which)
        return p.subset(torch.nonzero((at[p.base] >= 0) & (at[p.src] >= 0)).squeeze(1))


def download_trained(cfg, seed):
    from huggingface_hub import list_repo_files, snapshot_download
    name, rev = cfg["model_name"].format(seed=seed), ie.STATES["trained"]
    files = list_repo_files(name, revision=rev)
    weights = "*.safetensors" if any(f.endswith(".safetensors") for f in files) else "*.bin"
    snapshot_download(name, revision=rev, allow_patterns=["*.json", "*.txt", weights])
    log(f"downloaded {name}@{rev}")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


##############################################################################
##  Activation patching                                                     ##
##############################################################################

@torch.no_grad()
def capture_all(model, ids, mask, at, n_layers):
    """Outputs of every layer at the given token indices: {(layer, position): [B, H]}."""
    store, handles = {}, []
    rows = torch.arange(ids.shape[0], device=ids.device)
    for layer in range(n_layers):
        def hook(module, inp, output, layer=layer):
            h = ie._out_tensor(output)
            for p, a in at.items():
                store[(layer, p)] = h[rows, a].float()
        handles.append(ie.layer_module(model, layer).register_forward_hook(hook))
    try:
        model.gpt_neox(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return store


def valid_pairs(pairs, n):
    """The first n pairs whose base and source both have all four token positions."""
    ok = torch.ones(len(pairs), dtype=torch.bool)
    for p in ie.POSITIONS:
        at = ie.token_positions(pairs.pool, p)
        ok &= (at[pairs.base] >= 0) & (at[pairs.src] >= 0)
    keep = torch.nonzero(ok).squeeze(1)
    return pairs.subset(keep[:n]), ok.float().mean().item()


@torch.no_grad()
def patching_sweep(model, pairs, n_layers, batch, device):
    """IIA of full-vector patching (source -> base) at each (layer, position), scored with both candidates."""
    labels = {c: pairs.labels(c) for c in ie.CANDIDATES}
    at_all = {p: ie.token_positions(pairs.pool, p) for p in ie.POSITIONS}
    correct = defaultdict(list)
    for start in range(0, len(pairs), batch):
        sel = torch.arange(start, min(start + batch, len(pairs)))
        sub = pairs.subset(sel)
        b_ids, b_mask, b_last = ie.batch_tokens(sub.pool, sub.base, device)
        s_ids, s_mask, _ = ie.batch_tokens(sub.pool, sub.src, device)
        b_at = {p: at_all[p][sub.base].to(device) for p in ie.POSITIONS}
        s_at = {p: at_all[p][sub.src].to(device) for p in ie.POSITIONS}
        store = capture_all(model, s_ids, s_mask, s_at, n_layers)
        rows = torch.arange(len(sel), device=device)

        def score(key, logits):
            for c in ie.CANDIDATES:
                true, false = labels[c][0][sel].to(device), labels[c][1][sel].to(device)
                correct[key + (c,)].append((logits[rows, true] > logits[rows, false]).cpu())

        score((-1, "none"), ie.last_logits(model, b_ids, b_mask, b_last))
        for layer in range(n_layers):
            for p in ie.POSITIONS:
                src = store[(layer, p)]
                score((layer, p), ie.last_logits(model, b_ids, b_mask, b_last, layer, edit=lambda h, s=src: s,
                                                 at=b_at[p]))
    out = []
    for (layer, p, c), v in correct.items():
        v = torch.cat(v).float()
        changed = labels[c][2]
        out.append({"layer": layer, "position": p, "cand": c, "iia": v.mean().item(),
                    "iia_label_changed": v[changed].mean().item(), "iia_label_unchanged": v[~changed].mean().item()})
    return out


##############################################################################
##  Representation statistics                                               ##
##############################################################################

def seed_stats(H):
    H = H.double()
    absmean = H.abs().mean(0)
    top = torch.topk(absmean, 3)
    centred = H - H.mean(0)
    var = centred.pow(2).mean(0)
    return {"mean_norm": H.norm(dim=1).mean().item(), "mean_vector_norm": H.mean(0).norm().item(),
            "centred_rms_norm": centred.norm(dim=1).pow(2).mean().sqrt().item(),
            "top_dims": " ".join(map(str, top.indices.tolist())),
            "top_absmean": " ".join(f"{x:.2f}" for x in top.values.tolist()),
            "median_absmean": absmean.median().item(), "top_dim_variance_share": (var.max() / var.sum()).item()}


def pair_stats(H_ref, H_m):
    A, B = H_ref.double(), H_m.double()
    Q, raw = ie.procrustes(A.float(), B.float())  # exactly the residual reported as procrustes_cost
    Q = Q.double()
    rel = ((B @ Q - A) ** 2).sum() / (A ** 2).sum()
    a, b = A - A.mean(0), B - B.mean(0)
    a, b = a / a.norm(), b / b.norm()
    U, _, Vt = torch.linalg.svd(b.T @ a)
    shape = ((b @ (U @ Vt) - a) ** 2).sum()
    cka = (b.T @ a).norm() ** 2 / ((a.T @ a).norm() * (b.T @ b).norm())
    return {"procrustes_cost": raw, "relative_residual": rel.item(), "shape_distance": shape.item(),
            "linear_cka": cka.item()}


@torch.no_grad()
def hidden_set(model, pool, idx, layer, batch, device):
    hs = []
    for start in range(0, len(idx), batch):
        ids, mask, last = ie.batch_tokens(pool, idx[start:start + batch], device)
        hs.append(ie.hidden_last(model, ids, mask, last, layer))
    return torch.cat(hs)


##############################################################################
##  Stages                                                                  ##
##############################################################################

def run_diag(device, out, cfg, data_root, seeds_trained, seeds_random, n_patch, splits):
    exp = DiagExperiment(out / "_models", cfg, device, data_root, offline=True)
    n_layers = None
    acc_rows, patch_rows, stat_rows, pair_rows, coverage = [], [], [], [], {}
    H = {}
    for state, seeds in (("trained", seeds_trained), ("random", seeds_random)):
        for seed in seeds:
            t0 = time.time()
            model = exp.get_model(state, seed)
            n_layers = len(model.gpt_neox.layers)
            if state == "trained":
                for split in splits:
                    test = exp.pairs(split, "test")
                    idx = test.base.unique()[:cfg["test"]]
                    acc_rows.append({"seed": seed, "split": split, "task_accuracy": ie.task_accuracy(
                        model, exp.data(split)["pools"]["test"], idx, cfg, device)})
                pairs, cov = valid_pairs(exp.pairs("default", "test"), n_patch)
                coverage[seed] = {"valid_fraction": cov, "n_pairs": len(pairs)}
                for r in patching_sweep(model, pairs, n_layers, cfg["eval_batch"], device):
                    patch_rows.append({"seed": seed, **r})
            d = exp.data("default")
            H[(state, seed)] = hidden_set(model, d["pools"]["train"], d["sets"]["cal"]["base"], cfg["layer"],
                                          cfg["eval_batch"], device)
            stat_rows.append({"state": state, "seed": seed, **seed_stats(H[(state, seed)])})
            exp.drop_models()
            log(f"{state} seed{seed} done ({time.time() - t0:.0f}s)")
    for state, seeds in (("trained", seeds_trained), ("random", seeds_random)):
        for ref in seeds:
            for seed in seeds:
                if seed != ref:
                    pair_rows.append({"state": state, "ref": ref, "seed": seed,
                                      **pair_stats(H[(state, ref)], H[(state, seed)])})
    write_csv(out / "accuracy.csv", acc_rows)
    write_csv(out / "patching.csv", patch_rows)
    write_csv(out / "repr_seeds.csv", stat_rows)
    write_csv(out / "repr_pairs.csv", pair_rows)
    (out / "patching_pairs.json").write_text(json.dumps(coverage, indent=2))
    return acc_rows, patch_rows, stat_rows, pair_rows


def smoke(device):
    torch.set_float32_matmul_precision(ie.SMOKE["matmul_precision"])
    out = ROOT / "results" / "ioi_diag_smoke"
    if out.exists():
        shutil.rmtree(out)
    cfg = dict(ie.SMOKE)
    exp = DiagExperiment(out / "_models", cfg, device, SMOKE_ROOT, offline=True)
    tok = exp.tokenizer()

    # 1. token positions on hand-made prompts
    prompts, meta = ie.parse_sentences([
        "Then, Mary and John went to the store. John gave a apple to Mary",
        "Then, John and Mary went to the store. John gave a apple to Mary",
        "Mary and John went to the store. John gave a apple to Mary",  # sentence-initial IO: other token
    ], tok)
    pool = ie.make_pool(prompts, meta, tok)
    expect = {"last": [" to", " to", " to"], "s1": [" John"] * 3, "s2": [" John"] * 3, "io": [" Mary", " Mary", None]}
    for p, words in expect.items():
        at = ie.token_positions(pool, p)
        for row, word in enumerate(words):
            if word is None:
                assert at[row] == -1, (p, row, at[row])
            else:
                got = tok.decode([int(pool["ids"][row, at[row]])])
                assert got == word, (p, row, got, word)
    assert ie.token_positions(pool, "s1")[0] < ie.token_positions(pool, "s2")[0]
    assert ie.token_positions(pool, "io")[0] < ie.token_positions(pool, "s1")[0]  # ABBA: IO first
    assert ie.token_positions(pool, "io")[1] > ie.token_positions(pool, "s1")[1]  # BABA: IO second
    log("check 1 (token positions) passed")

    model = exp.get_model("trained", 1)
    n_layers = len(model.gpt_neox.layers)
    pairs, cov = valid_pairs(exp.pairs("default", "test"), 64)
    log(f"smoke pairs with all positions: {cov:.3f} of the test pairs, using {len(pairs)}")

    # 2. patching the last layer at END reproduces the source's own logits
    b_ids, b_mask, b_last = ie.batch_tokens(pairs.pool, pairs.base, device)
    s_ids, s_mask, s_last = ie.batch_tokens(pairs.pool, pairs.src, device)
    store = capture_all(model, s_ids, s_mask, {"last": s_last}, n_layers)
    patched = ie.last_logits(model, b_ids, b_mask, b_last, n_layers - 1, edit=lambda h: store[(n_layers - 1, "last")])
    own = ie.last_logits(model, s_ids, s_mask, s_last)
    err = (patched - own).abs().max().item()
    assert err < 5e-2 * own.abs().max().item(), err
    log(f"check 2 (last-layer END patch = source logits) passed, max abs diff {err:.4f}")

    # 3. phi = identity on all dims at S2 equals full-vector patching at S2 (both code paths agree)
    layer = cfg["layer"]
    cfg_s2 = dict(cfg, position="s2", n_var_dims=cfg["hidden"])
    via_phi = ie.intervened_logits(ie.IdentityMap(), (model, ie.IdentityAlignment()), pairs, cfg_s2, device)
    at = ie.token_positions(pairs.pool, "s2")
    store = capture_all(model, s_ids, s_mask, {"s2": at[pairs.src].to(device)}, n_layers)
    via_patch = ie.last_logits(model, b_ids, b_mask, b_last, layer, edit=lambda h: store[(layer, "s2")],
                               at=at[pairs.base].to(device))
    err = (via_phi - via_patch).abs().max().item()
    assert err < 1e-3 * via_patch.abs().max().item(), err
    # and position "last" is the original behaviour
    a = ie.intervened_logits(ie.IdentityMap(), (model, ie.IdentityAlignment()), pairs, dict(cfg, position="last"), device)
    b = ie.intervened_logits(ie.IdentityMap(), (model, ie.IdentityAlignment()), pairs, cfg, device)
    assert (a - b).abs().max().item() <= 1e-4 * b.abs().max().item()
    log(f"check 3 (S2 intervention via phi = patching; default position unchanged) passed, diff {err:.2e}")

    # 4. the sweep and the statistics run
    rows = patching_sweep(model, pairs, n_layers, cfg["eval_batch"], device)
    assert len(rows) == (1 + n_layers * len(ie.POSITIONS)) * len(ie.CANDIDATES)
    exp.drop_models()
    run_diag(device, out / "diag", cfg, SMOKE_ROOT, [1, 2], [1], 64, ["default"])
    log("check 4 (diag stage on smoke data) passed")

    # 5. an individual map at the S2 position trains and is saved
    m_exp = DiagExperiment(out / "s2_map", dict(cfg, seeds=[1], position="s2"), device, SMOKE_ROOT, offline=True)
    m_exp.write_config()
    t = time.time()
    m_exp.train_map("individual", "trained", "pos", "default", 0)
    assert m_exp.map_path("individual", "trained", "default", "pos", 0).exists()
    log(f"check 5 (S2-position map) passed ({time.time() - t:.0f}s)")
    log(f"smoke passed; peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


def diag(device):
    cfg = json.loads((EVAL_ROOT / "config.json").read_text())
    torch.set_float32_matmul_precision(cfg["matmul_precision"])
    for seed in EXTRA_SEEDS:
        download_trained(cfg, seed)
    out = ROOT / "results" / "ioi_diag" / "diag"
    t = time.time()
    run_diag(device, out, cfg, EVAL_ROOT, cfg["seeds"] + EXTRA_SEEDS, cfg["seeds"], 2048, ie.SPLITS)
    log(f"diag finished ({time.time() - t:.0f}s); peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


def extra_maps(device, task):
    cfg = dict(json.loads((EVAL_ROOT / "config.json").read_text()), seeds=EXTRA_SEEDS)
    torch.set_float32_matmul_precision(cfg["matmul_precision"])
    download_trained(cfg, EXTRA_SEEDS[task])
    exp = DiagExperiment(ROOT / "results" / "ioi_diag" / "extra_seeds", cfg, device, EVAL_ROOT, offline=True)
    exp.write_config()
    exp.train_map("individual", "trained", "pos", "default", task)
    log(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


##############################################################################
##  Phase 2                                                                 ##
##############################################################################

# A. Individual C_pos maps of seed 3 (and seed 1 as a control) at the sites suggested by the patching sweep.
LOC_MAPS = [(3, "last", 16), (3, "last", 19), (3, "s2", 13), (1, "last", 13), (1, "last", 19), (1, "s2", 9)]
# B. Sources without seeds 3 and 4: they are replaced by seeds 6 and 7 (chosen by the fixed rule "smallest
# numbers" among the good extra seeds), keeping four sources and the same reference model per fold.
REPLACE_HELDOUT, REPLACE_DROP, REPLACE_ADD = [1, 2, 5], [3, 4], [6, 7]


def loc_exp(cfg, data_root, root, seed, position, layer, device):
    cfg = dict(cfg, seeds=[seed], position=position, layer=layer)
    exp = DiagExperiment(Path(root) / f"seed{seed}_{position}_L{layer}", cfg, device, data_root, offline=True)
    exp.write_config()
    return exp


def replace_folds(seeds, heldouts, drop, add):
    """ioi_eval folds (same indices and reference models) with `drop` replaced by `add` for the given held-outs."""
    out = []
    for held in seeds:
        sources = [s for s in seeds if s != held]
        if held in heldouts:
            sources = [s for s in sources if s not in drop] + list(add)
        out.append({"heldout": held, "sources": sources, "ref": sources[0]})
    return out


def replace_tasks(fold_indices):
    return [(k, cand, split) for k in fold_indices for cand in ie.CANDIDATES for split in ie.SPLITS]


def shared_adapt(exp, fold_index, splits, cands, state="trained"):
    """The shared_joint rows of ioi_eval.Experiment.run_adapt (same inits, seeds and adaptation) for one fold."""
    c = exp.cfg
    fold = exp.folds()[fold_index]
    held, ref = fold["heldout"], fold["ref"]
    out = exp.path("adapt", f"fold{fold_index}_{state}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["key"] for x in out.read_text().splitlines()} if out.exists() else set()
    target = exp.get_model(state, held)
    with out.open("a") as fh:
        for split in splits:
            test = exp.pairs(split, "test")
            a_train, a_dev = exp.pairs(split, "adapt_train"), exp.pairs(split, "adapt_dev")
            Qp, cost = exp.procrustes_init(state, ref, held, split)
            inits = {"procrustes": Qp, "random_orthogonal": ie.random_orthogonal(
                c["hidden"], ie.stable_seed("rand_orth", fold_index, state, split)).to(exp.device)}
            for cand in cands:
                phi = exp.load_phi("joint", state, split, cand, fold_index)
                common = dict(fold=fold_index, heldout=held, ref=ref, sources=" ".join(map(str, fold["sources"])),
                              state=state, split=split, cand=cand, condition="shared_joint")
                for ainit, Q0 in inits.items():
                    for n in c["adapt_sizes"]:
                        key = f"{state}|{split}|{cand}|shared_joint|{ainit}|n{n}"
                        if key in done:
                            continue
                        t0 = time.time()
                        align, best_update, dev_loss, moved = exp.adapt(
                            phi, target, Q0, a_train, a_dev, cand, n,
                            ie.stable_seed("adapt", fold_index, state, split, cand, "shared_joint", ainit, n))
                        res = ie.evaluate(phi, (target, align), test, cand, c, exp.device)
                        fh.write(json.dumps(dict(res, key=key, adapt_init=ainit, adapt_examples=n,
                                                 adapt_best_update=best_update, adapt_dev_loss=dev_loss,
                                                 distance_from_init=moved,
                                                 procrustes_cost=cost if ainit == "procrustes" else None,
                                                 seconds=time.time() - t0, **common)) + "\n")
                        fh.flush()
                log(f"fold {fold_index} {split} {cand} done")


def smoke2(device):
    torch.set_float32_matmul_precision(ie.SMOKE["matmul_precision"])
    out = ROOT / "results" / "ioi_diag_smoke2"
    if out.exists():
        shutil.rmtree(out)
    cfg = dict(ie.SMOKE)
    t = time.time()
    exp = loc_exp(cfg, SMOKE_ROOT, out / "loc", 1, "s2", 5, device)
    exp.train_map("individual", "trained", "pos", "default", 0)
    assert exp.map_path("individual", "trained", "default", "pos", 0).exists()
    assert json.loads(exp.path("config.json").read_text())["layer"] == 5
    log(f"check A (map at another layer and position) passed ({time.time() - t:.0f}s)")
    folds = replace_folds(cfg["seeds"], [1], [3], [3])  # smoke data has seeds 1-3 only: exercises the plumbing
    assert replace_folds([1, 2, 3, 4, 5], REPLACE_HELDOUT, REPLACE_DROP, REPLACE_ADD)[4] == \
        {"heldout": 5, "sources": [1, 2, 6, 7], "ref": 1}
    assert replace_folds([1, 2, 3, 4, 5], REPLACE_HELDOUT, REPLACE_DROP, REPLACE_ADD)[0]["ref"] == 2
    exp = DiagExperiment(out / "replace", cfg, device, SMOKE_ROOT, offline=True, folds=folds)
    exp.write_config()
    t = time.time()
    for split in ie.SPLITS:
        exp.train_map("joint", "trained", "pos", split, 0)
    shared_adapt(exp, 0, ie.SPLITS, ["pos"])
    rows = [json.loads(x) for x in exp.path("adapt", "fold0_trained.jsonl").read_text().splitlines()]
    assert len(rows) == len(ie.SPLITS) * 2 * len(cfg["adapt_sizes"]), len(rows)
    log(f"check B (replaced-source joint map and adaptation) passed ({time.time() - t:.0f}s), {len(rows)} rows")
    log(f"smoke2 passed; peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


def eval_cfg():
    cfg = json.loads((EVAL_ROOT / "config.json").read_text())
    torch.set_float32_matmul_precision(cfg["matmul_precision"])
    return cfg


def replace_exp(cfg, device):
    folds = replace_folds(cfg["seeds"], REPLACE_HELDOUT, REPLACE_DROP, REPLACE_ADD)
    exp = DiagExperiment(ROOT / "results" / "ioi_diag" / "replace34", cfg, device, EVAL_ROOT, offline=True, folds=folds)
    exp.write_config()
    (exp.path("folds.json")).write_text(json.dumps(folds, indent=2))
    return exp


def fold_indices(cfg):
    return [cfg["seeds"].index(h) for h in REPLACE_HELDOUT]


##############################################################################
##  Phase 3: does seed 4 partly run the algorithm?                          ##
##############################################################################
# Seed 4 solves IOI on only ~58% of the prompts and binary patching IIA showed no effect anywhere, yet the shared
# C_pos map transfers to it (0.745 at n=4096; random models 0.50) and C_alpha does not. Three checks:
#   1. patching measured on the logit margin (sensitive to small effects), trained 1, 3, 4 and random 1, 4
#   2. linear probes of Z_pos / Z_alpha at every layer (END, S2), same models
#   3. IIA restricted to the pairs whose base and source the held-out model itself answers correctly
P3_TRAINED, P3_RANDOM, P3_HELDOUT = [1, 3, 4], [1, 4], [4, 1]
P3_POSITIONS = ["last", "s2"]


@torch.no_grad()
def patching_margins(model, pairs, n_layers, batch, device):
    """Full-vector patching (source -> base) at every (layer, position), measured on the logit margin
    m = logit[counterfactual answer] - logit[other name] of each candidate.

    d = m_patched - m_unpatched per pair. effect = mean d on label-changed pairs - mean d on label-unchanged pairs
    (the second term removes generic disruption). effect_norm = effect / mean |logit[IO] - logit[S]| of the
    unpatched base prompts: a patch that fully swaps the answer gives about 2, no effect gives 0.
    """
    cands = ie.CANDIDATES
    labels = {c: pairs.labels(c) for c in cands}
    at_all = {p: ie.token_positions(pairs.pool, p) for p in ie.POSITIONS}
    margins = defaultdict(list)
    base_gap = []
    for start in range(0, len(pairs), batch):
        sel = torch.arange(start, min(start + batch, len(pairs)))
        sub = pairs.subset(sel)
        b_ids, b_mask, b_last = ie.batch_tokens(sub.pool, sub.base, device)
        s_ids, s_mask, _ = ie.batch_tokens(sub.pool, sub.src, device)
        b_at = {p: at_all[p][sub.base].to(device) for p in ie.POSITIONS}
        s_at = {p: at_all[p][sub.src].to(device) for p in ie.POSITIONS}
        store = capture_all(model, s_ids, s_mask, s_at, n_layers)
        rows = torch.arange(len(sel), device=device)

        def record(key, logits):
            margins[key].append(torch.stack(
                [logits[rows, labels[c][0][sel].to(device)] - logits[rows, labels[c][1][sel].to(device)]
                 for c in cands], 1).cpu())

        logits = ie.last_logits(model, b_ids, b_mask, b_last)
        record((-1, "none"), logits)
        base_gap.append((logits[rows, sub.pool["tok_io"][sub.base].to(device)]
                         - logits[rows, sub.pool["tok_s"][sub.base].to(device)]).cpu())
        for layer in range(n_layers):
            for p in ie.POSITIONS:
                src = store[(layer, p)]
                record((layer, p), ie.last_logits(model, b_ids, b_mask, b_last, layer, edit=lambda h, s=src: s,
                                                  at=b_at[p]))
    base_gap = torch.cat(base_gap)
    scale = base_gap.abs().mean().item()
    none = torch.cat(margins[(-1, "none")])
    out = []
    for (layer, p), v in margins.items():
        v = torch.cat(v)
        for j, c in enumerate(cands):
            changed = labels[c][2]
            d = (v[:, j] - none[:, j]).double()
            dc, du = d[changed], d[~changed]
            se = (dc.var() / len(dc) + du.var() / len(du)).sqrt().item()
            effect = (dc.mean() - du.mean()).item()
            out.append({"layer": layer, "position": p, "cand": c, "effect": effect, "effect_se": se,
                        "effect_norm": effect / scale, "effect_norm_se": se / scale,
                        "delta_changed": dc.mean().item(), "delta_unchanged": du.mean().item(),
                        "iia": (v[:, j] > 0).float().mean().item(), "scale": scale,
                        "base_correct": (base_gap > 0).float().mean().item(), "n": len(d)})
    return out


@torch.no_grad()
def hidden_all(model, pool, idx, positions, n_layers, batch, device):
    at_all = {p: ie.token_positions(pool, p) for p in positions}
    out = defaultdict(list)
    for start in range(0, len(idx), batch):
        sel = idx[start:start + batch]
        ids, mask, _ = ie.batch_tokens(pool, sel, device)
        store = capture_all(model, ids, mask, {p: at_all[p][sel].to(device) for p in positions}, n_layers)
        for k, v in store.items():
            out[k].append(v.cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def probe_acc(Xtr, ytr, Xte, yte, device, l2=1e-3, iters=200):
    """L2-regularised logistic regression on standardised features (LBFGS); returns train / test accuracy."""
    Xtr, Xte = Xtr.to(device).float(), Xte.to(device).float()
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ytr, yte = ytr.to(device).float(), yte.to(device).float()
    with torch.enable_grad():
        w = torch.zeros(Xtr.shape[1], device=device, requires_grad=True)
        b = torch.zeros(1, device=device, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Xtr @ w + b, ytr) + l2 * (w ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
    with torch.no_grad():
        acc = lambda X, y: (((X @ w + b) > 0).float() == y).float().mean().item()
        return acc(Xtr, ytr), acc(Xte, yte)


def probe_rows(model, d, n_layers, n_probe, cfg, device):
    pools = d["pools"]
    test_idx = torch.unique(d["sets"]["test"]["base"])
    idx = {"train": d["sets"]["cal"]["base"], "test": test_idx[torch.randperm(len(test_idx),
                                                                            generator=torch.Generator().manual_seed(0))]}
    H, y = {}, {}
    for part, pool_name in (("train", "train"), ("test", "test")):
        pool = pools[pool_name]
        i = idx[part]
        i = i[ie.token_positions(pool, "s2")[i] >= 0][:n_probe]
        H[part] = hidden_all(model, pool, i, P3_POSITIONS, n_layers, cfg["eval_batch"], device)
        y[part] = {c: pool[c][i] for c in ie.CANDIDATES}
    out = []
    for layer in range(n_layers):
        for p in P3_POSITIONS:
            for c in ie.CANDIDATES:
                tr, te = probe_acc(H["train"][(layer, p)], y["train"][c], H["test"][(layer, p)], y["test"][c], device)
                out.append({"layer": layer, "position": p, "cand": c, "train_acc": tr, "test_acc": te,
                            "majority": max(y["test"][c].float().mean().item(), 1 - y["test"][c].float().mean().item()),
                            "n_train": len(y["train"][c]), "n_test": len(y["test"][c])})
    return out


def run_patch_probe(device, out, cfg, data_root, trained, random, n_patch, n_probe):
    exp = DiagExperiment(out / "_models", cfg, device, data_root, offline=True)
    d = exp.data("default")
    patch, probe = [], []
    for state, seeds in (("trained", trained), ("random", random)):
        for seed in seeds:
            t0 = time.time()
            model = exp.get_model(state, seed)
            n_layers = len(model.gpt_neox.layers)
            pairs, _ = valid_pairs(exp.pairs("default", "test"), n_patch)
            patch += [{"state": state, "seed": seed, **r}
                      for r in patching_margins(model, pairs, n_layers, cfg["eval_batch"], device)]
            probe += [{"state": state, "seed": seed, **r} for r in probe_rows(model, d, n_layers, n_probe, cfg, device)]
            exp.drop_models()
            log(f"{state} seed{seed} patching + probes done ({time.time() - t0:.0f}s)")
    write_csv(out / "patch_margins.csv", patch)
    write_csv(out / "probes.csv", probe)
    return patch, probe


@torch.no_grad()
def prompt_correct(model, pool, idx, cfg, device):
    out = []
    for start in range(0, len(idx), cfg["eval_batch"]):
        sel = idx[start:start + cfg["eval_batch"]]
        ids, mask, last = ie.batch_tokens(pool, sel, device)
        logits = ie.last_logits(model, ids, mask, last)
        rows = torch.arange(len(sel), device=device)
        out.append((logits[rows, pool["tok_io"][sel].to(device)] > logits[rows, pool["tok_s"][sel].to(device)]).cpu())
    return torch.cat(out)


@torch.no_grad()
def pair_correct(phi, member, pairs, cand, cfg, device):
    true, false, _ = pairs.labels(cand)
    out = []
    for start in range(0, len(pairs), cfg["eval_batch"]):
        sl = torch.arange(start, min(start + cfg["eval_batch"], len(pairs)))
        logits = ie.intervened_logits(phi, member, pairs.subset(sl), cfg, device)
        rows = torch.arange(len(sl), device=device)
        out.append((logits[rows, true[sl].to(device)] > logits[rows, false[sl].to(device)]).cpu())
    return torch.cat(out)


def restricted_iia(exp, fold_index, cands, ns, out_path, split="default", state="trained"):
    """IIA of the original maps on the held-out model, split by whether the model answers base and source
    correctly itself. Adaptation repeats ioi_eval.run_adapt (same Procrustes init and stable seeds), so the
    overall IIA should reproduce adapt_results.csv."""
    c, device = exp.cfg, exp.device
    fold = exp.folds()[fold_index]
    held, ref = fold["heldout"], fold["ref"]
    target = exp.get_model(state, held)
    test, a_train, a_dev = exp.pairs(split, "test"), exp.pairs(split, "adapt_train"), exp.pairs(split, "adapt_dev")
    uniq = torch.unique(torch.cat([test.base, test.src]))
    own = torch.zeros(len(test.pool["lengths"]), dtype=torch.bool)
    own[uniq] = prompt_correct(target, test.pool, uniq, c, device)
    both, base_ok = own[test.base] & own[test.src], own[test.base]
    Qp, _ = exp.procrustes_init(state, ref, held, split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as fh:
        for cand in cands:
            _, _, changed = test.labels(cand)
            runs = [("individual_self", None)] + [("shared_joint", n) for n in ns] + [("no_shared", ns[-1])]
            for cond, n in runs:
                t0 = time.time()
                if cond == "individual_self":
                    phi = exp.load_phi("individual", state, split, cand, c["seeds"].index(held))
                    align = ie.IdentityAlignment()
                else:
                    phi = exp.load_phi("joint", state, split, cand, fold_index) if cond == "shared_joint" \
                        else ie.IdentityMap()
                    align, _, _, _ = exp.adapt(phi, target, Qp, a_train, a_dev, cand, n, ie.stable_seed(
                        "adapt", fold_index, state, split, cand, cond, "procrustes", n))
                ok = pair_correct(phi, (target, align), test, cand, c, device).float()
                m = lambda mask: ok[mask].mean().item() if mask.any() else float("nan")
                row = {"heldout": held, "ref": ref, "state": state, "split": split, "cand": cand, "condition": cond,
                       "adapt_init": None if n is None else "procrustes", "adapt_examples": n,
                       "iia": ok.mean().item(), "n": len(ok),
                       "own_base_correct": base_ok.float().mean().item(), "own_both_correct": both.float().mean().item(),
                       "iia_both_correct": m(both), "iia_not_both_correct": m(~both),
                       "iia_both_correct_changed": m(both & changed), "iia_both_correct_unchanged": m(both & ~changed),
                       "iia_base_wrong": m(~base_ok), "seconds": time.time() - t0}
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                log(f"held-out {held} {cand} {cond} n={n}: iia {row['iia']:.3f}, both-correct {row['iia_both_correct']:.3f}"
                    f" ({row['own_both_correct']:.3f} of pairs), not {row['iia_not_both_correct']:.3f}")


def smoke3(device):
    torch.set_float32_matmul_precision(ie.SMOKE["matmul_precision"])
    out = ROOT / "results" / "ioi_diag_smoke3"
    if out.exists():
        shutil.rmtree(out)
    cfg = dict(ie.SMOKE)
    # A. probe sanity on synthetic data: a separable direction is found, shuffled labels are not
    g = torch.Generator().manual_seed(0)
    X = torch.randn(2000, 64, generator=g)
    y = X[:, 3] > 0
    tr, te = probe_acc(X[:1000], y[:1000], X[1000:], y[1000:], device)
    assert te > 0.95, te
    _, te_rand = probe_acc(X[:1000], torch.rand(1000, generator=g) > 0.5, X[1000:], y[1000:], device)
    assert te_rand < 0.65, te_rand
    log(f"check A (probe: separable {te:.3f}, shuffled {te_rand:.3f}) passed")
    # B. margin patching and probes run; unpatched rows have zero effect; last-layer END patch moves the margin
    t = time.time()
    patch, probe = run_patch_probe(device, out / "pp", cfg, SMOKE_ROOT, [1], [1], 64, 128)
    none = [r for r in patch if r["layer"] == -1]
    assert all(abs(r["effect"]) < 1e-6 for r in none)
    n_layers = max(r["layer"] for r in patch) + 1
    n_models = 2  # trained 1 and random 1
    assert len(patch) == n_models * (1 + n_layers * len(ie.POSITIONS)) * len(ie.CANDIDATES), len(patch)
    assert len(probe) == n_models * n_layers * len(P3_POSITIONS) * len(ie.CANDIDATES), len(probe)
    # the binary IIA of the margin rows equals the phase-1 patching_sweep on the same pairs
    exp = DiagExperiment(out / "_models", cfg, device, SMOKE_ROOT, offline=True)
    pairs, _ = valid_pairs(exp.pairs("default", "test"), 64)
    ref = {(r["layer"], r["position"], r["cand"]): r["iia"]
           for r in patching_sweep(exp.get_model("trained", 1), pairs, n_layers, cfg["eval_batch"], device)}
    exp.drop_models()
    diff = max(abs(r["iia"] - ref[(r["layer"], r["position"], r["cand"])]) for r in patch if r["state"] == "trained")
    assert diff < 1e-6, diff
    big = max((r for r in patch if r["state"] == "trained" and r["cand"] == "pos"), key=lambda r: r["effect_norm"])
    log(f"check B (margin patching + probes; IIA = patching_sweep) passed ({time.time() - t:.0f}s); largest pos "
        f"effect_norm {big['effect_norm']:.2f} at L{big['layer']} {big['position']}, base_correct {big['base_correct']:.3f}")
    # C. restricted IIA: the overall IIA equals ie.evaluate for the same map and alignment
    exp = ie.Experiment(SMOKE_ROOT, cfg, device, offline=True)
    path = out / "restricted.jsonl"
    restricted_iia(exp, 0, ["pos"], cfg["adapt_sizes"], path)
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert len(rows) == 1 + len(cfg["adapt_sizes"]) + 1, len(rows)
    phi = exp.load_phi("individual", "trained", "default", "pos", 0)
    ref = ie.evaluate(phi, (exp.get_model("trained", 1), ie.IdentityAlignment()), exp.pairs("default", "test"), "pos",
                      cfg, device)["iia"]
    assert abs(rows[0]["iia"] - ref) < 1e-6, (rows[0]["iia"], ref)
    log(f"check C (restricted IIA, {len(rows)} rows; self IIA matches evaluate) passed")
    log(f"smoke3 passed; peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["smoke", "diag", "extra-maps", "smoke2", "loc-maps", "replace-maps",
                                          "replace-adapt", "smoke3", "seed4"])
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this on a GPU node")
    device = torch.device("cuda")
    torch.set_num_threads(max(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), 1))
    log(f"stage={args.stage} task={args.task} device={torch.cuda.get_device_name(device)}")
    if args.stage == "smoke":
        smoke(device)
    elif args.stage == "diag":
        diag(device)
    elif args.stage == "extra-maps":
        extra_maps(device, args.task)
    elif args.stage == "smoke2":
        smoke2(device)
    elif args.stage == "loc-maps":
        seed, position, layer = LOC_MAPS[args.task]
        exp = loc_exp(eval_cfg(), EVAL_ROOT, ROOT / "results" / "ioi_diag" / "loc", seed, position, layer, device)
        exp.train_map("individual", "trained", "pos", "default", 0)
    elif args.stage == "replace-maps":
        cfg = eval_cfg()
        k, cand, split = replace_tasks(fold_indices(cfg))[args.task]
        replace_exp(cfg, device).train_map("joint", "trained", cand, split, k)
    elif args.stage == "replace-adapt":
        cfg = eval_cfg()
        shared_adapt(replace_exp(cfg, device), fold_indices(cfg)[args.task], ie.SPLITS, ie.CANDIDATES)
    elif args.stage == "smoke3":
        smoke3(device)
    else:  # seed4: task 0 = margin patching + probes, task 1 = restricted IIA
        cfg = eval_cfg()
        out = ROOT / "results" / "ioi_diag" / "seed4"
        if args.task == 0:
            run_patch_probe(device, out, cfg, EVAL_ROOT, P3_TRAINED, P3_RANDOM, 2048, 4096)
        else:
            exp = ie.Experiment(EVAL_ROOT, cfg, device, offline=True)  # reads the original maps and calibration
            path = out / "restricted.jsonl"
            path.unlink(missing_ok=True)
            for held in P3_HELDOUT:
                restricted_iia(exp, cfg["seeds"].index(held), ie.CANDIDATES, cfg["adapt_sizes"], path)
    log(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
