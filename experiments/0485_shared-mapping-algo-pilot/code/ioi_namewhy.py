#!/usr/bin/env python
"""Experiment 0485, IOI: why do some Pythia seeds fit namesplit C_name (seed 1) and others not (seeds 2-4)?

Observed before this script (phase 7, layer 12, END, revnet8): namesplit C_name, 4-way IIA of the shared phi adapted
to each held-out model (n=4096) falls in two groups, {1, 5, 6, 9} at .47-.53 and {2, 3, 4, 7, 8} at .25-.28; the
individual maps fit seed 1 (.95) and partly seed 5, not seeds 2-4. Phase 5 (seed 1 only): full END patching at L12
moves the source IO in .4-1% of pairs (L18: 47%); a rotation phi fails namesplit C_name (.23), revnet1 already .89.

Hypothesis H1 (name content at the site): at L12 END, the models of the first group carry the IO name in a form that
holds for names never seen in training, the second group does not (they may carry it only at later layers).
H2 (memorisation): every C_name map fits the training names; only the unseen names separate the seeds.

Measurements (no map training; namesplit data of results/ioi_eval; prompts whose last token is not a lone space):
  A  per trained seed 1-9 (and step0 of seed 1 as a control), per layer 0-23, at END (layer output = residual stream):
     lens     final_layer_norm + embed_out of the residual: rate of IO = argmax over the 100 name tokens
              (on test-name prompts) and IO > S rate.
     probe    ridge regression residual -> unembedding row of the IO token (target "unembed") and -> input embedding
              row (target "embed"), fitted on N_FIT training-name prompts; lambda chosen on eval-name prompts;
              score: IO = nearest (cosine) among the 20 test-name tokens on test-name prompts (chance .05), and
              among the 60 training-name tokens on held-out training-name prompts (chance 1/60).
     patch    full END patching (1024 dims) source -> base at the layer, on N_PATCH disjoint namesplit test pairs;
              rate of each of the four answers (src_io = C_name 4-way IIA without a map).
  B  seeds 1-5: the phase-7 individual namesplit C_name maps (restarts 0-2) and C_pos maps (restart 0), 4-way
     evaluation (eval4) on training-name pairs (first N_TRAIN_PAIRS disjoint train pairs), eval-name pairs (dev)
     and test-name pairs (test).
Predictions, fixed before the run:
  H1 -> at L12, probe (test names) and patch src_io are higher for {1,5,6,9} than {2,3,4,7,8}; Spearman rho across
        the 9 seeds between the L12 probe score and the namesplit C_name transfer > 0.
  H2 -> part B: C_name maps reach >= .9 4-way on training-name pairs for every seed, and stay low on test names
        for seeds 2-4.
  Not H1 -> the L12 measures do not separate the groups (then the difference is not in what END carries linearly).
Output: results/ioi_namewhy/{lens_probe.csv, patch.csv, maps_by_names.jsonl}; summary: ioi_namewhy_summary.py.
"""
import json
import os
import time

import torch

import ioi_diag as idg
import ioi_eval as ie
import ioi_ioname as io

OUT = ie.ROOT / "results" / "ioi_namewhy"
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAP_SEEDS = [1, 2, 3, 4, 5]
RANDOM_SEEDS = [1]  # control: step0 of seed 1 (what the probe reads from an untrained model)
N_FIT, N_SEEN, N_EVAL, N_TEST, N_PATCH, N_TRAIN_PAIRS = 4096, 2048, 2048, 4096, 2048, 4096
LAMBDAS = [1e-1, 1, 10, 100, 1000, 10000]
log = ie.log


def prompts(pool, n, space_id, skip=0):
    last = pool["ids"][torch.arange(len(pool["lengths"])), pool["lengths"] - 1].long()
    ok = torch.nonzero(last != space_id).squeeze(1)
    return ok[skip:skip + n]


def end_states(model, pool, idx, n_layers, batch, device):
    H = idg.hidden_all(model, pool, idx, ["last"], n_layers, batch, device)
    return torch.stack([H[(l, "last")] for l in range(n_layers)])  # [L, N, H]


def nearest_rate(pred, table, tok, cand_toks):
    """IO = nearest (cosine) among cand_toks rows of table."""
    C = table[cand_toks]
    C = C / C.norm(dim=1, keepdim=True)
    pred = pred / (pred.norm(dim=1, keepdim=True) + 1e-9)
    pick = cand_toks[(pred @ C.T).argmax(1)]
    return (pick == tok).float().mean().item()


def ridge(Xtr, Ytr, lam):
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + lam * torch.eye(d, device=Xtr.device, dtype=Xtr.dtype)
    return torch.linalg.solve(A, Xtr.T @ Ytr)


@torch.no_grad()
def part_a(exp, seed, device, state="trained"):
    d = exp.data("namesplit")
    pools, sp = d["pools"], exp.space_id()
    model = exp.get_model(state, seed)
    n_layers = len(model.gpt_neox.layers)
    batch = exp.cfg["eval_batch"]
    sets = {"fit": ("train", prompts(pools["train"], N_FIT, sp)),
            "seen": ("train", prompts(pools["train"], N_SEEN, sp, skip=N_FIT)),
            "eval": ("eval", prompts(pools["eval"], N_EVAL, sp)),
            "test": ("test", prompts(pools["test"], N_TEST, sp))}
    H, tok_io, tok_s = {}, {}, {}
    for k, (pn, idx) in sets.items():
        H[k] = end_states(model, pools[pn], idx, n_layers, batch, device)
        tok_io[k], tok_s[k] = pools[pn]["tok_io"][idx].to(device), pools[pn]["tok_s"][idx].to(device)
    names = {pn: torch.unique(torch.cat([pools[pn]["tok_io"], pools[pn]["tok_s"]])).to(device)
             for pn in ("train", "eval", "test")}
    all_names = torch.unique(torch.cat(list(names.values())))
    W_U = model.embed_out.weight.float()
    tables = {"unembed": W_U.double(), "embed": model.gpt_neox.embed_in.weight.double()}
    ln = model.gpt_neox.final_layer_norm
    rows = []
    for l in range(n_layers):
        r = {"state": state, "seed": seed, "layer": l}
        h = H["test"][l].to(device)
        logits = (ln(h) @ W_U.T)
        r["lens_io_top_names"] = (all_names[logits[:, all_names].argmax(1)] == tok_io["test"]).float().mean().item()
        rows_ = torch.arange(len(h), device=device)
        r["lens_io_gt_s"] = (logits[rows_, tok_io["test"]] > logits[rows_, tok_s["test"]]).float().mean().item()
        X = {k: H[k][l].to(device).double() for k in H}
        mu, sd = X["fit"].mean(0), X["fit"].std(0) + 1e-6
        X = {k: (v - mu) / sd for k, v in X.items()}
        for tname, T in tables.items():
            Y = T[tok_io["fit"]]
            best = None
            for lam in LAMBDAS:
                W = ridge(X["fit"], Y, lam * len(Y))
                s = nearest_rate(X["eval"] @ W, T, tok_io["eval"], names["eval"])
                if best is None or s > best[0]:
                    best = (s, lam, W)
            s_eval, lam, W = best
            r[f"probe_{tname}_lambda"] = lam
            r[f"probe_{tname}_eval"] = s_eval
            r[f"probe_{tname}_test"] = nearest_rate(X["test"] @ W, T, tok_io["test"], names["test"])
            r[f"probe_{tname}_seen"] = nearest_rate(X["seen"] @ W, T, tok_io["seen"], names["train"])
        rows.append(r)
    del H
    patch = patch_sweep(exp, model, n_layers, device)
    return rows, [dict(state=state, seed=seed, **p) for p in patch]


@torch.no_grad()
def patch_sweep(exp, model, n_layers, device):
    pairs = exp.pairs("namesplit", "test")
    pairs = pairs.subset(torch.arange(min(N_PATCH, len(pairs))))
    four = pairs.four()
    batch = exp.cfg["eval_batch"]
    picks = {l: [] for l in range(-1, n_layers)}
    for start in range(0, len(pairs), batch):
        sl = torch.arange(start, min(start + batch, len(pairs)))
        sub = pairs.subset(sl)
        b_ids, b_mask, b_last = ie.batch_tokens(sub.pool, sub.base, device)
        s_ids, s_mask, s_last = ie.batch_tokens(sub.pool, sub.src, device)
        store = idg.capture_all(model, s_ids, s_mask, {"last": s_last}, n_layers)
        f = four[sl].to(device)
        picks[-1].append(ie.last_logits(model, b_ids, b_mask, b_last).gather(1, f).argmax(1).cpu())
        for l in range(n_layers):
            src = store[(l, "last")]
            logits = ie.last_logits(model, b_ids, b_mask, b_last, l, edit=lambda h, s=src: s)
            picks[l].append(logits.gather(1, f).argmax(1).cpu())
    out = []
    for l, v in picks.items():
        v = torch.cat(v)
        out.append({"layer": l, "n": len(v), **{f"pick_{nm}": (v == i).float().mean().item() for i, nm in enumerate(io.FOUR)}})
    return out


def part_b(exps, device, path):
    exp = exps[0]
    sets = {"train_names": exp.pairs("namesplit", "train"), "eval_names": exp.pairs("namesplit", "dev"),
            "test_names": exp.pairs("namesplit", "test")}
    sets["train_names"] = sets["train_names"].subset(torch.arange(N_TRAIN_PAIRS))
    done = io.done_keys(path)
    jobs = [(r, "name") for r in range(len(exps))] + [(0, "pos")]
    for i, seed in enumerate(MAP_SEEDS):
        model = exp.get_model("trained", seed)
        for r, cand in jobs:
            key = f"{seed}|{cand}|r{r}"
            if key in done or not exps[r].map_path("individual", "trained", "namesplit", cand, i).exists():
                continue
            ck = torch.load(exps[r].map_path("individual", "trained", "namesplit", cand, i), map_location="cpu",
                            weights_only=False)
            phi = exps[r].load_phi("individual", "trained", "namesplit", cand, i)
            row = dict(key=key, seed=seed, trained_on=cand, restart=r, best_update=ck["info"]["best_update"],
                       rollbacks=ck["info"]["rollbacks"], best_dev_loss=ck["info"]["best_dev_loss"])
            for nm, pairs in sets.items():
                res = io.eval4(phi, (model, ie.IdentityAlignment()), pairs, exp.cfg, device)
                row.update({f"{nm}_{k}": res[k] for k in ("n", "name_iia4", "name_iia_vocab", "pos_iia4", "pos_iia_vocab")})
            io.write_row(path, row)
            log(f"{key}: C_name 4-way train/eval/test names {row['train_names_name_iia4']:.3f} / "
                f"{row['eval_names_name_iia4']:.3f} / {row['test_names_name_iia4']:.3f}; C_pos "
                f"{row['train_names_pos_iia4']:.3f} / {row['test_names_pos_iia4']:.3f}")
        exp.drop_models()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this on a GPU node")
    device = torch.device("cuda")
    torch.set_num_threads(max(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), 1))
    cfg = idg.eval_cfg()
    exps = io.stable_exps(cfg, io.EVAL_ROOT, io.S_OUT, device, restarts=3)
    exp = exps[0]
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    part_b(exps, device, OUT / "maps_by_names.jsonl")
    log(f"part B done ({time.time() - t0:.0f}s)")
    lens, patch = [], []
    for state, seed in [("trained", s) for s in SEEDS] + [("random", s) for s in RANDOM_SEEDS]:
        t = time.time()
        a, p = part_a(exp, seed, device, state)
        lens += a
        patch += p
        exp.drop_models()
        l12 = a[12]
        log(f"{state} seed {seed}: L12 probe unembed test {l12['probe_unembed_test']:.3f} (seen {l12['probe_unembed_seen']:.3f}), "
            f"embed test {l12['probe_embed_test']:.3f}, lens IO top {l12['lens_io_top_names']:.3f}, "
            f"patch src_io {p[13]['pick_src_io']:.3f} ({time.time() - t:.0f}s)")
    idg.write_csv(OUT / "lens_probe.csv", lens)
    idg.write_csv(OUT / "patch.csv", patch)
    log(f"done ({time.time() - t0:.0f}s); peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
