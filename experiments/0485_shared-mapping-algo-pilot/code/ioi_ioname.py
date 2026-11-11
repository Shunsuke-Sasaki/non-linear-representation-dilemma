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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["smoke5", "identity", "maps"])
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=1)
    parser.add_argument("--layers", type=int, nargs="+", default=LAYERS)
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
    else:
        cfg = idg.eval_cfg()
        tasks = map_tasks(args.layers)[args.task::args.num_tasks]
        log(f"{len(tasks)} maps: {tasks}")
        run_maps(cfg, EVAL_ROOT, OUT, tasks, OUT / "eval" / f"maps_task{args.task}of{args.num_tasks}.jsonl", device)
    log(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
