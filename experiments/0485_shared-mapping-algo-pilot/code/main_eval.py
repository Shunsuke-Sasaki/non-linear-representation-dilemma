#!/usr/bin/env python
"""Main evaluation of experiment 0485 (llm-jp/experiments#485).

Shared non-linear mapping (RevNet) learned jointly on several source MLPs and transferred to
unseen MLPs, on the distributive-law task with two candidate algorithms
(E = AndOrAnd, F = AndOr). Design: shared_mapp_diffrent_algo/main_eval_design.md.

Reuses das/Shared_Mapping.py (commit 88d6523) and adapts code from the hierarchical-equality
experiment: notebooks/Shared_Mapping.py (folds, models), notebooks/Joint_Alignment_PoC.py
(TrainableOrthogonalAlignment, joint training), notebooks/Alignment_Adaptation_Controls.py
(held-out adaptation) and notebooks/Alignment_Adaptation_Diagnostic.py (random orthogonal init).

Model states: "E", "F" = MLPs trained with intervention fitting to E / F; "R" = untrained MLPs
(one per seed, shared by both structures because no training happens).

Stages (every stage is resumable: finished outputs are reused):
  prepare                      config, pilot base models, random-init models, data
  models --task i              train E and F base models for the i-th new seed
  maps --task i --num-tasks n  train every n-th map (individual and shared-joint) from i
  adapt --task k               held-out alignment adaptation and test evaluation, fold k
  summarize                    tables
  smoke                        all stages at tiny size under results/main_eval_smoke
"""
import argparse
import copy
import json
import os
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

import wandb  # noqa: E402  (das.DAS logs to wandb; keep it silent)

from das.Classification_Model import MLPForClassification, make_model_DAS_fitted  # noqa: E402
from das.Dataset_Generation import (  # noqa: E402
    make_dataset_sample_model_AndOrAnd,
    make_intervention_dataset_AndOr,
    make_intervention_dataset_AndOr_DAS_Fitted,
    make_intervention_dataset_AndOrAnd,
    make_intervention_dataset_AndOrAnd_DAS_Fitted,
)
from das.Helper_Functions import set_seed  # noqa: E402
from das.RevNet import RevNet  # noqa: E402
from das.Shared_Mapping import (  # noqa: E402
    Alignment,
    IdentityMap,
    InterventionSet,
    estimate_alignment,
    evaluate,
    freeze,
    hidden_at,
    intervene_logits,
    train_mapping,
)

STRUCTURES = {"E": "AndOrAnd", "F": "AndOr"}
STATES = ["E", "F", "R"]
CANDIDATES = ["E", "F"]
CONDITIONS = ["no_shared", "single_model", "shared_joint"]

FULL = dict(
    seeds=[4287, 3837, 9097, 2635, 5137],
    pilot_run="results/pilot_20261106T034514Z",
    pilot_roles={"4287": "source", "3837": "target"},
    embedding_size=4,
    hidden=24,
    layer=1,  # second hidden layer, same position as pilot.py (model.mlp.h[1])
    variable_dims=[list(range(0, 12)), list(range(12, 24))],
    base_train=1_280_000, base_eval=10_000, base_test=10_000,
    base_max_epochs=50, base_patience=5, base_threshold=0.001, base_sched_patience=10,
    iv_train=1_280_000, iv_dev=10_000, iv_test=10_000, adapt_train=10_000, adapt_dev=10_000,
    data_seed=20261106, cal_size=10_000, cal_seed=20261107,
    revnet_blocks=10, revnet_width=24,
    map_batch=6400, map_lr=0.001, map_max_epochs=50, map_patience=5, map_threshold=0.001,
    map_sched_patience=10,
    n_inits=2,
    adapt_sizes=[0, 256, 1024, 4096, 10000], adapt_updates=2000, adapt_batch=256,
    adapt_lr=0.001, adapt_val_interval=200,
    adapt_inits=["procrustes", "random_orthogonal"],
)
SMOKE = dict(
    FULL,
    base_train=2000, base_eval=500, base_test=500, base_max_epochs=1,
    iv_train=2000, iv_dev=500, iv_test=500, adapt_train=512, adapt_dev=256, cal_size=500,
    map_max_epochs=1, map_batch=256, n_inits=1,
    adapt_sizes=[0, 256], adapt_updates=20, adapt_val_interval=10,
)


def stable_seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) % (2 ** 31)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def distributive_label(x, embedding_size=4):
    """y = b AND (a OR c), a = (x1 == x2), b = (x3 == x4), c = (x5 == x6) (same for E and F)."""
    p = x.reshape(x.shape[0], 6, embedding_size)
    a = (p[:, 0] == p[:, 1]).all(-1)
    b = (p[:, 2] == p[:, 3]).all(-1)
    c = (p[:, 4] == p[:, 5]).all(-1)
    return (b & (a | c)).long()


def intervention_set(raw, embedding_size=4):
    base = torch.stack([d["base"] for d in raw])
    sources = torch.stack([d["sources"] for d in raw], dim=1)
    label = torch.stack([d["label"] for d in raw]).long()
    iv = torch.tensor([d["intervention"] for d in raw], dtype=torch.bool)
    return InterventionSet(base, sources, label, iv, distributive_label(base, embedding_size))


# --- from notebooks/Joint_Alignment_PoC.py --------------------------------------------------
class TrainableOrthogonalAlignment(nn.Module):
    """Learn a right-side orthogonal update (Cayley) of an initial orthogonal alignment."""

    def __init__(self, initial):
        super().__init__()
        self.register_buffer("initial", initial.Q.detach().clone())
        self.skew = nn.Parameter(torch.zeros_like(self.initial))

    def matrix(self):
        skew = self.skew - self.skew.T
        identity = torch.eye(skew.shape[0], device=skew.device, dtype=skew.dtype)
        update = torch.linalg.solve(identity + skew, identity - skew)
        return self.initial @ update

    def forward(self, hidden):
        return hidden @ self.matrix()

    def inverse(self, aligned):
        return aligned @ self.matrix().T


def orthogonality_error(matrix):
    eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    return (matrix.T @ matrix - eye).abs().max().item()


# --- from notebooks/Alignment_Adaptation_Diagnostic.py ---------------------------------------
def random_orthogonal_alignment(n, seed, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn((n, n), generator=generator))
    q = q * torch.where(torch.diag(r) < 0, -1.0, 1.0)
    return Alignment(n, matrix=q).to(device)


class Experiment:
    def __init__(self, root, cfg, device):
        self.root, self.cfg, self.device = Path(root), cfg, device
        self.models, self.data, self._cal = {}, {}, None

    # ------------------------------------------------------------------ paths and config
    def path(self, *parts):
        return self.root.joinpath(*map(str, parts))

    def write_config(self):
        p = self.path("config.json")
        if p.exists():
            saved = json.loads(p.read_text())
            if saved != self.cfg:
                raise RuntimeError(f"{p} differs from the code's config; use a new root")
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.cfg, indent=1) + "\n")

    def folds(self):
        seeds = self.cfg["seeds"]
        out = []
        for k, held in enumerate(seeds):
            train = [s for s in seeds if s != held]
            out.append({"fold": k, "heldout": held, "train": train, "ref": train[0]})
        return out

    # ------------------------------------------------------------------ models
    def model_path(self, state, seed):
        return self.path("models", f"{state}_seed{seed}.pt")

    def get_model(self, state, seed):
        key = (state, seed)
        if key not in self.models:
            p = self.model_path(state, seed)
            if not p.exists():
                raise FileNotFoundError(p)
            model = MLPForClassification(input_size=self.cfg["hidden"])
            model.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["state_dict"])
            self.models[key] = freeze(model.to(self.device))
        return self.models[key]

    def import_pilot_models(self):
        run = ROOT / self.cfg["pilot_run"]
        for seed, role in self.cfg["pilot_roles"].items():
            for s in STRUCTURES:
                dst = self.model_path(s, seed)
                if dst.exists():
                    continue
                src = run / "checkpoints" / "base_models" / f"{s}_{role}_seed{seed}.pt"
                saved = torch.load(src, map_location="cpu", weights_only=False)
                atomic_save({"state_dict": saved["state_dict"], "test_accuracy": saved["test_accuracy"],
                             "origin": str(src.relative_to(ROOT))}, dst)
                log(f"imported {src.name} -> {dst.name}")

    def make_random_models(self):
        for seed in self.cfg["seeds"]:
            dst = self.model_path("R", seed)
            if dst.exists():
                continue
            torch.manual_seed(stable_seed("random_init", seed))
            model = MLPForClassification(input_size=self.cfg["hidden"])
            atomic_save({"state_dict": model.state_dict(), "test_accuracy": None,
                         "origin": f"untrained, torch seed {stable_seed('random_init', seed)}"}, dst)
            log(f"created random-init model R seed {seed}")

    def train_base_model(self, seed):
        """Same procedure as pilot.py train_base_models (seed reset before the three splits)."""
        c = self.cfg
        for s, name in STRUCTURES.items():
            dst = self.model_path(s, seed)
            if dst.exists():
                continue
            gen = (make_intervention_dataset_AndOrAnd_DAS_Fitted if name == "AndOrAnd"
                   else make_intervention_dataset_AndOr_DAS_Fitted)
            t0 = time.time()
            set_seed(seed)
            train = gen(c["base_train"], c["embedding_size"])
            dev = gen(c["base_eval"], c["embedding_size"])
            test = gen(c["base_test"], c["embedding_size"])
            log(f"base model {s} seed {seed}: data {time.time() - t0:.0f}s")
            model, acc = make_model_DAS_fitted(
                DAS_Train=train, DAS_Test=test, DAS_Eval=dev, Hidden_Layer_Size=c["hidden"],
                inter_dim=c["variable_dims"], DEVICE=self.device, Max_Epochs=c["base_max_epochs"],
                Early_Stopping_Epochs=c["base_patience"],
                early_stopping_improve_threshold=c["base_threshold"],
                ReduceLROnPlateau_patience=c["base_sched_patience"])
            atomic_save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                         "test_accuracy": acc, "origin": "main_eval.py models",
                         "seconds": time.time() - t0}, dst)
            log(f"base model {s} seed {seed}: test_accuracy={acc:.4f} total {time.time() - t0:.0f}s")
            del train, dev, test, model

    # ------------------------------------------------------------------ data
    def data_path(self, candidate, split):
        return self.path("data", f"{candidate}_{split}.pt")

    def splits(self):
        c = self.cfg
        return {"train": c["iv_train"], "dev": c["iv_dev"], "adapt_train": c["adapt_train"],
                "adapt_dev": c["adapt_dev"], "test": c["iv_test"]}

    def make_data(self):
        for cand in CANDIDATES:
            gen = make_intervention_dataset_AndOrAnd if cand == "E" else make_intervention_dataset_AndOr
            for split, size in self.splits().items():
                p = self.data_path(cand, split)
                if p.exists():
                    continue
                t0 = time.time()
                set_seed(stable_seed("iv", cand, split, self.cfg["data_seed"]))
                data = intervention_set(gen(size, self.cfg["embedding_size"]), self.cfg["embedding_size"])
                atomic_save(dict(zip(["base", "sources", "label", "iv", "base_label"], data.state())), p)
                changed = (data.label != data.base_label).float().mean().item()
                log(f"data {cand} {split}: n={size} label_changed={changed:.3f} {time.time() - t0:.0f}s")
        p = self.path("data", "calibration.pt")
        if not p.exists():
            set_seed(self.cfg["cal_seed"])
            x = [make_dataset_sample_model_AndOrAnd(self.cfg["embedding_size"])[0]
                 for _ in range(self.cfg["cal_size"])]
            atomic_save(torch.as_tensor(np.asarray(x), dtype=torch.float32), p)

    def get_data(self, candidate, split):
        key = (candidate, split)
        if key not in self.data:
            self.data[key] = InterventionSet.load(self.data_path(candidate, split), self.device)
        return self.data[key]

    def calibration(self):
        if self._cal is None:
            self._cal = torch.load(self.path("data", "calibration.pt"), map_location=self.device)
        return self._cal

    def procrustes(self, state_ref, ref, state_m, m):
        if (state_ref, ref) == (state_m, m):
            return Alignment(self.cfg["hidden"]).to(self.device), 0.0
        return estimate_alignment(self.get_model(state_ref, ref), self.get_model(state_m, m),
                                  self.calibration(), self.cfg["layer"], kind="orthogonal")

    # ------------------------------------------------------------------ maps
    def make_phi(self, seed):
        torch.manual_seed(seed)
        return RevNet(number_blocks=self.cfg["revnet_blocks"], in_features=self.cfg["hidden"],
                      hidden_size=self.cfg["revnet_width"]).to(self.device)

    def map_tasks(self):
        tasks = []
        for kind in ["joint", "individual"]:
            for state in STATES:
                for cand in CANDIDATES:
                    for idx in range(len(self.cfg["seeds"])):
                        for init in range(self.cfg["n_inits"]):
                            tasks.append((kind, state, cand, idx, init))
        return tasks

    def individual_path(self, state, seed, cand, init):
        return self.path("maps", "individual", f"{state}_seed{seed}_C{cand}_init{init}.pt")

    def joint_path(self, state, cand, fold, init):
        return self.path("maps", "joint", f"{state}_C{cand}_fold{fold}_init{init}.pt")

    def train_individual(self, state, seed, cand, init):
        p = self.individual_path(state, seed, cand, init)
        if p.exists():
            return
        c = self.cfg
        t0 = time.time()
        phi_seed = stable_seed("individual", state, seed, cand, init)
        phi = self.make_phi(phi_seed)
        members = [(self.get_model(state, seed), Alignment(c["hidden"]).to(self.device))]
        log(f"individual phi: state={state} seed={seed} C={cand} init={init}")
        generator = torch.Generator().manual_seed(phi_seed + 1)
        info = train_mapping(phi, members, self.get_data(cand, "train"), self.get_data(cand, "dev"),
                             c["variable_dims"], c["layer"], ids_per_member=c["map_batch"],
                             max_epochs=c["map_max_epochs"], patience=c["map_patience"],
                             improve_threshold=c["map_threshold"], lr=c["map_lr"],
                             scheduler_patience=c["map_sched_patience"], generator=generator, log=log)
        test, _ = evaluate(phi, members[0], self.get_data(cand, "test"), c["variable_dims"], c["layer"])
        info.update(seconds=time.time() - t0, init_seed=phi_seed)
        atomic_save({"phi": phi.state_dict(), "info": info, "test": test}, p)
        log(f"  done: test IIA={test['iia']:.4f} epochs={info['epochs']} {info['seconds']:.0f}s")

    def train_joint(self, state, cand, fold_index, init):
        p = self.joint_path(state, cand, fold_index, init)
        if p.exists():
            return
        c = self.cfg
        fold = self.folds()[fold_index]
        t0 = time.time()
        phi_seed = stable_seed("joint", state, cand, fold_index, init)
        phi = self.make_phi(phi_seed)
        members, costs = [], {}
        for s in fold["train"]:
            align, cost = self.procrustes(state, fold["ref"], state, s)
            costs[str(s)] = cost
            members.append((self.get_model(state, s),
                            align if s == fold["ref"] else TrainableOrthogonalAlignment(align).to(self.device)))
        log(f"joint phi: state={state} C={cand} fold={fold_index} (held out {fold['heldout']}, "
            f"ref {fold['ref']}) init={init}")
        info = self._train_joint_mapping(phi, members, self.get_data(cand, "train"),
                                         self.get_data(cand, "dev"), phi_seed)
        test = {}
        for s, member in zip(fold["train"], members):
            test[str(s)] = evaluate(phi, member, self.get_data(cand, "test"), c["variable_dims"], c["layer"])[0]
        alignments = {str(s): (a.matrix().detach().cpu() if isinstance(a, TrainableOrthogonalAlignment)
                               else torch.eye(c["hidden"]))
                      for s, (_, a) in zip(fold["train"], members)}
        info.update(seconds=time.time() - t0, init_seed=phi_seed, procrustes_cost=costs)
        atomic_save({"phi": phi.state_dict(), "alignments": alignments, "info": info, "test": test}, p)
        log(f"  done: member test IIA={[round(v['iia'], 4) for v in test.values()]} "
            f"epochs={info['epochs']} {info['seconds']:.0f}s")

    def _train_joint_mapping(self, phi, members, train, dev, init_seed):
        """notebooks/Joint_Alignment_PoC.py train_joint_mapping (reference member stays fixed)."""
        c = self.cfg
        trainable = [a for _, a in members[1:]]
        params = list(phi.parameters())
        for a in trainable:
            params.extend(a.parameters())
        optimizer = optim.Adam(params, lr=c["map_lr"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=c["map_sched_patience"])
        best_loss, best_state, best_aligns, bad, history, updates = float("inf"), None, None, 0, [], 0
        batch_size = c["map_batch"] // len(members)
        generator = torch.Generator(device="cpu").manual_seed(init_seed + 1)
        for epoch in range(c["map_max_epochs"]):
            phi.train()
            order = torch.randperm(len(train), generator=generator).to(train.base.device)
            total, batches = 0.0, 0
            for start in range(0, len(train), batch_size):
                sub = train.subset(order[start:start + batch_size])
                optimizer.zero_grad()
                loss = torch.stack([
                    nn.functional.cross_entropy(
                        intervene_logits(phi, m, sub, c["variable_dims"], c["layer"]), sub.label)
                    for m in members]).mean()
                loss.backward()
                optimizer.step()
                total += loss.item()
                batches += 1
                updates += 1
            stats = [evaluate(phi, m, dev, c["variable_dims"], c["layer"])[0] for m in members]
            dev_loss = sum(s["loss"] for s in stats) / len(stats)
            scheduler.step(total / batches)
            if dev_loss < best_loss:
                improvement = best_loss - dev_loss
                best_loss, best_state = dev_loss, copy.deepcopy(phi.state_dict())
                best_aligns = [copy.deepcopy(a.state_dict()) for a in trainable]
                bad = 0 if improvement > c["map_threshold"] else bad + 1
            else:
                bad += 1
            orth = max((orthogonality_error(a.matrix()) for a in trainable), default=0.0)
            history.append({"epoch": epoch + 1, "train_loss": total / batches, "dev_loss": dev_loss,
                            "dev_iia_per_member": [s["iia"] for s in stats],
                            "lr": optimizer.param_groups[0]["lr"], "updates": updates,
                            "orthogonality_max_error": orth})
            log(f"  joint epoch {epoch + 1}: train_loss={total / batches:.4f} dev_loss={dev_loss:.4f} "
                f"dev_iia={[round(s['iia'], 4) for s in stats]} bad={bad}")
            if bad >= c["map_patience"]:
                break
        if best_state is None:
            raise RuntimeError("joint mapping training produced no checkpoint")
        phi.load_state_dict(best_state)
        for a, st in zip(trainable, best_aligns):
            a.load_state_dict(st)
        phi.eval()
        return {"best_dev_loss": best_loss, "epochs": len(history), "updates": updates, "history": history}

    def load_phi(self, path):
        saved = torch.load(path, map_location=self.device, weights_only=False)
        phi = self.make_phi(0)
        phi.load_state_dict(saved["phi"])
        phi.eval()
        for prm in phi.parameters():
            prm.requires_grad_(False)
        return phi

    # ------------------------------------------------------------------ adaptation
    def adapt_alignment(self, phi, model, initial, ids, seed):
        """notebooks/Alignment_Adaptation_Controls.py adapt_alignment (phi and model frozen)."""
        c = self.cfg
        cand_train, cand_dev = self._adapt_data

        def dev_loss(a):
            return evaluate(phi, (model, a), cand_dev, c["variable_dims"], c["layer"])[0]["loss"]

        alignment = TrainableOrthogonalAlignment(initial).to(self.device)
        if len(ids) == 0:
            return alignment, 0, dev_loss(alignment)
        optimizer = optim.Adam(alignment.parameters(), lr=c["adapt_lr"])
        best_state, best_loss, best_update = copy.deepcopy(alignment.state_dict()), dev_loss(alignment), 0
        gen = torch.Generator(device="cpu").manual_seed(seed + 1)
        ids = ids.to(self.device)
        for update in range(1, c["adapt_updates"] + 1):
            sel = torch.randint(len(ids), (c["adapt_batch"],), generator=gen).to(self.device)
            batch = cand_train.subset(ids[sel])
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(
                intervene_logits(phi, (model, alignment), batch, c["variable_dims"], c["layer"]), batch.label)
            loss.backward()
            optimizer.step()
            if update % c["adapt_val_interval"] == 0 or update == c["adapt_updates"]:
                d = dev_loss(alignment)
                if d < best_loss:
                    best_loss, best_state, best_update = d, copy.deepcopy(alignment.state_dict()), update
        alignment.load_state_dict(best_state)
        return alignment, best_update, best_loss

    def combos(self):
        out = [(s, cand, t) for s in STRUCTURES for cand in CANDIDATES for t in STRUCTURES]
        return out + [("R", cand, "R") for cand in CANDIDATES]

    def run_adapt(self, fold_index):
        c = self.cfg
        fold = self.folds()[fold_index]
        held, ref = fold["heldout"], fold["ref"]
        out = self.path("adapt", f"fold{fold_index}.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        done = set()
        if out.exists():
            for line in out.read_text().splitlines():
                r = json.loads(line)
                done.add(r["key"])
        order = torch.randperm(c["adapt_train"], generator=torch.Generator().manual_seed(
            stable_seed("adapt_order", c["data_seed"])))
        t_start = time.time()
        with out.open("a") as fh:
            def write(row):
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                done.add(row["key"])

            for (S, C, T) in self.combos():
                self._adapt_data = (self.get_data(C, "adapt_train"), self.get_data(C, "adapt_dev"))
                test = self.get_data(C, "test")
                target = self.get_model(T, held)
                for r in range(c["n_inits"]):
                    key = f"{S}{C}{T}|individual_self|r{r}"
                    if key not in done:
                        phi = self.load_phi(self.individual_path(T, held, C, r))
                        res, _ = evaluate(phi, (target, Alignment(c["hidden"]).to(self.device)), test,
                                          c["variable_dims"], c["layer"])
                        write(dict(res, key=key, fold=fold_index, heldout=held, ref=ref, S=S, C=C, T=T,
                                   condition="individual_self", repeat=r, adapt_init=None, adapt_examples=None))
                    phis = {"no_shared": IdentityMap().to(self.device),
                            "single_model": self.load_phi(self.individual_path(S, ref, C, r)),
                            "shared_joint": self.load_phi(self.joint_path(S, C, fold_index, r))}
                    for ainit in c["adapt_inits"]:
                        if ainit == "procrustes":
                            initial, cost = self.procrustes(S, ref, T, held)
                        else:
                            initial, cost = random_orthogonal_alignment(
                                c["hidden"], stable_seed("random_orthogonal", fold_index, S, T, r), self.device), None
                        for cond, phi in phis.items():
                            for n in c["adapt_sizes"]:
                                key = f"{S}{C}{T}|{cond}|r{r}|{ainit}|n{n}"
                                if key in done:
                                    continue
                                t0 = time.time()
                                seed = stable_seed("adapt", fold_index, S, C, T, cond, r, ainit, n)
                                a, best_update, dev_loss = self.adapt_alignment(phi, target, initial, order[:n], seed)
                                res, _ = evaluate(phi, (target, a), test, c["variable_dims"], c["layer"])
                                write(dict(res, key=key, fold=fold_index, heldout=held, ref=ref, S=S, C=C, T=T,
                                           condition=cond, repeat=r, adapt_init=ainit, adapt_examples=n,
                                           adapt_best_update=best_update, adapt_dev_loss=dev_loss,
                                           alignment_cost=cost, orthogonality_error=orthogonality_error(a.matrix()),
                                           seconds=time.time() - t0))
                    log(f"fold {fold_index} {S}{C}{T} repeat {r} done ({time.time() - t_start:.0f}s so far)")

    # ------------------------------------------------------------------ summary
    def summarize(self):
        import pandas as pd

        def md(frame, index=True):
            return "```\n" + frame.to_string(index=index) + "\n```"

        c = self.cfg
        lines = ["# 0485 main evaluation summary", ""]
        rows = []
        for f in sorted(self.path("adapt").glob("fold*.jsonl")):
            rows += [json.loads(x) for x in f.read_text().splitlines()]
        df = pd.DataFrame(rows)
        df.to_csv(self.path("adapt_results.csv"), index=False)

        acc = []
        for state in STRUCTURES:
            for seed in c["seeds"]:
                model = self.get_model(state, seed)
                x = self.get_data("E", "test").base
                with torch.no_grad():
                    pred = model(x).argmax(1)
                acc.append({"state": state, "seed": seed,
                            "task_accuracy": (pred == distributive_label(x, c["embedding_size"])).float().mean().item()})
        lines += ["## Base-model task accuracy (test bases of candidate E)", "",
                  md(pd.DataFrame(acc).pivot(index="seed", columns="state", values="task_accuracy")), ""]

        base = []
        for cand in CANDIDATES:
            d = self.get_data(cand, "test")
            base.append({"C": cand, "label_changed_rate": (d.label != d.base_label).float().mean().item(),
                         "label_1_rate": d.label.float().mean().item()})
        lines += ["## Baselines of the test intervention sets", "", md(pd.DataFrame(base), index=False), ""]

        if len(df):
            main = df[df.condition != "individual_self"]
            g = main.groupby(["S", "C", "T", "adapt_init", "condition", "adapt_examples"])["iia"]
            table = g.mean().unstack("adapt_examples").round(4)
            lines += ["## Held-out IIA, mean over folds and repeats", "", md(table), ""]
            rng = (g.min().unstack("adapt_examples").round(3).astype(str) + "–"
                   + g.max().unstack("adapt_examples").round(3).astype(str))
            lines += ["## Held-out IIA, min–max", "", md(rng), ""]
            ind = df[df.condition == "individual_self"].groupby(["S", "C", "T"])["iia"].agg(["mean", "min", "max"])
            lines += ["## Individual map on the held-out model itself (reference)", "", md(ind.round(4)), ""]
            sub = main[main.adapt_examples == max(c["adapt_sizes"])]
            ch = sub.groupby(["S", "C", "T", "adapt_init", "condition"])[["iia_label_changed", "iia_label_unchanged"]].mean()
            lines += [f"## IIA split by whether the label changes (n={max(c['adapt_sizes'])})", "",
                      md(ch.round(4)), ""]
        self.path("summary.md").write_text("\n".join(lines) + "\n")
        log(f"wrote {self.path('summary.md')}")


def run_stage(exp, stage, task, num_tasks):
    if stage == "prepare":
        exp.write_config()
        exp.import_pilot_models()
        exp.make_random_models()
        exp.make_data()
    elif stage == "models":
        new = [s for s in exp.cfg["seeds"] if str(s) not in exp.cfg["pilot_roles"]]
        exp.train_base_model(new[task])
    elif stage == "maps":
        tasks = exp.map_tasks()
        for kind, state, cand, idx, init in tasks[task::num_tasks]:
            if kind == "individual":
                exp.train_individual(state, exp.cfg["seeds"][idx], cand, init)
            else:
                exp.train_joint(state, cand, idx, init)
    elif stage == "adapt":
        exp.run_adapt(task)
    elif stage == "summarize":
        exp.summarize()
    else:
        raise ValueError(stage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "models", "maps", "adapt", "summarize", "smoke"])
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=1)
    parser.add_argument("--root", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this on a GPU node")
    device = torch.device("cuda")
    torch.set_num_threads(max(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), 1))
    wandb.init(mode="disabled")
    log(f"stage={args.stage} task={args.task}/{args.num_tasks} device={torch.cuda.get_device_name(device)}")

    if args.stage == "smoke":
        root = Path(args.root or ROOT / "results" / "main_eval_smoke")
        if root.exists():
            shutil.rmtree(root)
        exp = Experiment(root, SMOKE, device)
        run_stage(exp, "prepare", 0, 1)
        for i in range(len([s for s in SMOKE["seeds"] if str(s) not in SMOKE["pilot_roles"]])):
            run_stage(exp, "models", i, 1)
        run_stage(exp, "maps", 0, 1)
        for k in range(len(SMOKE["seeds"])):
            run_stage(exp, "adapt", k, 1)
        run_stage(exp, "summarize", 0, 1)
        n_rows = sum(len(f.read_text().splitlines()) for f in exp.path("adapt").glob("fold*.jsonl"))
        log(f"smoke finished: {n_rows} adaptation rows")
        return

    exp = Experiment(args.root or ROOT / "results" / "main_eval", FULL, device)
    if args.stage not in ("prepare", "models"):
        if not exp.path("config.json").exists():
            raise RuntimeError("run the prepare stage first")
        exp.write_config()  # verifies that the config did not change
    t0 = time.time()
    run_stage(exp, args.stage, args.task, args.num_tasks)
    log(f"stage {args.stage} task {args.task} finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
