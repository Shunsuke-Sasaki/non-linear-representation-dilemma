#!/usr/bin/env python
"""Shared non-linear alignment maps evaluated on unseen models (research plan, 2026-10-09).

Stages (each one is resumable: finished checkpoints are reused):
  dev        development models: individual RevNet reproduction (trained + random) and the
             numerical / permuted-copy controls of plan §10                 -> <root>/dev/
  train      main seeds: MLPs, individual phi_m, shared linear R, shared nonlinear phi
             (leave-one-model-out folds)                                    -> <root>/...
  evaluate   all IIA evaluations (within-model, unseen model, single-model transfer,
             cross-model swap)                                              -> <root>/<align>/results.jsonl
  summarize  tables                                                          -> <root>/<align>/summary.md
  all        train + evaluate + summarize

Example:
  .venv/bin/python notebooks/Shared_Mapping.py dev --device cuda
  .venv/bin/python notebooks/Shared_Mapping.py all --device cuda
  .venv/bin/python notebooks/Shared_Mapping.py all --device cuda --align orthogonal   # §5.2 sensitivity
"""
import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import wandb  # noqa: E402  (imported by das.DAS; keep it silent)
wandb.init(mode="disabled")

from das.Dataset_Generation import (make_model_dataset,  # noqa: E402
                                    make_intervention_dataset_variable_intervention_all)
from das.RevNet import RevNet  # noqa: E402
from das.Rotation_Model import Rotation  # noqa: E402
from das.Shared_Mapping import (Alignment, IdentityMap, InterventionSet,  # noqa: E402
                                estimate_alignment, evaluate, freeze, mlp_loss_acc, new_model,
                                permuted_copy, seed_everything_for, train_mapping, train_mlp,
                                hidden_at, head_from, intervene_logits)

STATES = ["trained", "random"]

FULL = dict(
    setting="Both Equality Relations", embedding_size=4, hidden=16, layer=0,
    variable_dims=[list(range(0, 8)), list(range(8, 16))],
    mlp_train=1048576, mlp_dev=10000, mlp_test=10000, mlp_batch=1024, mlp_max_epochs=20, mlp_patience=3,
    iv_train=1280000, iv_dev=10000, iv_test=10000, iv_data_seed=20261009,
    cal_size=10000, cal_seed=20261010,
    revnet_blocks=10, revnet_width=128,
    map_batch=6400, map_lr=0.001, map_max_epochs=50, map_patience=5, map_threshold=0.001, map_sched_patience=10,
    dev_seeds=[5234, 4641], main_seeds=[4287, 3837, 9097, 2635, 5137], n_inits=2,
)
QUICK = dict(FULL, mlp_train=65536, mlp_max_epochs=5, iv_train=64000, iv_dev=2000, iv_test=2000,
             cal_size=2000, map_max_epochs=3, n_inits=1)


def stable_seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) % (2 ** 31)


class Experiment:
    def __init__(self, args):
        self.cfg = QUICK if args.quick else FULL
        self.device = args.device
        self.align_kind = args.align
        repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        self.root = os.path.abspath(os.path.join(repo, "Results", "SharedMapping", "quick" if args.quick else "full"))
        self.run_dir = os.path.join(self.root, self.align_kind)
        for d in [self.root, self.run_dir]:
            os.makedirs(d, exist_ok=True)
        self.log_path = os.path.join(self.root, "log.txt")
        self.models = {}
        self._iv = None
        self._cal = None
        self._alignments = {}
        with open(os.path.join(self.root, "config.json"), "w") as f:
            json.dump(self.cfg, f, indent=1)

    # ------------------------------------------------------------------ utils
    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def path(self, *parts):
        p = os.path.join(self.root, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    # ------------------------------------------------------------------ data
    def intervention_data(self):
        """One shared set of intervention IDs for every model and condition (plan §7.2, §8.1)."""
        if self._iv is None:
            c = self.cfg
            p = self.path("data", "interventions.pt")
            if not os.path.exists(p):
                self.log("Generating intervention datasets ...")
                seed_everything_for(c["iv_data_seed"])
                splits = {name: InterventionSet.from_raw(
                    make_intervention_dataset_variable_intervention_all(c[f"iv_{name}"], c["embedding_size"]),
                    c["embedding_size"]) for name in ["train", "dev", "test"]}
                torch.save({k: dict(zip(["base", "sources", "label", "iv", "base_label"], v.state()))
                            for k, v in splits.items()}, p)
            raw = torch.load(p)
            self._iv = {k: InterventionSet(v["base"], v["sources"], v["label"], v["iv"], v["base_label"]).to(self.device)
                        for k, v in raw.items()}
            for k, v in self._iv.items():
                self.log(f"  interventions {k}: n={len(v)} counterfactual label=1 rate={v.label.float().mean():.3f} "
                         f"label changed rate={(v.label != v.base_label).float().mean():.3f} "
                         f"left/right/both={v.iv[:, 0].logical_and(~v.iv[:, 1]).sum().item()}/"
                         f"{(~v.iv[:, 0]).logical_and(v.iv[:, 1]).sum().item()}/{v.iv.all(1).sum().item()}")
        return self._iv

    def calibration_inputs(self):
        """Normal inputs used only to estimate A_m; independent of dev/test interventions (§5.3)."""
        if self._cal is None:
            p = self.path("data", "calibration.pt")
            if not os.path.exists(p):
                seed_everything_for(self.cfg["cal_seed"])
                X, _ = make_model_dataset(self.cfg["cal_size"], self.cfg["embedding_size"], "cpu")
                torch.save(X, p)
            self._cal = torch.load(p).to(self.device)
        return self._cal

    # ------------------------------------------------------------------ MLPs
    def get_models(self, seed):
        """Trained MLP and its saved initial state (the 'random' state) for one seed."""
        if seed in self.models:
            return self.models[seed]
        c = self.cfg
        p = self.path("models", f"seed{seed}.pt")
        if not os.path.exists(p):
            self.log(f"Training MLP seed {seed}")
            g = seed_everything_for(seed)
            model = new_model(c["hidden"]).to(self.device)
            init_state = {k: v.clone() for k, v in model.state_dict().items()}
            X_tr, y_tr = make_model_dataset(c["mlp_train"], c["embedding_size"], self.device)
            X_dev, y_dev = make_model_dataset(c["mlp_dev"], c["embedding_size"], self.device)
            X_te, y_te = make_model_dataset(c["mlp_test"], c["embedding_size"], self.device)
            self.log(f"  MLP data label=1 rate train={y_tr.mean():.3f} dev={y_dev.mean():.3f} test={y_te.mean():.3f}")
            model, info = train_mlp(model, X_tr, y_tr, X_dev, y_dev, batch_size=c["mlp_batch"],
                                    max_epochs=c["mlp_max_epochs"], patience=c["mlp_patience"], generator=g, log=self.log)
            random_model = new_model(c["hidden"]).to(self.device)
            random_model.load_state_dict(init_state)
            info["test_acc"] = {"trained": mlp_loss_acc(model, X_te, y_te)[1],
                                "random": mlp_loss_acc(random_model, X_te, y_te)[1]}
            self.log(f"  seed {seed}: test acc trained={info['test_acc']['trained']:.4f} "
                     f"random={info['test_acc']['random']:.4f} (best epoch {info['best_epoch']})")
            torch.save({"trained": model.state_dict(), "random": init_state, "info": info}, p)
        ckpt = torch.load(p, map_location=self.device)
        out = {"info": ckpt["info"]}
        for state in STATES:
            m = new_model(c["hidden"]).to(self.device)
            m.load_state_dict(ckpt[state])
            out[state] = freeze(m)
        self.models[seed] = out
        return out

    # ------------------------------------------------------------------ alignment
    def alignment(self, ref_seed, seed, state, kind=None):
        """A_m of `seed` relative to `ref_seed`, estimated from normal activations only."""
        kind = kind or self.align_kind
        key = (ref_seed, seed, state, kind)
        if key not in self._alignments:
            if ref_seed == seed:
                self._alignments[key] = (Alignment(self.cfg["hidden"]).to(self.device), 0.0)
            else:
                ref = self.get_models(ref_seed)[state]
                m = self.get_models(seed)[state]
                self._alignments[key] = estimate_alignment(ref, m, self.calibration_inputs(), self.cfg["layer"], kind)
        return self._alignments[key]

    # ------------------------------------------------------------------ maps
    def make_phi(self, kind, seed):
        torch.manual_seed(seed)
        if kind == "revnet":
            phi = RevNet(number_blocks=self.cfg["revnet_blocks"], in_features=self.cfg["hidden"],
                         hidden_size=self.cfg["revnet_width"])
        elif kind == "rotation":
            phi = Rotation(self.cfg["hidden"])
        elif kind == "identity":
            phi = IdentityMap()
        else:
            raise ValueError(kind)
        return phi.to(self.device)

    def train_or_load_phi(self, ckpt_path, kind, init_seed, members, ids_per_member):
        phi = self.make_phi(kind, init_seed)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device)
            phi.load_state_dict(ckpt["phi"])
            phi.eval()
            return phi, ckpt["info"]
        c, iv = self.cfg, self.intervention_data()
        t0 = time.time()
        info = train_mapping(phi, members, iv["train"], iv["dev"], c["variable_dims"], c["layer"],
                             ids_per_member=ids_per_member, max_epochs=c["map_max_epochs"], patience=c["map_patience"],
                             improve_threshold=c["map_threshold"], lr=c["map_lr"],
                             scheduler_patience=c["map_sched_patience"],
                             generator=seed_everything_for(init_seed + 1), log=self.log)
        info["seconds"] = time.time() - t0
        info["init_seed"] = init_seed
        torch.save({"phi": phi.state_dict(), "info": info}, ckpt_path)
        self.log(f"  saved {os.path.relpath(ckpt_path, self.root)} ({info['seconds']:.0f}s, {info['epochs']} epochs)")
        return phi, info

    def individual_phi(self, seed, state, init):
        """phi_m trained on model m alone (A = identity, so it is reusable across folds)."""
        p = self.path("individual", state, f"seed{seed}_init{init}.pt")
        model = self.get_models(seed)[state]
        member = (model, Alignment(self.cfg["hidden"]).to(self.device))
        if not os.path.exists(p):
            self.log(f"Individual RevNet: seed {seed} {state} init {init}")
        return self.train_or_load_phi(p, "revnet", stable_seed("individual", seed, state, init), [member],
                                      self.cfg["map_batch"])

    def folds(self, seeds):
        """Leave-one-model-out; the reference is the first training seed in seed order (§5.1)."""
        out = []
        for k, held in enumerate(seeds):
            train = [s for s in seeds if s != held]
            out.append({"fold": k, "heldout": held, "train": train, "ref": train[0]})
        return out

    def shared_phi(self, condition, fold, state, init):
        kind = {"shared_nonlinear": "revnet", "shared_linear": "rotation"}[condition]
        p = os.path.join(self.run_dir, "maps", condition, state, f"fold{fold['fold']}_init{init}.pt")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        members = [(self.get_models(s)[state], self.alignment(fold["ref"], s, state)[0]) for s in fold["train"]]
        if not os.path.exists(p):
            self.log(f"{condition}: fold {fold['fold']} (held out {fold['heldout']}, ref {fold['ref']}) {state} init {init}")
        return self.train_or_load_phi(p, kind, stable_seed(condition, self.align_kind, fold["fold"], state, init),
                                      members, self.cfg["map_batch"] // len(members))

    # ------------------------------------------------------------------ stages
    def stage_dev(self):
        """Plan §12 steps 1-2 on development models only."""
        c, iv = self.cfg, self.intervention_data()
        vd, layer = c["variable_dims"], c["layer"]
        report = {"models": {}, "individual": {}, "controls": {}}
        for seed in c["dev_seeds"]:
            report["models"][seed] = self.get_models(seed)["info"]["test_acc"]
        # Step 1: individual non-linear map reaches high IIA on trained and random models.
        phis = {}
        for seed in c["dev_seeds"]:
            for state in STATES:
                p = self.path("dev", "individual", state, f"seed{seed}.pt")
                member = (self.get_models(seed)[state], Alignment(c["hidden"]).to(self.device))
                if not os.path.exists(p):
                    self.log(f"[dev] Individual RevNet: seed {seed} {state}")
                phi, info = self.train_or_load_phi(p, "revnet", stable_seed("dev", seed, state), [member], c["map_batch"])
                phis[(seed, state)] = phi
                res, _ = evaluate(phi, member, iv["test"], vd, layer)
                report["individual"][f"{seed}/{state}"] = {"test": res, "epochs": info["epochs"],
                                                           "seconds": info.get("seconds")}
                self.log(f"[dev] individual seed {seed} {state}: test IIA {res['iia']:.4f}")
        # Step 2: numerical and coordinate controls (§10).
        X = self.calibration_inputs()
        seed0 = c["dev_seeds"][0]
        ctl = report["controls"]
        with torch.no_grad():
            for state in STATES:
                model = self.get_models(seed0)[state]
                phi = phis[(seed0, state)]
                ident = Alignment(c["hidden"]).to(self.device)
                h = hidden_at(model, X, layer)
                s = {"phi_inverse_max_abs_err": (phi.inverse(phi(h)) - h).abs().max().item(),
                     "fresh_revnet_inverse_max_abs_err": (lambda f: (f.inverse(f(h)) - h).abs().max().item())(
                         self.make_phi("revnet", 0))}
                base_logits = model(iv["test"].base)
                null = iv["test"].subset(slice(None))
                null.iv = torch.zeros_like(null.iv)
                out = intervene_logits(phi, (model, ident), null, vd, layer)
                s["null_intervention_max_abs_logit_diff"] = (out - base_logits).abs().max().item()
                s["null_intervention_pred_agreement"] = (out.argmax(1) == base_logits.argmax(1)).float().mean().item()
                selfswap = iv["test"].subset(slice(None))
                selfswap.sources = torch.stack([selfswap.base, selfswap.base])
                selfswap.iv = torch.ones_like(selfswap.iv)
                out = intervene_logits(phi, (model, ident), selfswap, vd, layer)
                s["self_swap_max_abs_logit_diff"] = (out - base_logits).abs().max().item()
                s["self_swap_pred_agreement"] = (out.argmax(1) == base_logits.argmax(1)).float().mean().item()
                s["head_split_max_abs_diff"] = (head_from(model, h, layer) - model(X)).abs().max().item()
                # Permuted copy: same mechanism, different coordinates.
                g = torch.Generator().manual_seed(stable_seed("perm-copy", seed0, state))
                pi = torch.randperm(c["hidden"], generator=g)
                twin = freeze(permuted_copy(model, layer, pi))
                s["perm_copy_max_abs_logit_diff"] = (twin(X) - model(X)).abs().max().item()
                oracle = Alignment(c["hidden"], perm=torch.argsort(pi)).to(self.device)
                for kind in ["permutation", "orthogonal"]:
                    est, cost = estimate_alignment(model, twin, X, layer, kind)
                    if kind == "permutation":
                        alive = h.std(0) > 1e-8
                        s["perm_copy_estimated_perm_exact"] = bool((est.perm.cpu() == torch.argsort(pi)).all())
                        s["perm_copy_estimated_perm_match_alive_units"] = (
                            (est.perm.cpu() == torch.argsort(pi))[alive.cpu()].float().mean().item())
                        s["n_dead_units"] = int((~alive).sum().item())
                    s[f"perm_copy_{kind}_alignment_cost"] = cost
                    s[f"perm_copy_{kind}_reconstruction_max_abs_err"] = (est(hidden_at(twin, X, layer)) - h).abs().max().item()
                    s[f"perm_copy_iia_{kind}_estimated"] = evaluate(phi, (twin, est), iv["test"], vd, layer)[0]["iia"]
                s["perm_copy_iia_oracle"] = evaluate(phi, (twin, oracle), iv["test"], vd, layer)[0]["iia"]
                s["original_iia"] = evaluate(phi, (model, ident), iv["test"], vd, layer)[0]["iia"]
                s["perm_copy_iia_no_alignment"] = evaluate(phi, (twin, ident), iv["test"], vd, layer)[0]["iia"]
                # Preview of transfer between two genuinely different dev models.
                other = c["dev_seeds"][1]
                for kind in ["permutation", "orthogonal"]:
                    A, cost = estimate_alignment(model, self.get_models(other)[state], X, layer, kind)
                    s[f"transfer_{seed0}_to_{other}_{kind}_cost"] = cost
                    s[f"transfer_{seed0}_to_{other}_{kind}_iia"] = evaluate(
                        phi, (self.get_models(other)[state], A), iv["test"], vd, layer)[0]["iia"]
                ctl[state] = s
                self.log(f"[dev] controls {state}: " + json.dumps(s))
        with open(self.path("dev", "dev_report.json"), "w") as f:
            json.dump(report, f, indent=1)
        self.log(f"[dev] report written to {self.path('dev', 'dev_report.json')}")

    def stage_train(self):
        c = self.cfg
        seeds = c["main_seeds"]
        for seed in seeds:
            self.get_models(seed)
        self.intervention_data()
        for state in STATES:
            for seed in seeds:
                for init in range(c["n_inits"]):
                    self.individual_phi(seed, state, init)
        for condition in ["shared_linear", "shared_nonlinear"]:
            for state in STATES:
                for fold in self.folds(seeds):
                    for init in range(c["n_inits"]):
                        self.shared_phi(condition, fold, state, init)

    def stage_evaluate(self):
        c, iv = self.cfg, self.intervention_data()
        vd, layer, test = c["variable_dims"], c["layer"], iv["test"]
        seeds = c["main_seeds"]
        records, preds = [], {}

        def add(rec, correct):
            records.append(rec)
            preds["|".join(f"{k}={rec[k]}" for k in ["condition", "state", "fold", "init", "map_seed",
                                                      "target", "source"])] = correct.to(torch.bool)

        for seed in seeds:
            info = self.get_models(seed)["info"]
            for state in STATES:
                records.append({"condition": "normal_accuracy", "state": state, "target": seed,
                                "acc": info["test_acc"][state], "best_epoch": info["best_epoch"]})
        folds = self.folds(seeds)
        for state in STATES:
            self.log(f"Evaluating {state} ...")
            # Alignment cost of every model to each reference used.
            for ref in sorted({f["ref"] for f in folds}):
                for seed in seeds:
                    records.append({"condition": "alignment_cost", "state": state, "ref": ref, "target": seed,
                                    "align": self.align_kind, "cost": self.alignment(ref, seed, state)[1],
                                    "A": self.alignment(ref, seed, state)[0].describe()})
            # Individual maps: self IIA and transfer of phi_n to every other model m via A_{m->n}.
            for n in seeds:
                for init in range(c["n_inits"]):
                    phi, info = self.individual_phi(n, state, init)
                    for m in seeds:
                        A = self.alignment(n, m, state)[0]
                        res, corr = evaluate(phi, (self.get_models(m)[state], A), test, vd, layer)
                        add(dict(res, condition="individual" if m == n else "single_model_transfer", state=state,
                                 fold=None, init=init, map_seed=n, target=m, source=m, role="self" if m == n else "unseen",
                                 align=self.align_kind, train_epochs=info["epochs"]), corr)
            for fold in folds:
                # Alignment only (phi = id).
                for m in seeds:
                    A = self.alignment(fold["ref"], m, state)[0]
                    res, corr = evaluate(IdentityMap(), (self.get_models(m)[state], A), test, vd, layer)
                    add(dict(res, condition="alignment_only", state=state, fold=fold["fold"], init=None,
                             map_seed=fold["ref"], target=m, source=m,
                             role="heldout" if m == fold["heldout"] else "train", align=self.align_kind), corr)
                for condition in ["shared_linear", "shared_nonlinear"]:
                    for init in range(c["n_inits"]):
                        phi, info = self.shared_phi(condition, fold, state, init)
                        members = {m: (self.get_models(m)[state], self.alignment(fold["ref"], m, state)[0]) for m in seeds}
                        for m in seeds:
                            role = "heldout" if m == fold["heldout"] else "train"
                            res, corr = evaluate(phi, members[m], test, vd, layer)
                            add(dict(res, condition=condition, state=state, fold=fold["fold"], init=init,
                                     map_seed=fold["ref"], target=m, source=m, role=role, align=self.align_kind,
                                     train_epochs=info["epochs"], train_updates=info["updates"],
                                     train_examples=info["examples_processed"]), corr)
                            # §9 cross-model swap (evaluation only): base from m, all sources from n.
                            for n in seeds:
                                if n == m:
                                    continue
                                res, corr = evaluate(phi, members[m], test, vd, layer, source=members[n])
                                add(dict(res, condition=f"{condition}_cross", state=state, fold=fold["fold"],
                                         init=init, map_seed=fold["ref"], target=m, source=n,
                                         role="involves_heldout" if fold["heldout"] in (m, n) else "train_pair",
                                         align=self.align_kind), corr)
        with open(os.path.join(self.run_dir, "results.jsonl"), "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        torch.save(preds, os.path.join(self.run_dir, "predictions.pt"))
        self.log(f"Wrote {len(records)} records to {os.path.join(self.run_dir, 'results.jsonl')}")

    def stage_summarize(self):
        import pandas as pd
        path = os.path.join(self.run_dir, "results.jsonl")
        df = pd.DataFrame([json.loads(l) for l in open(path)])
        seeds = self.cfg["main_seeds"]
        out = [f"# Shared mapping results ({'quick' if self.cfg is QUICK else 'full'}, align={self.align_kind})", ""]

        def table(frame):
            return frame.to_markdown(floatfmt=".4f") if hasattr(frame, "to_markdown") else frame.to_string()

        acc = df[df.condition == "normal_accuracy"].pivot(index="target", columns="state", values="acc")
        out += ["## Normal task accuracy (test)", "", table(acc), ""]

        iia = df[df.condition.isin(["individual", "single_model_transfer", "alignment_only",
                                    "shared_linear", "shared_nonlinear"])].copy()
        # Per (condition, state, target model as unseen/self): average over inits first (§8.1).
        unseen = iia[iia.role.isin(["self", "heldout"])]
        per_seed = unseen.groupby(["condition", "state", "target"]).iia.mean().unstack("target")[seeds]
        out += ["## IIA on the evaluated model (individual: own model; others: held-out model of the fold)",
                "", "Averaged over map initialisations. Columns are model seeds.", "", table(per_seed), ""]

        # single_model_transfer restricted to the fold reference -> held-out (plan §14.7 priority 1).
        ref_of = {f["heldout"]: f["ref"] for f in self.folds(seeds)}
        smt = iia[(iia.condition == "single_model_transfer")]
        smt = smt[smt.apply(lambda r: ref_of[r.target] == r.map_seed, axis=1)]
        smt_seed = smt.groupby(["state", "target"]).iia.mean().unstack("target")[seeds]
        out += ["## Single-model phi transfer: phi_ref (trained on fold reference only) -> held-out model", "",
                table(smt_seed), ""]

        train_side = iia[iia.role == "train"].groupby(["condition", "state"]).iia.agg(["mean", "min", "max"])
        out += ["## IIA on the models used for training the shared map", "", table(train_side), ""]

        summary_rows = []
        for cond, frame in [("individual", per_seed.loc["individual"]),
                            ("single_model_transfer(ref)", smt_seed),
                            ("alignment_only", per_seed.loc["alignment_only"]),
                            ("shared_linear", per_seed.loc["shared_linear"]),
                            ("shared_nonlinear", per_seed.loc["shared_nonlinear"])]:
            for state in STATES:
                v = frame.loc[state].values
                summary_rows.append({"condition": cond, "state": state, "mean": v.mean(), "min": v.min(), "max": v.max()})
            d = frame.loc["trained"].values - frame.loc["random"].values
            summary_rows.append({"condition": cond, "state": "trained - random", "mean": d.mean(), "min": d.min(), "max": d.max()})
        out += ["## Summary over the 5 models (folds are not independent; §8.2)", "",
                table(pd.DataFrame(summary_rows).set_index(["condition", "state"])), ""]

        # Every transfer IIA is reported next to the individual IIA of the same target model and their
        # difference, because individual fits reach different levels in the trained and random states.
        gap_rows = []
        transfer = {"shared_nonlinear": per_seed.loc["shared_nonlinear"], "shared_linear": per_seed.loc["shared_linear"],
                    "single_model_transfer(ref)": smt_seed, "alignment_only": per_seed.loc["alignment_only"]}
        for cond, frame in transfer.items():
            for state in STATES:
                ind, tr = per_seed.loc["individual"].loc[state], frame.loc[state]
                for label, v in [("individual", ind), ("held-out", tr), ("individual - held-out", ind - tr)]:
                    gap_rows.append(dict({"condition": cond, "state": state, "quantity": label},
                                         **{str(s): v[s] for s in seeds}, mean=v.mean(), min=v.min(), max=v.max()))
        out += ["## Held-out IIA next to the individual IIA of the same model (difference can be negative)", "",
                table(pd.DataFrame(gap_rows).set_index(["condition", "state", "quantity"])), ""]

        cols = ["iia_left", "iia_right", "iia_both", "iia_label_changed", "iia_label_unchanged"]
        br = unseen.groupby(["condition", "state"])[cols].mean()
        out += ["## Breakdown on the evaluated model (mean over models and inits)", "", table(br), ""]

        cross = df[df.condition.str.endswith("_cross", na=False)]
        if len(cross):
            ct = cross.groupby(["condition", "state", "role"]).iia.agg(["mean", "min", "max"])
            out += ["## Cross-model swap (§9, evaluation only): base model m, sources from model n", "", table(ct), ""]

        ac = df[df.condition == "alignment_cost"].groupby(["state", "ref", "target"]).cost.first().unstack("target")
        out += [f"## Alignment cost ({self.align_kind}) to the reference model", "", table(ac), ""]

        text = "\n".join(out)
        with open(os.path.join(self.run_dir, "summary.md"), "w") as f:
            f.write(text)
        print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["dev", "train", "evaluate", "summarize", "all"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--align", choices=["permutation", "orthogonal"], default="permutation")
    parser.add_argument("--quick", action="store_true", help="tiny sizes for a smoke test")
    args = parser.parse_args()
    exp = Experiment(args)
    exp.log(f"=== stage {args.stage} (quick={args.quick}, align={args.align}, device={args.device}) ===")
    if args.stage == "dev":
        exp.stage_dev()
    if args.stage in ("train", "all"):
        exp.stage_train()
    if args.stage in ("evaluate", "all"):
        exp.stage_evaluate()
    if args.stage in ("summarize", "all"):
        exp.stage_summarize()


if __name__ == "__main__":
    main()
