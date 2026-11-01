#!/usr/bin/env python
"""Pilot fixed-vs-joint alignment learning and held-out alignment adaptation."""
import argparse
import copy
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Shared_Mapping import Experiment, stable_seed

from das.Shared_Mapping import evaluate, freeze, intervene_logits, train_mapping


class TrainableOrthogonalAlignment(nn.Module):
    """Learn a right-side orthogonal update from an estimated Procrustes alignment."""

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


def train_joint_mapping(exp, phi, members, train, dev, init_seed, log):
    for model, _ in members:
        freeze(model)
    trainable_alignments = [alignment for _, alignment in members[1:]]
    params = list(phi.parameters())
    for alignment in trainable_alignments:
        params.extend(alignment.parameters())
    optimizer = optim.Adam(params, lr=exp.cfg["map_lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=exp.cfg["map_sched_patience"])
    best_loss, best_state, best_bad, history = float("inf"), None, 0, []
    best_alignments = None
    total_updates = 0
    batch_size = exp.cfg["map_batch"] // len(members)
    generator = torch.Generator(device="cpu").manual_seed(init_seed + 1)
    for epoch in range(exp.cfg["map_max_epochs"]):
        phi.train()
        order = torch.randperm(len(train), generator=generator).to(train.base.device)
        total_loss, batches = 0.0, 0
        for start in range(0, len(train), batch_size):
            subset = train.subset(order[start:start + batch_size])
            optimizer.zero_grad()
            losses = [
                nn.functional.cross_entropy(
                    intervene_logits(phi, member, subset, exp.cfg["variable_dims"], exp.cfg["layer"]),
                    subset.label)
                for member in members
            ]
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
            total_updates += 1

        dev_stats = [
            evaluate(phi, member, dev, exp.cfg["variable_dims"], exp.cfg["layer"])[0]
            for member in members
        ]
        dev_loss = sum(item["loss"] for item in dev_stats) / len(dev_stats)
        scheduler.step(total_loss / batches)
        if dev_loss < best_loss:
            improvement = best_loss - dev_loss
            best_loss = dev_loss
            best_state = copy.deepcopy(phi.state_dict())
            best_alignments = [copy.deepcopy(alignment.state_dict()) for alignment in trainable_alignments]
            best_bad = 0 if improvement > exp.cfg["map_threshold"] else best_bad + 1
        else:
            best_bad += 1
        history.append({
            "epoch": epoch + 1,
            "train_loss": total_loss / batches,
            "dev_loss": dev_loss,
            "dev_iia_per_member": [item["iia"] for item in dev_stats],
            "lr": optimizer.param_groups[0]["lr"],
            "updates": total_updates,
            "orthogonality_max_error": max(
                (alignment.matrix().T @ alignment.matrix()
                 - torch.eye(alignment.initial.shape[0], device=alignment.initial.device)).abs().max().item()
                for alignment in trainable_alignments
            ) if trainable_alignments else 0.0,
        })
        log(f"  joint epoch {epoch + 1}: train_loss={total_loss / batches:.4f} "
            f"dev_loss={dev_loss:.4f} dev_iia={[round(item['iia'], 4) for item in dev_stats]} "
            f"bad={best_bad}")
        if best_bad >= exp.cfg["map_patience"]:
            break
    if best_state is None:
        raise RuntimeError("Joint mapping training did not produce a valid checkpoint")
    phi.load_state_dict(best_state)
    for alignment, state in zip(trainable_alignments, best_alignments):
        alignment.load_state_dict(state)
    phi.eval()
    return {
        "best_dev_loss": best_loss,
        "epochs": len(history),
        "updates": total_updates,
        "history": history,
    }


def adapt_heldout(phi, model, initial_alignment, data, variable_dims, layer, ids, updates, batch_size, seed):
    alignment = TrainableOrthogonalAlignment(initial_alignment).to(data.base.device)
    optimizer = optim.Adam(alignment.parameters(), lr=0.001)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for parameter in phi.parameters():
        parameter.requires_grad_(False)
    for _ in range(updates):
        selected = torch.randint(len(ids), (batch_size,), generator=generator)
        subset = data.subset(ids[selected].to(ids.device))
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(
            intervene_logits(phi, (model, alignment), subset, variable_dims, layer), subset.label)
        loss.backward()
        optimizer.step()
    matrix = alignment.matrix()
    orthogonality_error = (matrix.T @ matrix - torch.eye(matrix.shape[0], device=matrix.device)).abs().max().item()
    if orthogonality_error > 1e-4:
        raise RuntimeError(f"Adapted alignment violated orthogonality: {orthogonality_error}")
    return alignment, orthogonality_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--init", type=int, default=0)
    parser.add_argument("--states", default="trained,random")
    parser.add_argument("--adapt-sizes", default="0,256,1024,4096,10000")
    parser.add_argument("--adapt-updates", type=int, default=2000)
    parser.add_argument("--adapt-batch-size", type=int, default=256)
    args = parser.parse_args()
    states = args.states.split(",")
    adapt_sizes = [int(value) for value in args.adapt_sizes.split(",")]
    if any(state not in ("trained", "random") for state in states):
        parser.error("--states accepts trained and/or random")
    if any(size < 0 or size > 10000 for size in adapt_sizes):
        parser.error("adaptation sizes must be between 0 and 10000")
    if args.adapt_updates < 1 or args.adapt_batch_size < 1:
        parser.error("adaptation updates and batch size must be positive")

    exp = Experiment(argparse.Namespace(quick=False, device=args.device, align="orthogonal"))
    seeds = exp.cfg["main_seeds"]
    if args.fold < 0 or args.fold >= len(seeds):
        parser.error(f"--fold must be between 0 and {len(seeds) - 1}")
    heldout = seeds[args.fold]
    training_seeds = [seed for seed in seeds if seed != heldout]
    ref_seed = training_seeds[0]
    output_dir = os.path.join(exp.root, "joint_poc", f"fold{args.fold}_init{args.init}")
    os.makedirs(output_dir, exist_ok=True)
    exp.log(f"Joint-alignment PoC: fold={args.fold} ref={ref_seed} heldout={heldout} states={states}")
    intervention = exp.intervention_data()
    calibration = exp.calibration_inputs()
    results = []

    for state in states:
        members = []
        initial_alignments = {}
        for seed in training_seeds:
            model = exp.get_models(seed)[state]
            alignment, _ = exp.alignment(ref_seed, seed, state, kind="orthogonal")
            initial_alignments[seed] = alignment
            if seed == ref_seed:
                members.append((model, alignment))
            else:
                members.append((model, TrainableOrthogonalAlignment(alignment).to(exp.device)))

        init_seed = stable_seed("shared_nonlinear", "orthogonal", args.fold, state, args.init)
        condition_phis = {}
        for condition in ("fixed", "joint"):
            exp.log(f"Training {condition} mapping: state={state}, init_seed={init_seed}")
            phi = exp.make_phi("revnet", init_seed)
            train_started = time.time()
            if condition == "fixed":
                fixed_members = [
                    (model, initial_alignments[seed])
                    for model, seed in zip([item[0] for item in members], training_seeds)
                ]
                info = train_mapping(
                    phi, fixed_members, intervention["train"], intervention["dev"],
                    exp.cfg["variable_dims"], exp.cfg["layer"],
                    ids_per_member=exp.cfg["map_batch"] // len(fixed_members),
                    max_epochs=exp.cfg["map_max_epochs"], patience=exp.cfg["map_patience"],
                    improve_threshold=exp.cfg["map_threshold"], lr=exp.cfg["map_lr"],
                    scheduler_patience=exp.cfg["map_sched_patience"],
                    generator=torch.Generator(device="cpu").manual_seed(init_seed + 1), log=exp.log)
            else:
                joint_members = []
                for seed, (model, _) in zip(training_seeds, members):
                    alignment = initial_alignments[seed] if seed == ref_seed else next(
                        item[1] for item, member_seed in zip(members, training_seeds) if member_seed == seed)
                    joint_members.append((model, alignment))
                info = train_joint_mapping(
                    exp, phi, joint_members, intervention["train"], intervention["dev"], init_seed, exp.log)
            info["seconds"] = time.time() - train_started
            info["init_seed"] = init_seed
            checkpoint = os.path.join(output_dir, f"{state}_{condition}_phi.pt")
            alignment_state = {
                str(seed): (
                    alignment.matrix().detach().cpu()
                    if isinstance(alignment, TrainableOrthogonalAlignment)
                    else alignment.Q.detach().cpu()
                )
                for seed, (_, alignment) in zip(training_seeds, members if condition == "joint" else fixed_members)
                if seed != ref_seed
            }
            torch.save({"phi": phi.state_dict(), "info": info, "alignments": alignment_state}, checkpoint)
            exp.log(f"Saved {checkpoint}; {info['epochs']} epochs, {info['seconds']:.1f}s")
            condition_phis[condition] = phi

        heldout_model = exp.get_models(heldout)[state]
        initial_test_alignment, alignment_cost = exp.alignment(ref_seed, heldout, state, kind="orthogonal")
        id_generator = torch.Generator(device="cpu").manual_seed(
            stable_seed("joint_poc_adaptation_ids", args.fold))
        adaptation_order = torch.randperm(len(intervention["train"]), generator=id_generator)
        adaptation_order = adaptation_order[:max(adapt_sizes)].to(exp.device)
        for condition, phi in condition_phis.items():
            for n_examples in adapt_sizes:
                if n_examples == 0:
                    adapted_alignment = initial_test_alignment
                    orthogonality_error = 0.0
                else:
                    adapted_alignment, orthogonality_error = adapt_heldout(
                        phi, heldout_model, initial_test_alignment, intervention["train"],
                        exp.cfg["variable_dims"], exp.cfg["layer"],
                        adaptation_order[:n_examples], args.adapt_updates, args.adapt_batch_size,
                        stable_seed("joint_poc_adapt", args.fold, state, n_examples))
                metrics, _ = evaluate(
                    phi, (heldout_model, adapted_alignment), intervention["test"],
                    exp.cfg["variable_dims"], exp.cfg["layer"])
                row = {
                    "fold": args.fold, "reference": ref_seed, "heldout": heldout,
                    "state": state, "condition": condition, "phi_init_seed": init_seed,
                    "adapt_examples": n_examples,
                    "adapt_updates": 0 if n_examples == 0 else args.adapt_updates,
                    "adapt_batch_size": args.adapt_batch_size,
                    "alignment_cost": alignment_cost,
                    "orthogonality_max_error": orthogonality_error,
                    **metrics,
                }
                results.append(row)
                exp.log(f"PoC result: {state}/{condition}, n={n_examples}, "
                        f"test IIA={metrics['iia']:.4f}, loss={metrics['loss']:.4f}")
        del members, condition_phis
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result_path = os.path.join(output_dir, "results.json")
    with open(result_path, "w") as result_file:
        json.dump({
            "config": {
                "fold": args.fold, "init": args.init, "states": states,
                "adapt_sizes": adapt_sizes, "adapt_updates": args.adapt_updates,
                "adapt_batch_size": args.adapt_batch_size, "device": args.device,
            },
            "results": results,
        }, result_file, indent=2)
    exp.log(f"Wrote {result_path}")


if __name__ == "__main__":
    main()
