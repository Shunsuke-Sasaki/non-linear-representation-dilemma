#!/usr/bin/env python
"""Diagnose held-out alignment adaptation for random fold 1 and a trained control."""
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
from Joint_Alignment_PoC import TrainableOrthogonalAlignment
from Shared_Mapping import Experiment, stable_seed

from das.Shared_Mapping import Alignment, evaluate, freeze, intervene_logits


FOLD = 1
INIT = 1
HELDOUT = 3837
N_EXAMPLES = 10000
BATCH_SIZE = 256
LEARNING_RATE = 0.001
VALIDATION_INTERVAL = 200
STANDARD_UPDATES = 2000
EXTENDED_UPDATES = 10000
RANDOM_ORTHOGONAL_SEEDS = (11001, 11002, 11003)
STATES = ("random", "trained")


def random_orthogonal_alignment(n, seed, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn((n, n), generator=generator)
    q, r = torch.linalg.qr(matrix)
    signs = torch.where(torch.diag(r) < 0, -1.0, 1.0)
    q = q * signs
    return Alignment(n, matrix=q).to(device)


def metrics(phi, model, alignment, data, exp):
    return evaluate(
        phi, (model, alignment), data,
        exp.cfg["variable_dims"], exp.cfg["layer"],
    )[0]


def atomic_json(path, content):
    temporary = path + ".tmp"
    with open(temporary, "w") as output:
        json.dump(content, output, indent=2)
    os.replace(temporary, path)


def run_adaptation(
    exp, state, phi, model, initial_alignment, train_data, dev_data, test_data,
    train_ids, test_ids, run_seed, max_updates, run_name, output_dir,
):
    alignment = TrainableOrthogonalAlignment(initial_alignment).to(exp.device)
    initial_matrix = alignment.matrix().detach().clone()
    for parameter in phi.parameters():
        parameter.requires_grad_(False)
    phi.eval()
    freeze(model)

    selected_train_data = train_data.subset(train_ids)
    selected_test_data = test_data.subset(test_ids)
    optimizer = optim.Adam(alignment.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator(device="cpu").manual_seed(run_seed)
    best_state = copy.deepcopy(alignment.state_dict())
    best_dev_loss = float("inf")
    best_update = 0
    previous_matrix = alignment.matrix().detach().clone()
    window_grad_norms = []
    window_batch_losses = []
    curve = []

    def measure(update):
        nonlocal best_state, best_dev_loss, best_update, previous_matrix
        train_metrics = metrics(phi, model, alignment, selected_train_data, exp)
        dev_metrics = metrics(phi, model, alignment, dev_data, exp)
        current_matrix = alignment.matrix().detach()
        delta = current_matrix - previous_matrix
        total_delta = current_matrix - initial_matrix
        record = {
            "update": update,
            "adapt_train_loss": train_metrics["loss"],
            "adapt_train_iia": train_metrics["iia"],
            "adapt_dev_loss": dev_metrics["loss"],
            "adapt_dev_iia": dev_metrics["iia"],
            "gradient_norm_mean_since_previous_measurement": (
                sum(window_grad_norms) / len(window_grad_norms) if window_grad_norms else None
            ),
            "gradient_norm_max_since_previous_measurement": (
                max(window_grad_norms) if window_grad_norms else None
            ),
            "batch_loss_mean_since_previous_measurement": (
                sum(window_batch_losses) / len(window_batch_losses) if window_batch_losses else None
            ),
            "alignment_update_frobenius_since_previous_measurement": torch.linalg.vector_norm(delta).item(),
            "alignment_update_frobenius_from_initial": torch.linalg.vector_norm(total_delta).item(),
            "orthogonality_max_error": (
                current_matrix.T @ current_matrix
                - torch.eye(current_matrix.shape[0], device=current_matrix.device)
            ).abs().max().item(),
        }
        curve.append(record)
        window_grad_norms.clear()
        window_batch_losses.clear()
        previous_matrix = current_matrix.clone()
        if dev_metrics["loss"] < best_dev_loss:
            best_dev_loss = dev_metrics["loss"]
            best_update = update
            best_state = copy.deepcopy(alignment.state_dict())

    started = time.time()
    measure(0)
    for update in range(1, max_updates + 1):
        selected = torch.randint(N_EXAMPLES, (BATCH_SIZE,), generator=generator)
        batch = train_data.subset(train_ids[selected.to(train_ids.device)])
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(
            intervene_logits(
                phi, (model, alignment), batch,
                exp.cfg["variable_dims"], exp.cfg["layer"],
            ),
            batch.label,
        )
        loss.backward()
        gradient_norm = torch.linalg.vector_norm(alignment.skew.grad).item()
        window_grad_norms.append(gradient_norm)
        window_batch_losses.append(loss.item())
        optimizer.step()

        if update % VALIDATION_INTERVAL == 0 or update == max_updates:
            measure(update)
            atomic_json(
                os.path.join(output_dir, f"{run_name}.curve.json"),
                {"state": state, "run": run_name, "curve": curve},
            )
            print(
                f"{state}/{run_name}: update={update}/{max_updates} "
                f"train_loss={curve[-1]['adapt_train_loss']:.4f} "
                f"train_iia={curve[-1]['adapt_train_iia']:.4f} "
                f"dev_loss={curve[-1]['adapt_dev_loss']:.4f} "
                f"dev_iia={curve[-1]['adapt_dev_iia']:.4f} "
                f"grad_norm={curve[-1]['gradient_norm_mean_since_previous_measurement']:.4g} "
                f"alignment_delta={curve[-1]['alignment_update_frobenius_from_initial']:.4g}",
                flush=True,
            )

    alignment.load_state_dict(best_state)
    test_metrics = metrics(phi, model, alignment, selected_test_data, exp)
    best_matrix = alignment.matrix().detach().cpu()
    checkpoint_path = os.path.join(output_dir, f"{run_name}.pt")
    torch.save(
        {
            "state": state,
            "run": run_name,
            "best_update": best_update,
            "best_dev_loss": best_dev_loss,
            "alignment_matrix": best_matrix,
            "test_metrics": test_metrics,
            "curve": curve,
        },
        checkpoint_path,
    )
    return {
        "state": state,
        "run": run_name,
        "initialization": "normal_activity_orthogonal" if run_name.startswith("normal_") else "random_orthogonal",
        "max_updates": max_updates,
        "run_seed": run_seed,
        "best_update_selected_by_dev_loss": best_update,
        "best_dev_loss": best_dev_loss,
        "best_dev_iia": next(row["adapt_dev_iia"] for row in curve if row["update"] == best_update),
        "best_train_loss": next(row["adapt_train_loss"] for row in curve if row["update"] == best_update),
        "best_train_iia": next(row["adapt_train_iia"] for row in curve if row["update"] == best_update),
        "test_iia_at_dev_selected_checkpoint": test_metrics["iia"],
        "test_loss_at_dev_selected_checkpoint": test_metrics["loss"],
        "final_curve_orthogonality_max_error": curve[-1]["orthogonality_max_error"],
        "best_checkpoint": os.path.relpath(checkpoint_path, os.path.dirname(output_dir)),
        "seconds": time.time() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    exp = Experiment(argparse.Namespace(quick=False, device=args.device, align="orthogonal"))
    fold = exp.folds(exp.cfg["main_seeds"])[FOLD]
    if fold["heldout"] != HELDOUT:
        raise RuntimeError(f"Expected fold {FOLD} to hold out {HELDOUT}, got {fold['heldout']}")

    output_dir = os.path.join(exp.root, "adaptation_diagnostic_random_fold1_init1_v2")
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.json")
    config = {
        "fold": FOLD,
        "heldout_seed": HELDOUT,
        "reference_seed": fold["ref"],
        "shared_map_init": INIT,
        "states": list(STATES),
        "conditions_per_state": [
            {"name": "normal_2k", "initialization": "normal_activity_orthogonal", "max_updates": STANDARD_UPDATES},
            *[
                {"name": f"random_orthogonal_{seed}", "initialization": "random_orthogonal",
                 "orthogonal_seed": seed, "max_updates": STANDARD_UPDATES}
                for seed in RANDOM_ORTHOGONAL_SEEDS
            ],
            {"name": "normal_10k", "initialization": "normal_activity_orthogonal", "max_updates": EXTENDED_UPDATES},
        ],
        "adapt_examples": N_EXAMPLES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "validation_interval": VALIDATION_INTERVAL,
        "checkpoint_selection": "minimum adapt-dev cross-entropy; final test evaluated once per run",
        "data_indices": "reused fold1 adaptation_controls_random IDs; adapt-train and test IDs identical across states",
    }
    if os.path.exists(results_path):
        previous = json.loads(open(results_path).read())
        if previous.get("config") != config:
            raise RuntimeError(f"Diagnostic output exists with a different config: {results_path}")
        raise RuntimeError(f"Diagnostic is already complete: {results_path}")
    atomic_json(results_path, {"config": config, "results": []})

    base = exp.root
    train_data = exp.path("joint_poc", "data", "adapt_train.pt")
    dev_data = exp.path("joint_poc", "data", "adapt_dev.pt")
    from das.Shared_Mapping import InterventionSet
    adapt_train = InterventionSet.load(train_data, exp.device)
    adapt_dev = InterventionSet.load(dev_data, exp.device)
    test = exp.intervention_data()["test"]

    index_path = os.path.join(base, "adaptation_controls_random", f"fold{FOLD}", "data_indices.pt")
    indices = torch.load(index_path, map_location="cpu", weights_only=False)
    train_ids = indices["adapt_train_ids"].to(exp.device)
    test_ids = indices["test_ids"].to(exp.device)
    if len(train_ids) != N_EXAMPLES or len(test_ids) != len(test):
        raise RuntimeError("Saved fold-1 adaptation/test indices have unexpected sizes")

    result_rows = []
    for state in STATES:
        model = exp.get_models(HELDOUT)[state]
        normal_alignment, alignment_cost = exp.alignment(
            fold["ref"], HELDOUT, state, kind="orthogonal"
        )
        checkpoint_path = (
            os.path.join(base, "adaptation_controls_random", f"fold{FOLD}", state,
                         f"shared_joint_init{INIT}.pt")
            if state == "random"
            else os.path.join(base, "adaptation_controls_v2", f"fold{FOLD}",
                              f"trained_shared_joint_init{INIT}.pt")
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Missing shared-joint checkpoint: {checkpoint_path}")
        saved = torch.load(checkpoint_path, map_location=exp.device, weights_only=False)
        phi = exp.make_phi(
            "revnet", stable_seed("shared_nonlinear", "orthogonal", FOLD, state, INIT)
        )
        phi.load_state_dict(saved["phi"])
        phi.eval()
        freeze(model)

        starts = [("normal_2k", normal_alignment, STANDARD_UPDATES, 21001)]
        starts.extend(
            (
                f"random_orthogonal_{seed}",
                random_orthogonal_alignment(exp.cfg["hidden"], seed, exp.device),
                STANDARD_UPDATES,
                seed + 30000,
            )
            for seed in RANDOM_ORTHOGONAL_SEEDS
        )
        starts.append(("normal_10k", normal_alignment, EXTENDED_UPDATES, 21001))
        for name, initial, updates, run_seed in starts:
            state_output_dir = os.path.join(output_dir, state)
            os.makedirs(state_output_dir, exist_ok=True)
            run = run_adaptation(
                exp, state, phi, model, initial, adapt_train, adapt_dev, test,
                train_ids, test_ids, run_seed, updates, name, state_output_dir,
            )
            run["shared_checkpoint"] = os.path.relpath(checkpoint_path, base)
            run["alignment_cost"] = alignment_cost
            result_rows.append(run)
            atomic_json(results_path, {"config": config, "results": result_rows})
            print(
                f"Completed {state}/{name}: best_update={run['best_update_selected_by_dev_loss']} "
                f"best_dev_iia={run['best_dev_iia']:.4f} "
                f"test_iia={run['test_iia_at_dev_selected_checkpoint']:.4f}",
                flush=True,
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Wrote diagnostic results: {results_path}")


if __name__ == "__main__":
    main()
