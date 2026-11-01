#!/usr/bin/env python
"""Diagnostic that adapts both the shared map and alignment on random fold 1."""
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

from das.Shared_Mapping import evaluate, freeze, intervene_logits


FOLD = 1
HELDOUT = 3837
INIT = 1
N_EXAMPLES = 10000
MAX_UPDATES = 10000
BATCH_SIZE = 256
VALIDATION_INTERVAL = 200
ALIGNMENT_LR = 0.001
ADAPT_SEED = 21001


def evaluate_split(phi, model, alignment, data, exp):
    return evaluate(
        phi, (model, alignment), data,
        exp.cfg["variable_dims"], exp.cfg["layer"],
    )[0]


def atomic_json(path, value):
    temporary = path + ".tmp"
    with open(temporary, "w") as output:
        json.dump(value, output, indent=2)
    os.replace(temporary, path)


def parameter_delta_norm(parameters, initial_state):
    squared_norm = 0.0
    for name, parameter in parameters.items():
        difference = parameter.detach() - initial_state[name].to(parameter.device)
        squared_norm += torch.sum(difference * difference).item()
    return squared_norm ** 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    exp = Experiment(argparse.Namespace(quick=False, device=args.device, align="orthogonal"))
    fold = exp.folds(exp.cfg["main_seeds"])[FOLD]
    if fold["heldout"] != HELDOUT:
        raise RuntimeError(f"Expected fold {FOLD} to hold out {HELDOUT}, got {fold['heldout']}")

    output_dir = os.path.join(exp.root, "adaptation_diagnostic_random_fold1_joint_update_10k")
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.json")
    config = {
        "fold": FOLD,
        "heldout_seed": HELDOUT,
        "shared_map_init": INIT,
        "state": "random",
        "map_initialization": "fold1 random shared_joint_init1 checkpoint",
        "alignment_initialization": "normal-activity orthogonal alignment",
        "adapt_examples": N_EXAMPLES,
        "max_updates": MAX_UPDATES,
        "batch_size": BATCH_SIZE,
        "map_learning_rate": exp.cfg["map_lr"],
        "alignment_learning_rate": ALIGNMENT_LR,
        "validation_interval": VALIDATION_INTERVAL,
        "checkpoint_selection": "minimum adapt-dev cross-entropy; final test evaluated once",
        "data_indices": "reused fold1 adaptation_controls_random IDs",
        "scope": "diagnostic only; both map and alignment use adapt-train labels",
    }
    if os.path.exists(results_path):
        previous = json.loads(open(results_path).read())
        if previous.get("config") != config:
            raise RuntimeError(f"Diagnostic output exists with a different config: {results_path}")
        raise RuntimeError(f"Diagnostic is already complete: {results_path}")
    atomic_json(results_path, {"config": config, "history": [], "result": None})

    from das.Shared_Mapping import InterventionSet

    adapt_train = InterventionSet.load(
        exp.path("joint_poc", "data", "adapt_train.pt"), exp.device
    )
    adapt_dev = InterventionSet.load(
        exp.path("joint_poc", "data", "adapt_dev.pt"), exp.device
    )
    test = exp.intervention_data()["test"]
    ids_path = os.path.join(
        exp.root, "adaptation_controls_random", f"fold{FOLD}", "data_indices.pt"
    )
    ids = torch.load(ids_path, map_location="cpu", weights_only=False)
    train_ids = ids["adapt_train_ids"].to(exp.device)
    test_ids = ids["test_ids"].to(exp.device)
    if len(train_ids) != N_EXAMPLES or len(test_ids) != len(test):
        raise RuntimeError("Saved fold-1 adaptation/test indices have unexpected sizes")

    checkpoint_path = os.path.join(
        exp.root, "adaptation_controls_random", f"fold{FOLD}", "random",
        f"shared_joint_init{INIT}.pt",
    )
    saved = torch.load(checkpoint_path, map_location=exp.device, weights_only=False)
    phi = exp.make_phi(
        "revnet", stable_seed("shared_nonlinear", "orthogonal", FOLD, "random", INIT)
    )
    phi.load_state_dict(saved["phi"])
    initial_phi_state = {
        name: parameter.detach().clone()
        for name, parameter in phi.named_parameters()
    }
    phi.train()
    model = exp.get_models(HELDOUT)["random"]
    freeze(model)
    initial_alignment, alignment_cost = exp.alignment(
        fold["ref"], HELDOUT, "random", kind="orthogonal"
    )
    alignment = TrainableOrthogonalAlignment(initial_alignment).to(exp.device)
    initial_matrix = alignment.matrix().detach().clone()
    initial_alignment_state = copy.deepcopy(alignment.state_dict())

    optimizer = optim.Adam(
        [
            {"params": list(phi.parameters()), "lr": exp.cfg["map_lr"]},
            {"params": list(alignment.parameters()), "lr": ALIGNMENT_LR},
        ]
    )
    batch_generator = torch.Generator(device="cpu").manual_seed(ADAPT_SEED)
    best_dev_loss = float("inf")
    best_update = 0
    best_phi_state = copy.deepcopy(phi.state_dict())
    best_alignment_state = copy.deepcopy(alignment.state_dict())
    history = []
    window_map_grad_norms = []
    window_alignment_grad_norms = []
    window_batch_losses = []
    started = time.time()

    def measure(update):
        nonlocal best_dev_loss, best_update, best_phi_state, best_alignment_state
        phi.eval()
        train_stats = evaluate_split(
            phi, model, alignment, adapt_train.subset(train_ids), exp
        )
        dev_stats = evaluate_split(phi, model, alignment, adapt_dev, exp)
        matrix = alignment.matrix().detach()
        record = {
            "update": update,
            "adapt_train_loss": train_stats["loss"],
            "adapt_train_iia": train_stats["iia"],
            "adapt_dev_loss": dev_stats["loss"],
            "adapt_dev_iia": dev_stats["iia"],
            "map_gradient_norm_mean_since_previous_measurement": (
                sum(window_map_grad_norms) / len(window_map_grad_norms)
                if window_map_grad_norms else None
            ),
            "alignment_gradient_norm_mean_since_previous_measurement": (
                sum(window_alignment_grad_norms) / len(window_alignment_grad_norms)
                if window_alignment_grad_norms else None
            ),
            "batch_loss_mean_since_previous_measurement": (
                sum(window_batch_losses) / len(window_batch_losses)
                if window_batch_losses else None
            ),
            "alignment_update_frobenius_from_initial": torch.linalg.vector_norm(
                matrix - initial_matrix
            ).item(),
            "map_parameter_update_l2_from_initial": parameter_delta_norm(
                dict(phi.named_parameters()), initial_phi_state
            ),
            "orthogonality_max_error": (
                matrix.T @ matrix
                - torch.eye(matrix.shape[0], device=matrix.device)
            ).abs().max().item(),
        }
        history.append(record)
        window_map_grad_norms.clear()
        window_alignment_grad_norms.clear()
        window_batch_losses.clear()
        if dev_stats["loss"] < best_dev_loss:
            best_dev_loss = dev_stats["loss"]
            best_update = update
            best_phi_state = copy.deepcopy(phi.state_dict())
            best_alignment_state = copy.deepcopy(alignment.state_dict())
        atomic_json(results_path, {"config": config, "history": history, "result": None})
        print(
            f"joint-update {update}/{MAX_UPDATES}: "
            f"train_loss={train_stats['loss']:.4f} train_iia={train_stats['iia']:.4f} "
            f"dev_loss={dev_stats['loss']:.4f} dev_iia={dev_stats['iia']:.4f} "
            f"map_grad={record['map_gradient_norm_mean_since_previous_measurement'] or 0:.4g} "
            f"alignment_grad={record['alignment_gradient_norm_mean_since_previous_measurement'] or 0:.4g}",
            flush=True,
        )
        phi.train()

    measure(0)
    for update in range(1, MAX_UPDATES + 1):
        selected = torch.randint(N_EXAMPLES, (BATCH_SIZE,), generator=batch_generator)
        batch = adapt_train.subset(train_ids[selected.to(train_ids.device)])
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(
            intervene_logits(
                phi, (model, alignment), batch,
                exp.cfg["variable_dims"], exp.cfg["layer"],
            ),
            batch.label,
        )
        loss.backward()
        map_grad_norm = torch.sqrt(sum(
            torch.sum(parameter.grad.detach() ** 2)
            for parameter in phi.parameters()
            if parameter.grad is not None
        )).item()
        alignment_grad_norm = torch.linalg.vector_norm(alignment.skew.grad).item()
        window_map_grad_norms.append(map_grad_norm)
        window_alignment_grad_norms.append(alignment_grad_norm)
        window_batch_losses.append(loss.item())
        optimizer.step()
        if update % VALIDATION_INTERVAL == 0 or update == MAX_UPDATES:
            measure(update)

    phi.load_state_dict(best_phi_state)
    alignment.load_state_dict(best_alignment_state)
    phi.eval()
    test_metrics = evaluate_split(
        phi, model, alignment, test.subset(test_ids), exp
    )
    best_matrix = alignment.matrix().detach()
    orthogonality_error = (
        best_matrix.T @ best_matrix
        - torch.eye(best_matrix.shape[0], device=best_matrix.device)
    ).abs().max().item()
    checkpoint_path = os.path.join(output_dir, "best_dev_checkpoint.pt")
    torch.save(
        {
            "phi": phi.state_dict(),
            "alignment": best_alignment_state,
            "best_update": best_update,
            "best_dev_loss": best_dev_loss,
            "test_metrics": test_metrics,
            "history": history,
            "alignment_cost": alignment_cost,
            "orthogonality_max_error": orthogonality_error,
        },
        checkpoint_path,
    )
    result = {
        "best_update_selected_by_dev_loss": best_update,
        "best_dev_loss": best_dev_loss,
        "best_dev_iia": next(row["adapt_dev_iia"] for row in history if row["update"] == best_update),
        "best_train_loss": next(row["adapt_train_loss"] for row in history if row["update"] == best_update),
        "best_train_iia": next(row["adapt_train_iia"] for row in history if row["update"] == best_update),
        "test_iia_at_dev_selected_checkpoint": test_metrics["iia"],
        "test_loss_at_dev_selected_checkpoint": test_metrics["loss"],
        "orthogonality_max_error": orthogonality_error,
        "seconds": time.time() - started,
        "checkpoint": os.path.basename(checkpoint_path),
    }
    atomic_json(results_path, {"config": config, "history": history, "result": result})
    print(f"joint-update result: {json.dumps(result)}", flush=True)


if __name__ == "__main__":
    main()
