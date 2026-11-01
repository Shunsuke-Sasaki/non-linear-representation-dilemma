#!/usr/bin/env python
"""Compare shared-map contributions to held-out alignment adaptation."""
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
from Joint_Alignment_PoC import TrainableOrthogonalAlignment, train_joint_mapping
from Shared_Mapping import Experiment, stable_seed

from das.Dataset_Generation import make_intervention_dataset_variable_intervention_all
from das.Shared_Mapping import (
    Alignment,
    IdentityMap,
    InterventionSet,
    evaluate,
    freeze,
    intervene_logits,
    seed_everything_for,
    train_mapping,
)

CONDITIONS = ("no_shared", "single_model", "shared_fixed", "shared_joint")
ADAPT_SEED = 20261011
ADAPT_DEV_SEED = 20261012
ADAPT_SIZE_OPTIONS = (0, 256, 1024, 4096, 10000)
ADAPT_UPDATES = 2000
ADAPT_BATCH_SIZE = 256
ADAPT_LR = 0.001
ADAPT_VALIDATION_INTERVAL = 200


def load_or_create_intervention_data(exp, name, size, seed):
    path = exp.path("joint_poc", "data", f"{name}.pt")
    if not os.path.exists(path):
        seed_everything_for(seed)
        raw = make_intervention_dataset_variable_intervention_all(size, exp.cfg["embedding_size"])
        InterventionSet.from_raw(raw, exp.cfg["embedding_size"]).save(path)
    return InterventionSet.load(path, exp.device)


def initial_members(exp, fold, state, train_seeds):
    members = []
    alignments = {}
    for seed in train_seeds:
        model = exp.get_models(seed)[state]
        alignment, _ = exp.alignment(fold["ref"], seed, state, kind="orthogonal")
        alignments[seed] = alignment
        members.append((model, alignment))
    return members, alignments


def train_joint_condition(exp, fold, state, init, members, alignments, data, output_dir):
    phi_seed = stable_seed("shared_nonlinear", "orthogonal", fold["fold"], state, init)
    state_dir = os.path.join(output_dir, state)
    os.makedirs(state_dir, exist_ok=True)
    phi_path = (
        os.path.join(output_dir, f"{state}_shared_joint_init{init}.pt")
        if state == "trained"
        else os.path.join(state_dir, f"shared_joint_init{init}.pt")
    )
    phi = exp.make_phi("revnet", phi_seed)
    if os.path.exists(phi_path):
        saved = torch.load(phi_path, map_location=exp.device, weights_only=False)
        phi.load_state_dict(saved["phi"])
        phi.eval()
        return phi, saved["info"], phi_path
    previous_joint_path = os.path.join(
        exp.root, "joint_poc", f"fold{fold['fold']}_init{init}", f"{state}_joint_phi.pt"
    )
    if os.path.exists(previous_joint_path):
        saved = torch.load(previous_joint_path, map_location=exp.device, weights_only=False)
        if saved["info"].get("init_seed") == phi_seed:
            phi.load_state_dict(saved["phi"])
            phi.eval()
            os.makedirs(output_dir, exist_ok=True)
            torch.save(saved, phi_path)
            return phi, saved["info"], phi_path

    joint_members = []
    trainable_by_seed = {}
    for seed, (model, _) in zip(fold["train"], members):
        initial = alignments[seed]
        alignment = initial if seed == fold["ref"] else TrainableOrthogonalAlignment(initial).to(exp.device)
        trainable_by_seed[seed] = alignment
        joint_members.append((model, alignment))

    started = time.time()
    info = train_joint_mapping(
        exp, phi, joint_members, data["train"], data["dev"], phi_seed, exp.log
    )
    info["seconds"] = time.time() - started
    info["init_seed"] = phi_seed
    trained_alignments = {
        str(seed): (
            alignment.matrix().detach().cpu()
            if isinstance(alignment, TrainableOrthogonalAlignment)
            else (
                alignment.Q.detach().cpu()
                if alignment.kind == "orthogonal"
                else torch.eye(alignment.n)
            )
        )
        for seed, alignment in trainable_by_seed.items()
    }
    torch.save({"phi": phi.state_dict(), "info": info, "alignments": trained_alignments}, phi_path)
    return phi, info, phi_path


def validate_adaptation(phi, model, alignment, data, exp):
    return evaluate(phi, (model, alignment), data, exp.cfg["variable_dims"], exp.cfg["layer"])[0]["loss"]


def adapt_alignment(phi, model, initial, train_data, dev_data, exp, example_ids, seed):
    alignment = TrainableOrthogonalAlignment(initial).to(exp.device)
    for parameter in phi.parameters():
        parameter.requires_grad_(False)
    phi.eval()
    n_examples = len(example_ids)
    if n_examples == 0:
        return alignment, 0, validate_adaptation(phi, model, alignment, dev_data, exp), 0.0

    indices = example_ids.to(exp.device)
    optimizer = optim.Adam(alignment.parameters(), lr=ADAPT_LR)
    best_state = copy.deepcopy(alignment.state_dict())
    best_dev_loss = validate_adaptation(phi, model, alignment, dev_data, exp)
    best_update, total_updates = 0, 0
    batch_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    for update in range(1, ADAPT_UPDATES + 1):
        selected = torch.randint(n_examples, (ADAPT_BATCH_SIZE,), generator=batch_generator)
        batch_ids = indices[selected.to(indices.device)]
        batch = train_data.subset(batch_ids)
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(
            intervene_logits(phi, (model, alignment), batch, exp.cfg["variable_dims"], exp.cfg["layer"]),
            batch.label,
        )
        loss.backward()
        optimizer.step()
        total_updates = update
        if update % ADAPT_VALIDATION_INTERVAL == 0 or update == ADAPT_UPDATES:
            dev_loss = validate_adaptation(phi, model, alignment, dev_data, exp)
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_state = copy.deepcopy(alignment.state_dict())
                best_update = update
    alignment.load_state_dict(best_state)
    matrix = alignment.matrix()
    error = (
        matrix.T @ matrix
        - torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    ).abs().max().item()
    if error > 1e-4:
        raise RuntimeError(f"Adapted alignment violated orthogonality: {error}")
    return alignment, best_update, best_dev_loss, error


def save_results(path, config, rows):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as output:
        json.dump({"config": config, "results": rows}, output, indent=2)
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--inits", default="0,1")
    parser.add_argument("--adapt-sizes", default="0,256,1024,4096,10000")
    parser.add_argument("--states", default="random", choices=("trained", "random", "trained,random"))
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()
    folds = [int(value) for value in args.folds.split(",")]
    inits = [int(value) for value in args.inits.split(",")]
    adapt_sizes = [int(value) for value in args.adapt_sizes.split(",")]
    states = args.states.split(",")
    if len(set(states)) != len(states):
        parser.error("--states must not contain duplicates")
    if not folds or any(fold not in range(5) for fold in folds):
        parser.error("--folds must be comma-separated fold indices from 0 to 4")
    if not inits or any(init not in (0, 1) for init in inits):
        parser.error("--inits must be a non-empty subset of 0,1")
    if not adapt_sizes or any(size not in ADAPT_SIZE_OPTIONS for size in adapt_sizes):
        parser.error(f"--adapt-sizes must use values from {ADAPT_SIZE_OPTIONS}")

    exp = Experiment(argparse.Namespace(quick=False, device=args.device, align="orthogonal"))
    output_name = args.output_name or (
        "adaptation_controls_v2" if states == ["trained"] else "adaptation_controls_random"
        if states == ["random"] else "adaptation_controls_trained_random"
    )
    output_root = os.path.join(exp.root, output_name)
    os.makedirs(output_root, exist_ok=True)
    output_path = os.path.join(output_root, "results.json")
    rows = []
    config = {
        "folds": folds,
        "inits": inits,
        "states": states,
        "conditions": list(CONDITIONS),
        "adapt_sizes": adapt_sizes,
        "adapt_updates": ADAPT_UPDATES,
        "adapt_batch_size": ADAPT_BATCH_SIZE,
        "adapt_learning_rate": ADAPT_LR,
        "adapt_validation_interval": ADAPT_VALIDATION_INTERVAL,
        "adapt_train_seed": ADAPT_SEED,
        "adapt_dev_seed": ADAPT_DEV_SEED,
        "adapt_subset_protocol": "one nested train-index permutation shared by all folds and conditions",
        "individual_maps": "Existing individual RevNet checkpoints for the fold reference.",
        "fixed_shared_maps": "Existing orthogonal shared nonlinear checkpoints.",
        "shared_joint_maps": "Trained in this script with the existing paired joint-training procedure.",
        "exploratory_fold": 0,
        "confirmatory_folds": [fold for fold in folds if fold != 0],
    }
    if os.path.exists(output_path):
        previous = json.loads(open(output_path).read())
        if previous.get("config") != config:
            raise RuntimeError(f"Existing results use a different config: {output_path}")
        rows = previous["results"]
    save_results(output_path, config, rows)

    iv_train = load_or_create_intervention_data(
        exp, "adapt_train", exp.cfg["iv_dev"], ADAPT_SEED
    )
    iv_dev = load_or_create_intervention_data(
        exp, "adapt_dev", exp.cfg["iv_dev"], ADAPT_DEV_SEED
    )
    test_data = exp.intervention_data()["test"]
    test_ids_generator = torch.Generator(device="cpu").manual_seed(
        stable_seed("adaptation_controls_test_ids")
    )

    for fold_index in folds:
        fold = exp.folds(exp.cfg["main_seeds"])[fold_index]
        heldout = fold["heldout"]
        fold_dir = os.path.join(output_root, f"fold{fold_index}")
        os.makedirs(fold_dir, exist_ok=True)

        for init in inits:
            common_order = torch.randperm(
                len(iv_train), generator=torch.Generator(device="cpu").manual_seed(
                    stable_seed("adaptation_controls_subset")
                )
            )[:max(adapt_sizes)].to(exp.device)
            test_indices = torch.randperm(
                len(test_data), generator=test_ids_generator
            ).to(exp.device)
            torch.save(
                {"adapt_train_ids": common_order.cpu(), "test_ids": test_indices.cpu()},
                os.path.join(fold_dir, "data_indices.pt"),
            )

            for state in states:
                model = exp.get_models(heldout)[state]
                train_members, alignments = initial_members(exp, fold, state, fold["train"])
                initial_test_alignment, alignment_cost = exp.alignment(
                    fold["ref"], heldout, state, kind="orthogonal"
                )
                map_specs = {
                    "no_shared": IdentityMap().to(exp.device),
                    "single_model": exp.individual_phi(fold["ref"], state, init)[0],
                    "shared_fixed": exp.make_phi("revnet", stable_seed(
                        "shared_nonlinear", "orthogonal", fold_index, state, init
                    )),
                }
                fixed_path = os.path.join(
                    exp.run_dir, "maps", "shared_nonlinear", state, f"fold{fold_index}_init{init}.pt"
                )
                if not os.path.exists(fixed_path):
                    raise FileNotFoundError(f"Missing fixed shared-map checkpoint: {fixed_path}")
                map_specs["shared_fixed"].load_state_dict(
                    torch.load(fixed_path, map_location=exp.device, weights_only=False)["phi"]
                )
                map_specs["shared_fixed"].eval()
                joint_phi, _, joint_path = train_joint_condition(
                    exp, fold, state, init, train_members, alignments,
                    exp.intervention_data(), fold_dir,
                )
                map_specs["shared_joint"] = joint_phi
                exp.log(f"Fold {fold_index}, state={state}, repeat {init}: joint checkpoint {joint_path}")

                individual_self_phi = exp.individual_phi(heldout, state, init)[0]
                individual_self_metrics, _ = evaluate(
                    individual_self_phi, (model, Alignment(exp.cfg["hidden"]).to(exp.device)),
                    test_data, exp.cfg["variable_dims"], exp.cfg["layer"],
                )
                for condition, phi in map_specs.items():
                    phi.eval()
                    for n_examples in adapt_sizes:
                        adaptation_seed = stable_seed("adaptation_controls_opt", fold_index, init, n_examples)
                        adapted, best_update, dev_loss, orthogonal_error = adapt_alignment(
                            phi, model, initial_test_alignment, iv_train, iv_dev, exp,
                            common_order[:n_examples],
                            adaptation_seed,
                        )
                        test_subset = test_data.subset(test_indices)
                        metrics, _ = evaluate(
                            phi, (model, adapted), test_subset,
                            exp.cfg["variable_dims"], exp.cfg["layer"],
                        )
                        row = {
                            "fold": fold_index,
                            "heldout": heldout,
                            "reference": fold["ref"],
                            "fold_role": "exploratory" if fold_index == 0 else "confirmatory",
                            "state": state,
                            "condition": condition,
                            "optimization_repeat": init,
                            "mapping_checkpoint": {
                                "no_shared": "identity",
                                "single_model": f"individual/{state}/seed{fold['ref']}_init{init}.pt",
                                "shared_fixed": os.path.relpath(fixed_path, exp.root),
                                "shared_joint": os.path.relpath(joint_path, exp.root),
                            }[condition],
                            "adapt_examples": n_examples,
                            "adapt_updates_run": ADAPT_UPDATES if n_examples else 0,
                            "adapt_best_update": best_update,
                            "adapt_dev_loss": dev_loss,
                            "adapt_train_seed": ADAPT_SEED,
                            "adapt_dev_seed": ADAPT_DEV_SEED,
                            "adapt_subset_seed": stable_seed("adaptation_controls_subset"),
                            "adapt_optimization_seed": adaptation_seed,
                            "alignment_cost": alignment_cost,
                            "orthogonality_max_error": orthogonal_error,
                            "individual_self_iia": individual_self_metrics["iia"],
                            "individual_minus_iia": individual_self_metrics["iia"] - metrics["iia"],
                            **metrics,
                        }
                        rows = [
                            existing for existing in rows
                            if not (
                                existing["fold"] == fold_index
                                and existing["state"] == state
                                and existing["optimization_repeat"] == init
                                and existing["condition"] == condition
                                and existing["adapt_examples"] == n_examples
                            )
                        ]
                        rows.append(row)
                        save_results(output_path, config, rows)
                        exp.log(
                            f"Control result: fold={fold_index} state={state} repeat={init} "
                            f"{condition} n={n_examples} test IIA={metrics['iia']:.4f}, "
                            f"gap={row['individual_minus_iia']:.4f}, dev_loss={dev_loss:.4f}"
                        )
                    for parameter in phi.parameters():
                        parameter.requires_grad_(True)
                del train_members, alignments
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    exp.log(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
