#!/usr/bin/env python3
import argparse
import copy
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "code" / "source"
sys.path.insert(0, str(SOURCE))

from das.Classification_Model import make_model_DAS_fitted
from das.DAS import phi_class
from das.DAS_MLP import Distributed_Alignment_Search_MLP
from das.Dataset_Generation import (
    make_dataset_sample_model_AndOrAnd,
    make_intervention_dataset_AndOr,
    make_intervention_dataset_AndOrAnd,
    make_intervention_dataset_AndOr_DAS_Fitted,
    make_intervention_dataset_AndOrAnd_DAS_Fitted,
)
from das.Helper_Functions import set_seed
from das.RevNet import RevNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "pilot_config.json")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def make_runtime_config(config, smoke):
    runtime = copy.deepcopy(config)
    if smoke:
        runtime["model"]["training"].update(
            train_examples=128,
            eval_examples=64,
            test_examples=64,
            batch_size=64,
            max_epochs=1,
            early_stopping_patience=1,
        )
        runtime["shared_mapping"].update(
            train_examples=128,
            eval_examples=64,
            test_examples=64,
            batch_size=64,
            max_epochs=1,
            early_stopping_patience=1,
        )
        runtime["target_alignment"].update(
            adaptation_examples=128,
            eval_examples=64,
            test_examples=64,
            activity_examples=128,
            batch_size=64,
            max_epochs=1,
            early_stopping_patience=1,
        )
    return runtime


def atomic_torch_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary_path)
    temporary_path.replace(path)


def atomic_json_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(path)


def raw_dataset_to_tensors(raw_data, metadata):
    return {
        "base": torch.stack([item["base"] for item in raw_data]),
        "label": torch.stack([item["label"] for item in raw_data]),
        "sources": torch.stack([item["sources"] for item in raw_data]),
        "intervention": torch.tensor([item["intervention"] for item in raw_data], dtype=torch.bool),
        "metadata": metadata,
    }


def save_raw_dataset(raw_data, path, metadata):
    if path.exists():
        return
    data = raw_dataset_to_tensors(raw_data, metadata)
    atomic_torch_save(data, path)


def make_dataset(structure, size, fitted, seed):
    set_seed(seed)
    if fitted:
        generator = (
            make_intervention_dataset_AndOrAnd_DAS_Fitted
            if structure == "E"
            else make_intervention_dataset_AndOr_DAS_Fitted
        )
    else:
        generator = make_intervention_dataset_AndOrAnd if structure == "E" else make_intervention_dataset_AndOr
    return generator(size, 4)


def save_model_data(raw_data, model_dir, structure, role, seed, split):
    path = model_dir / f"{structure}_{role}_seed{seed}_{split}.pt"
    save_raw_dataset(
        raw_data,
        path,
        {"kind": "model_training", "structure": structure, "role": role, "seed": seed, "split": split},
    )


def train_base_models(config, device, run_root):
    model_config = config["model"]
    training = model_config["training"]
    source_seed = model_config["model_seeds"]["source"]
    target_seed = model_config["model_seeds"]["held_out_target"]
    models = {}
    model_data_dir = run_root / "data" / "base_models"
    checkpoint_dir = run_root / "checkpoints" / "base_models"
    base_results = {}

    for structure, model_name in config["structures"].items():
        for role, seed in (("source", source_seed), ("target", target_seed)):
            set_seed(seed)
            train_data = make_fitted_dataset(model_name, training["train_examples"])
            eval_data = make_fitted_dataset(model_name, training["eval_examples"])
            test_data = make_fitted_dataset(model_name, training["test_examples"])
            save_model_data(train_data, model_data_dir, structure, role, seed, "train")
            save_model_data(eval_data, model_data_dir, structure, role, seed, "eval")
            save_model_data(test_data, model_data_dir, structure, role, seed, "test")

            model, test_accuracy = make_model_DAS_fitted(
                DAS_Train=train_data,
                DAS_Test=test_data,
                DAS_Eval=eval_data,
                Hidden_Layer_Size=model_config["hidden_size"],
                inter_dim=[list(range(0, 12)), list(range(12, 24))],
                DEVICE=device,
                Max_Epochs=training["max_epochs"],
                Early_Stopping_Epochs=training["early_stopping_patience"],
                early_stopping_improve_threshold=training["early_stopping_improvement_threshold"],
                ReduceLROnPlateau_patience=training["reduce_lr_on_plateau_patience"],
            )
            checkpoint_path = checkpoint_dir / f"{structure}_{role}_seed{seed}.pt"
            atomic_torch_save(
                {
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "test_accuracy": test_accuracy,
                },
                checkpoint_path,
            )
            base_results[f"{structure}_{role}"] = {
                "seed": seed,
                "test_accuracy": test_accuracy,
                "checkpoint": str(checkpoint_path.relative_to(run_root)),
            }
            atomic_json_save(base_results, run_root / "base_model_results.json")
            models[(structure, role)] = model
            del train_data, eval_data, test_data
            print(
                f"base_model structure={structure} role={role} seed={seed} "
                f"test_accuracy={test_accuracy:.6f}",
                flush=True,
            )

    return models


def make_fitted_dataset(model_name, size):
    generator = (
        make_intervention_dataset_AndOrAnd_DAS_Fitted
        if model_name == "AndOrAnd"
        else make_intervention_dataset_AndOr_DAS_Fitted
    )
    return generator(size, 4)


def make_mapping_data(candidate, count, seed, split, run_root, scope):
    path = run_root / "data" / "mapping" / scope / f"{candidate}_{split}_seed{seed}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    raw_data = make_dataset(candidate, count, fitted=False, seed=seed)
    data = raw_dataset_to_tensors(
        raw_data,
        {
            "kind": scope,
            "candidate": candidate,
            "split": split,
            "seed": seed,
        },
    )
    atomic_torch_save(data, path)
    return data


class TensorDatasetDAS(Distributed_Alignment_Search_MLP):
    def Prepare_Dataset(self, raw_dataset):
        if not isinstance(raw_dataset, dict):
            return super().Prepare_Dataset(raw_dataset)
        variable_interventions = raw_dataset["intervention"].to(self.Device)
        expanded_interventions = torch.zeros(
            (variable_interventions.shape[0], self.Hidden_Layer_Size),
            dtype=torch.bool,
            device=self.Device,
        )
        for variable_index, dimensions in enumerate(self.Variable_Dimensions):
            expanded_interventions[:, dimensions] = variable_interventions[:, variable_index].unsqueeze(1)
        clean_dataset = {
            "base": raw_dataset["base"].to(self.Device),
            "label": raw_dataset["label"].to(self.Device),
            "sources": [raw_dataset["sources"][:, index].to(self.Device) for index in range(2)],
            "intervention": expanded_interventions,
        }
        return clean_dataset, clean_dataset["label"].shape[0]


def make_shared_mapping(config, source_model, source_structure, candidate, source_seed, device, run_root):
    mapping_config = config["shared_mapping"]
    seed_scheme = config["seed_scheme"]
    structure_index = list(config["structures"]).index(source_structure)
    candidate_index = list(config["structures"]).index(candidate)
    mapping_seed = (
        source_seed
        + seed_scheme["shared_mapping_init_offset"]
        + structure_index * 10
        + candidate_index
    )
    training_seed = (
        source_seed
        + seed_scheme["shared_mapping_training_offset"]
        + structure_index * 10
        + candidate_index
    )
    source_offset = seed_scheme["source_mapping_offset"]
    split_offsets = seed_scheme["split_offsets"]
    train_seed = source_seed + source_offset + split_offsets["train_or_adapt"]
    eval_seed = source_seed + source_offset + split_offsets["eval"]
    test_seed = source_seed + source_offset + split_offsets["test"]
    train_data = make_mapping_data(
        candidate,
        mapping_config["train_examples"],
        train_seed,
        "train",
        run_root,
        "source",
    )
    eval_data = make_mapping_data(
        candidate,
        mapping_config["eval_examples"],
        eval_seed,
        "eval",
        run_root,
        "source",
    )
    test_data = make_mapping_data(
        candidate,
        mapping_config["test_examples"],
        test_seed,
        "test",
        run_root,
        "source",
    )

    set_seed(mapping_seed)
    transform = RevNet(
        number_blocks=mapping_config["number_of_blocks"],
        in_features=config["model"]["hidden_size"],
        hidden_size=mapping_config["hidden_size"],
    ).to(device)
    optimizer = optim.Adam(transform.parameters(), lr=mapping_config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=mapping_config["reduce_lr_on_plateau_patience"],
    )
    transform_class = phi_class(
        transform,
        transform.inverse,
        nn.CrossEntropyLoss(),
        optimizer,
        scheduler,
    )
    experiment = TensorDatasetDAS(
        Model=source_model,
        Model_Layer=source_model.mlp.h[1],
        Train_Data_Raw=train_data,
        Test_Data_Raw=test_data,
        Eval_Data_Raw=eval_data,
        Hidden_Layer_Size=config["model"]["hidden_size"],
        Variable_Dimensions=[list(range(0, 12)), list(range(12, 24))],
        Transformation_Class=transform_class,
        Device=device,
    )
    del train_data, eval_data, test_data
    set_seed(training_seed)
    best_eval_loss = experiment.train_test(
        batch_size=mapping_config["batch_size"],
        epochs=mapping_config["max_epochs"],
        mode=1,
        early_stopping_threshold=mapping_config["early_stopping_patience"],
        early_stopping_improve_threshold=mapping_config["early_stopping_improvement_threshold"],
        verbose=True,
    )
    source_test_iia = experiment.train_test(batch_size=mapping_config["batch_size"], mode=2)
    best_transform = experiment.Transformation_Class.phi
    best_transform.eval()
    for parameter in best_transform.parameters():
        parameter.requires_grad = False
    checkpoint_path = run_root / "checkpoints" / "shared_mapping" / f"{source_structure}{candidate}.pt"
    atomic_torch_save(
        {
            "state_dict": best_transform.state_dict(),
            "source_structure": source_structure,
            "candidate": candidate,
            "source_seed": source_seed,
            "mapping_seed": mapping_seed,
            "training_seed": training_seed,
            "best_eval_loss": best_eval_loss,
            "source_test_iia": source_test_iia,
        },
        checkpoint_path,
    )
    experiment.Cleanup()
    print(
        f"shared_mapping source={source_structure} candidate={candidate} "
        f"eval_loss={best_eval_loss:.6f} source_test_iia={source_test_iia:.6f}",
        flush=True,
    )
    return best_transform, best_eval_loss, source_test_iia


def make_activity_inputs(count, seed, device):
    set_seed(seed)
    samples = [make_dataset_sample_model_AndOrAnd(4)[0] for _ in range(count)]
    return torch.as_tensor(np.asarray(samples), dtype=torch.float32, device=device)


def capture_hidden(model, inputs, batch_size):
    activations = []

    def capture_hook(module, module_inputs, output):
        activations.append(output.detach())

    hook = model.mlp.h[1].register_forward_hook(capture_hook)
    model.eval()
    with torch.no_grad():
        for start in range(0, inputs.size(0), batch_size):
            model(inputs[start : start + batch_size])
    hook.remove()
    return torch.cat(activations, dim=0)


def procrustes_rotation(target_activations, source_activations):
    target_centered = target_activations - target_activations.mean(dim=0, keepdim=True)
    source_centered = source_activations - source_activations.mean(dim=0, keepdim=True)
    cross_covariance = target_centered.T @ source_centered
    u, _, vh = torch.linalg.svd(cross_covariance, full_matrices=False)
    rotation = u @ vh
    if torch.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    return rotation


class OrthogonalAdapter(nn.Module):
    def __init__(self, initial_rotation):
        super().__init__()
        self.register_buffer("initial_rotation", initial_rotation.detach().clone())
        self.skew_parameter = nn.Parameter(torch.zeros_like(initial_rotation))

    def matrix(self):
        skew = self.skew_parameter - self.skew_parameter.T
        return torch.matrix_exp(skew) @ self.initial_rotation

    def forward(self, values):
        return values @ self.matrix()

    def inverse(self, values):
        return values @ self.matrix().T


class TargetTransform(nn.Module):
    def __init__(self, shared_mapping, alignment):
        super().__init__()
        self.shared_mapping = shared_mapping
        self.alignment = alignment
        for parameter in self.shared_mapping.parameters():
            parameter.requires_grad = False
        self.shared_mapping.eval()

    def forward(self, values):
        return self.shared_mapping(self.alignment(values))

    def inverse(self, values):
        return self.alignment.inverse(self.shared_mapping.inverse(values))


def validation_loss(experiment, batch_size):
    data = experiment.Eval_Dataset
    indices = list(range(experiment.Eval_Samples_Number))
    batches = experiment.chunk_list(indices, batch_size)
    weighted_loss_sum = 0.0
    total_correct = 0
    total_samples = 0
    for batch in batches:
        loss, total_correct, total_samples, _ = experiment.process_Batch(
            mode=3,
            data=data,
            ac_batch=batch,
            total_correct=total_correct,
            total_samples=total_samples,
        )
        weighted_loss_sum += loss.item() * len(batch)
    return weighted_loss_sum / max(total_samples, 1), total_correct / max(total_samples, 1)


def train_target_alignment(experiment, config):
    alignment_config = config["target_alignment"]
    for parameter in experiment.Model.parameters():
        parameter.requires_grad = False
    experiment.Model.eval()
    transform = experiment.Transformation_Class.phi
    transform.eval()
    for parameter in transform.shared_mapping.parameters():
        parameter.requires_grad = False
    for parameter in transform.alignment.parameters():
        parameter.requires_grad = True

    optimizer = experiment.Transformation_Class.optimizer
    scheduler = experiment.Transformation_Class.scheduler
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    steps_without_improvement = 0
    train_data = experiment.Train_Dataset
    indices = list(range(experiment.Train_Sample_Number))
    for epoch in range(alignment_config["max_epochs"]):
        random.shuffle(indices)
        batches = experiment.chunk_list(indices, alignment_config["batch_size"])
        total_loss = 0.0
        for batch in batches:
            optimizer.zero_grad()
            loss, _, _, _ = experiment.process_Batch(
                mode=1,
                data=train_data,
                ac_batch=batch,
                total_correct=0,
                total_samples=0,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        eval_loss, eval_accuracy = validation_loss(experiment, alignment_config["batch_size"])
        scheduler.step(eval_loss)
        if eval_loss < best_loss:
            significant_improvement = (
                eval_loss + alignment_config["early_stopping_improvement_threshold"] < best_loss
            )
            best_loss = eval_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(transform.alignment.state_dict())
            steps_without_improvement = 0 if significant_improvement else steps_without_improvement + 1
        else:
            steps_without_improvement += 1
        print(
            f"target_alignment epoch={epoch + 1} "
            f"train_loss={total_loss / max(len(batches), 1):.6f} "
            f"eval_loss={eval_loss:.6f} eval_iia={eval_accuracy:.6f}",
            flush=True,
        )
        if steps_without_improvement >= alignment_config["early_stopping_patience"]:
            break

    if best_state is None:
        raise RuntimeError("Target alignment did not produce a validation checkpoint")
    transform.alignment.load_state_dict(best_state)
    transform.eval()
    for parameter in transform.parameters():
        parameter.requires_grad = False
    return best_loss, best_epoch


def make_target_data(candidate, target_seed, config, run_root):
    alignment_config = config["target_alignment"]
    seed_scheme = config["seed_scheme"]
    target_offset = seed_scheme["target_adaptation_offset"]
    split_offsets = seed_scheme["split_offsets"]
    train_seed = target_seed + target_offset + split_offsets["train_or_adapt"]
    eval_seed = target_seed + target_offset + split_offsets["eval"]
    test_seed = target_seed + target_offset + split_offsets["test"]
    train_data = make_mapping_data(
        candidate,
        alignment_config["adaptation_examples"],
        train_seed,
        "adapt",
        run_root,
        "target",
    )
    eval_data = make_mapping_data(
        candidate,
        alignment_config["eval_examples"],
        eval_seed,
        "eval",
        run_root,
        "target",
    )
    test_data = make_mapping_data(
        candidate,
        alignment_config["test_examples"],
        test_seed,
        "test",
        run_root,
        "target",
    )
    return train_data, eval_data, test_data


def run_condition(config, condition, condition_index, models, mappings, device, run_root, activity_inputs):
    source_structure, candidate, target_structure = condition
    source_seed = config["model"]["model_seeds"]["source"]
    target_seed = config["model"]["model_seeds"]["held_out_target"]
    source_model = models[(source_structure, "source")]
    target_model = models[(target_structure, "target")]
    shared_mapping, source_eval_loss, source_test_iia = mappings[(source_structure, candidate)]

    target_activations = capture_hidden(
        target_model,
        activity_inputs,
        config["target_alignment"]["batch_size"],
    )
    source_activations = capture_hidden(
        source_model,
        activity_inputs,
        config["target_alignment"]["batch_size"],
    )
    initial_rotation = procrustes_rotation(target_activations, source_activations)
    alignment = OrthogonalAdapter(initial_rotation).to(device)
    target_transform = TargetTransform(shared_mapping, alignment).to(device)
    optimizer = optim.Adam(alignment.parameters(), lr=config["target_alignment"]["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=config["target_alignment"]["reduce_lr_on_plateau_patience"],
    )
    transform_class = phi_class(
        target_transform,
        target_transform.inverse,
        nn.CrossEntropyLoss(),
        optimizer,
        scheduler,
    )

    train_data, eval_data, test_data = make_target_data(
        candidate,
        target_seed,
        config,
        run_root,
    )
    alignment_seed = (
        target_seed
        + config["seed_scheme"]["target_alignment_training_offset"]
        + condition_index
    )
    set_seed(alignment_seed)
    experiment = TensorDatasetDAS(
        Model=target_model,
        Model_Layer=target_model.mlp.h[1],
        Train_Data_Raw=train_data,
        Test_Data_Raw=test_data,
        Eval_Data_Raw=eval_data,
        Hidden_Layer_Size=config["model"]["hidden_size"],
        Variable_Dimensions=[list(range(0, 12)), list(range(12, 24))],
        Transformation_Class=transform_class,
        Device=device,
    )
    del train_data, eval_data, test_data
    alignment_eval_loss, best_epoch = train_target_alignment(experiment, config)
    test_iia = experiment.train_test(
        batch_size=config["target_alignment"]["batch_size"],
        mode=2,
    )
    checkpoint_path = run_root / "checkpoints" / "target_alignment" / f"{condition}.pt"
    atomic_torch_save(
        {
            "state_dict": alignment.state_dict(),
            "initial_rotation": initial_rotation.detach().cpu(),
            "source_seed": source_seed,
            "target_seed": target_seed,
            "alignment_seed": alignment_seed,
            "condition": condition,
            "candidate": candidate,
            "alignment_eval_loss": alignment_eval_loss,
            "best_epoch": best_epoch,
            "test_iia": test_iia,
        },
        checkpoint_path,
    )
    experiment.Cleanup()
    result = {
        "source_structure": source_structure,
        "candidate": candidate,
        "target_structure": target_structure,
        "source_seed": source_seed,
        "target_seed": target_seed,
        "alignment_seed": alignment_seed,
        "source_mapping_eval_loss": source_eval_loss,
        "source_mapping_test_iia": source_test_iia,
        "target_alignment_eval_loss": alignment_eval_loss,
        "target_alignment_best_epoch": best_epoch,
        "target_test_iia": test_iia,
        "target_adaptation_examples": config["target_alignment"]["adaptation_examples"],
    }
    print(f"condition={condition} target_test_iia={test_iia:.6f}", flush=True)
    return result


def main():
    args = parse_args()
    config = json.loads(args.config.read_text())
    config = make_runtime_config(config, args.smoke)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this experiment must run on a GPU node")
    device = torch.device("cuda")
    torch.set_num_threads(max(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), 1))
    wandb.init(mode="disabled")

    run_kind = "smoke" if args.smoke else "pilot"
    run_id = f"{run_kind}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_root = ROOT / "results" / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": config["source_commit"],
        "smoke": args.smoke,
        "config": config,
    }
    atomic_json_save(manifest, run_root / "manifest.json")

    print(f"run_id={run_id}", flush=True)
    print(f"device={torch.cuda.get_device_name(device)}", flush=True)
    models = train_base_models(config, device, run_root)

    mappings = {}
    for source_structure in config["structures"]:
        for candidate in config["structures"]:
            source_seed = config["model"]["model_seeds"]["source"]
            mappings[(source_structure, candidate)] = make_shared_mapping(
                config,
                models[(source_structure, "source")],
                source_structure,
                candidate,
                source_seed,
                device,
                run_root,
            )

    activity_count = config["target_alignment"]["activity_examples"]
    activity_inputs = make_activity_inputs(
        activity_count,
        config["seed_scheme"]["activity_inputs_seed"],
        device,
    )
    atomic_torch_save(
        {
            "inputs": activity_inputs.detach().cpu(),
            "seed": config["seed_scheme"]["activity_inputs_seed"],
        },
        run_root / "data" / "alignment" / "activity_inputs.pt",
    )
    results = {}
    for condition_index, condition in enumerate(config["conditions"]):
        result = run_condition(
            config,
            condition,
            condition_index,
            models,
            mappings,
            device,
            run_root,
            activity_inputs,
        )
        results[condition] = result
        atomic_json_save(results, run_root / "results.json")

    print(f"completed run_id={run_id}", flush=True)


if __name__ == "__main__":
    main()
