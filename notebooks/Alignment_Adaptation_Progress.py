#!/usr/bin/env python
"""Print progress for Alignment_Adaptation_Controls.py."""
import json
import os
import argparse
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="adaptation_controls_v2",
                        help="result directory under Results/SharedMapping/full")
    args = parser.parse_args()
    results_root = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "Results", "SharedMapping", "full", args.root
    ))
    results_path = os.path.join(results_root, "results.json")
    log_path = os.path.join(os.path.dirname(results_root), "log.txt")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"No adaptation-control results yet: {results_path}")
    with open(results_path) as result_file:
        payload = json.load(result_file)
    config, rows = payload["config"], payload["results"]
    expected = (
        len(config["folds"])
        * len(config["inits"])
        * len(config["conditions"])
        * len(config["adapt_sizes"])
    )
    groups = Counter((row["fold"], row["optimization_repeat"]) for row in rows)
    checkpoints = []
    for fold in config["folds"]:
        for state in config["states"]:
            for init in config["inits"]:
                relative = f"fold{fold}/{state}/shared_joint_init{init}.pt"
                legacy_relative = f"fold{fold}/{state}_shared_joint_init{init}.pt"
                if os.path.exists(os.path.join(results_root, relative)):
                    checkpoints.append(relative)
                elif os.path.exists(os.path.join(results_root, legacy_relative)):
                    checkpoints.append(legacy_relative)
    print(f"Results: {len(rows)}/{expected} ({100 * len(rows) / expected:.1f}%)")
    expected_checkpoints = len(config["folds"]) * len(config["states"]) * len(config["inits"])
    print(f"Joint-map checkpoints: {len(checkpoints)}/{expected_checkpoints}")
    print(
        "Fold/repeat groups: "
        + ", ".join(
            f"fold{fold}/rep{init}={count}/"
            f"{len(config['conditions']) * len(config['adapt_sizes'])}"
            for (fold, init), count in sorted(groups.items())
        )
    )
    if os.path.exists(log_path):
        with open(log_path) as log_file:
            lines = [line.rstrip() for line in log_file if line.strip()]
        current = [
            line for line in lines
            if "Fold " in line or "Control result:" in line or "joint epoch " in line
        ]
        print("Latest:", current[-1] if current else "no control-run log entries")


if __name__ == "__main__":
    main()
