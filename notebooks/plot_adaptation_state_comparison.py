#!/usr/bin/env python
"""Compare trained and random adaptation-control curves."""

import json
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "Results" / "SharedMapping" / "full"
OUTPUT_DIR = ROOT / "figures" / "adaptation_controls"
ADAPT_SIZES = [0, 256, 1024, 4096, 10000]
STATES = [
    ("trained", "Task-trained MLPs"),
    ("random", "Randomly initialized MLPs"),
]
CONDITIONS = {
    "no_shared": ("No shared mapping", "#555555"),
    "single_model": ("Single-model mapping", "#0072B2"),
    "shared_fixed": ("Shared mapping (fixed alignment)", "#D55E00"),
    "shared_joint": ("Shared mapping (joint alignment)", "#009E73"),
}
X = np.arange(len(ADAPT_SIZES))
X_LABELS = ["0", "256", "1,024", "4,096", "10,000"]


def available_path(path):
    if not path.exists():
        return path
    version = 2
    while True:
        candidate = path.with_name(f"{path.stem}_v{version}{path.suffix}")
        if not candidate.exists():
            return candidate
        version += 1


def load_state(state):
    result_dir = RESULT_ROOT / (
        "adaptation_controls_v2" if state == "trained" else "adaptation_controls_random"
    )
    result_path = result_dir / "results.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"Missing {state} adaptation results: {result_path}")

    with result_path.open(encoding="utf-8") as result_file:
        payload = json.load(result_file)
    config = payload["config"]
    folds = config["confirmatory_folds"]
    repeats = config["inits"]
    expected = len(folds) * len(repeats) * len(CONDITIONS) * len(ADAPT_SIZES)
    rows = [
        row
        for row in payload["results"]
        if row["state"] == state and row["fold"] in folds
    ]
    if len(rows) != expected:
        raise ValueError(
            f"Expected {expected} confirmatory {state} measurements, found {len(rows)}"
        )

    indexed = {}
    for row in rows:
        key = (
            row["fold"],
            row["optimization_repeat"],
            row["condition"],
            row["adapt_examples"],
        )
        if key in indexed:
            raise ValueError(f"Duplicate {state} result for {key}")
        indexed[key] = row["iia"]

    expected_keys = {
        (fold, repeat, condition, size)
        for fold in folds
        for repeat in repeats
        for condition in CONDITIONS
        for size in ADAPT_SIZES
    }
    if indexed.keys() != expected_keys:
        raise ValueError(f"Incomplete {state} results")

    heldout_by_fold = {row["fold"]: row["heldout"] for row in rows}
    return folds, repeats, indexed, heldout_by_fold


def fold_curve(fold, repeats, indexed, condition):
    return np.asarray(
        [
            np.mean(
                [
                    indexed[(fold, repeat, condition, size)]
                    for repeat in repeats
                ]
            )
            for size in ADAPT_SIZES
        ]
    )


def plot(replace_current=False):
    loaded = {state: load_state(state) for state, _ in STATES}
    reference_folds, reference_repeats, _, reference_heldout = loaded["trained"]
    for state, _ in STATES[1:]:
        folds, repeats, _, heldout_by_fold = loaded[state]
        if (
            folds != reference_folds
            or repeats != reference_repeats
            or heldout_by_fold != reference_heldout
        ):
            raise ValueError("Trained and random results use different folds or repeats")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=False)

    for ax, (state, title) in zip(axes, STATES):
        folds, repeats, indexed, _ = loaded[state]
        for condition, (label, color) in CONDITIONS.items():
            fold_curves = np.stack(
                [
                    fold_curve(fold, repeats, indexed, condition)
                    for fold in folds
                ]
            )
            for values in fold_curves:
                ax.plot(
                    X,
                    values,
                    color=color,
                    alpha=0.24,
                    linewidth=1.0,
                    marker=".",
                    markersize=3,
                    zorder=1,
                )
            ax.plot(
                X,
                fold_curves.mean(axis=0),
                color=color,
                linewidth=2.4,
                marker="o",
                markersize=5,
                label=label,
                zorder=3,
            )

        ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1, zorder=0)
        y_limits = (0.45, 1.01) if state == "trained" else (0.48, 0.515)
        panel_title = title if state == "trained" else f"{title} (zoomed y-axis)"
        ax.set(
            title=panel_title,
            xlabel="Adaptation examples",
            xlim=(-0.1, len(X) - 0.9),
            ylim=y_limits,
        )
        ax.set_ylabel("Interchange intervention accuracy (IIA)")
        ax.set_xticks(X, X_LABELS)
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    condition_handles = [
        plt.Line2D(
            [0],
            [0],
            color=color,
            linewidth=2.4,
            marker="o",
            markersize=5,
            label=label,
        )
        for label, color in CONDITIONS.values()
    ]
    fig.legend(
        handles=condition_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        "Held-out adaptation curves: trained vs. random models",
        y=0.99,
        fontsize=14,
    )
    fig.text(
        0.5,
        0.01,
        "Thin lines: individual confirmatory folds, colored by condition; "
        "thick lines: condition means. "
        "Each fold averages two optimization repeats. "
        "The random-model panel uses a zoomed y-axis.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.86))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUTPUT_DIR / "adaptation_controls_trained_vs_random"
    if replace_current:
        base = OUTPUT_DIR / "adaptation_controls_trained_vs_random_v2"
        png_path = base.with_suffix(".png")
        pdf_path = base.with_suffix(".pdf")
    else:
        png_path = available_path(base.with_suffix(".png"))
        pdf_path = available_path(base.with_suffix(".pdf"))
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    caption = (
        "Held-out interchange intervention accuracy (IIA) as a function of the "
        "number of labeled intervention examples used to adapt the held-out "
        "alignment, shown separately for task-trained and randomly initialized "
        "MLPs. Each panel shows the four confirmatory leave-one-model-out folds "
        "(held-out seeds 3837, 9097, 2635, and 5137); the exploratory fold "
        "(seed 4287) is excluded. Thin condition-colored lines show individual "
        "fold means (each averaged over two optimization repeats); thick lines "
        "show condition means. "
        "The dashed line indicates chance-level "
        "performance. The random-model panel uses a zoomed y-axis for readability. "
        "All mappings are frozen during held-out adaptation; only "
        "the held-out orthogonal alignment is updated. Fold-level variability "
        "is displayed descriptively; no confidence intervals are shown."
    )
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Caption: {caption}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replace-current",
        action="store_true",
        help="Replace the current v2 trained-vs-random PNG and PDF.",
    )
    plot(replace_current=parser.parse_args().replace_current)
