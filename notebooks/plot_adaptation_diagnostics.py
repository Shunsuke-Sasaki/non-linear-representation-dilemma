#!/usr/bin/env python
"""Plot paired trained-random differences and random source-to-held-out fit."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "Results" / "SharedMapping" / "full"
OUTPUT_DIR = ROOT / "figures" / "adaptation_controls"
RESULT_PATHS = {
    "trained": RESULT_ROOT / "adaptation_controls_v2" / "results.json",
    "random": RESULT_ROOT / "adaptation_controls_random" / "results.json",
}
CONDITIONS = {
    "no_shared": ("No shared mapping", "#555555"),
    "single_model": ("Single-model mapping", "#0072B2"),
    "shared_fixed": ("Shared mapping (fixed alignment)", "#D55E00"),
    "shared_joint": ("Shared mapping (joint alignment)", "#009E73"),
}
ADAPT_SIZES = [0, 256, 1024, 4096, 10000]
X = np.arange(len(ADAPT_SIZES))
X_LABELS = ["0", "256", "1,024", "4,096", "10,000"]
CONFIRMATORY_FOLDS = [1, 2, 3, 4]

# Source-side test IIA: means over the four training models, from the
# source-fit table in mapping evaluation/random_adaptation_controls_results.md.
SOURCE_Iia = {
    "shared_fixed": {
        1: [0.5015, 0.7415],
        2: [0.4978, 0.4997],
        3: [0.4993, 0.4985],
        4: [0.5019, 0.4938],
    },
    "shared_joint": {
        1: [0.5020, 0.7441],
        2: [0.5046, 0.5037],
        3: [0.4984, 0.4999],
        4: [0.4985, 0.4971],
    },
}
ADAPT_SIZE_COLORS = plt.get_cmap("viridis")(np.linspace(0.08, 0.9, len(ADAPT_SIZES)))
CASE_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
CASE_LABELS = {
    (fold, repeat): f"Seed {seed}, init {repeat}"
    for fold, seed in zip(CONFIRMATORY_FOLDS, [3837, 9097, 2635, 5137])
    for repeat in [0, 1]
}


def next_available(path):
    if not path.exists():
        return path
    version = 2
    while True:
        candidate = path.with_name(f"{path.stem}_v{version}{path.suffix}")
        if not candidate.exists():
            return candidate
        version += 1


def load_results(state):
    path = RESULT_PATHS[state]
    if not path.is_file():
        raise FileNotFoundError(f"Missing {state} adaptation results: {path}")
    with path.open(encoding="utf-8") as result_file:
        payload = json.load(result_file)

    folds = payload["config"]["confirmatory_folds"]
    repeats = payload["config"]["inits"]
    if folds != CONFIRMATORY_FOLDS or repeats != [0, 1]:
        raise ValueError(f"Unexpected {state} folds/repeats: {folds}/{repeats}")

    rows = [
        row
        for row in payload["results"]
        if row["fold"] in CONFIRMATORY_FOLDS and row["state"] == state
    ]
    expected = len(CONFIRMATORY_FOLDS) * len(repeats) * len(CONDITIONS) * len(ADAPT_SIZES)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} {state} rows, found {len(rows)}")

    indexed = {}
    for row in rows:
        key = (
            row["fold"],
            row["optimization_repeat"],
            row["condition"],
            row["adapt_examples"],
        )
        if key in indexed:
            raise ValueError(f"Duplicate {state} row: {key}")
        indexed[key] = row["iia"]
    return repeats, indexed


def save_figure(fig, base_name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = next_available(OUTPUT_DIR / f"{base_name}.png")
    pdf_path = next_available(OUTPUT_DIR / f"{base_name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def plot_paired_state_difference():
    trained_repeats, trained = load_results("trained")
    random_repeats, random = load_results("random")
    if trained_repeats != random_repeats:
        raise ValueError("Trained and random runs have different optimization repeats")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8), sharex=True, sharey=True)
    for ax, (condition, (label, color)) in zip(axes, CONDITIONS.items()):
        paired_differences = np.asarray(
            [
                [
                    np.mean(
                        [
                            trained[(fold, repeat, condition, size)]
                            - random[(fold, repeat, condition, size)]
                            for repeat in trained_repeats
                        ]
                    )
                    for size in ADAPT_SIZES
                ]
                for fold in CONFIRMATORY_FOLDS
            ]
        )
        for fold_values in paired_differences:
            ax.plot(
                X,
                fold_values,
                color="#888888",
                alpha=0.52,
                linewidth=1,
                marker=".",
                markersize=4,
                zorder=1,
            )
        ax.plot(
            X,
            paired_differences.mean(axis=0),
            color=color,
            linewidth=2.5,
            marker="o",
            markersize=5,
            label="Mean across folds",
            zorder=3,
        )
        ax.axhline(0, color="#555555", linestyle="--", linewidth=0.9, zorder=0)
        ax.set_title(label)
        ax.set_xticks(X, X_LABELS)
        ax.set_xlim(-0.1, len(X) - 0.9)
        ax.grid(axis="y", color="#dddddd", linewidth=0.65)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes:
        ax.set_xlabel("Adaptation examples")
    axes[0].set_ylabel("Paired difference in IIA\n(trained − random)")
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(
        "Paired trained-versus-random adaptation differences",
        y=0.99,
        fontsize=13,
    )
    fig.text(
        0.5,
        0.015,
        "Thin gray lines: fold-level paired differences (averaged over two repeats); "
        "thick colored lines: mean across four confirmatory folds. "
        "Positive values favor task-trained MLPs; no confidence intervals shown.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.9), w_pad=1.2)
    save_figure(fig, "trained_minus_random_paired_iia")


def plot_random_source_to_heldout():
    repeats, random = load_results("random")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharex=True, sharey=True)
    condition_order = ["shared_fixed", "shared_joint"]

    for ax, condition in zip(axes, condition_order):
        label, color = CONDITIONS[condition]
        source_values = SOURCE_Iia[condition]
        for case_index, (fold, repeat) in enumerate(
            (fold, repeat) for fold in CONFIRMATORY_FOLDS for repeat in repeats
        ):
                x_value = source_values[fold][repeat]
                y_values = np.asarray(
                    [
                        random[(fold, repeat, condition, size)]
                        for size in ADAPT_SIZES
                    ]
                )
                ax.plot(
                    np.full(len(X), x_value),
                    y_values,
                    color="#777777",
                    alpha=0.35,
                    linewidth=0.9,
                    zorder=1,
                )
                for size_index, y_value in enumerate(y_values):
                    ax.scatter(
                        x_value,
                        y_value,
                        color=ADAPT_SIZE_COLORS[size_index],
                        marker=CASE_MARKERS[case_index],
                        s=34,
                        alpha=0.78,
                        edgecolor="white",
                        linewidth=0.35,
                        zorder=2,
                    )

        ax.axvline(0.5, color="#777777", linestyle="--", linewidth=0.9, zorder=0)
        ax.axhline(0.5, color="#777777", linestyle="--", linewidth=0.9, zorder=0)
        ax.set_title(label)
        ax.set_xlabel("Source-side IIA (mean over four training models)")
        ax.set_xlim(0.48, 0.78)
        ax.set_ylim(0.48, 0.53)
        ax.grid(color="#e2e2e2", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Held-out IIA after alignment adaptation")
    size_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=ADAPT_SIZE_COLORS[index],
            markeredgecolor="white",
            markersize=7,
            label=f"{size:,} examples",
        )
        for index, size in enumerate(ADAPT_SIZES)
    ]
    size_legend = fig.legend(
        handles=size_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=5,
        frameon=False,
        fontsize=8,
    )
    case_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=CASE_MARKERS[index],
            linestyle="none",
            color="#444444",
            markersize=6,
            label=CASE_LABELS[(fold, repeat)],
        )
        for index, (fold, repeat) in enumerate(
            (fold, repeat) for fold in CONFIRMATORY_FOLDS for repeat in repeats
        )
    ]
    fig.legend(
        handles=case_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        ncol=4,
        frameon=False,
        fontsize=7.5,
        title="Case (marker shape)",
        title_fontsize=8,
    )
    fig.suptitle(
        "Random MLPs: source fit versus held-out adaptation",
        y=0.99,
        fontsize=13,
    )
    fig.text(
        0.5,
        0.015,
        "Each vertical trajectory is one fold × optimization-repeat pair; "
        "point color indicates adaptation size. "
        "Dashed lines mark chance level. Source IIA values are training-set "
        "test means reported in the experiment record.",
        ha="center",
        fontsize=8.2,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.79))
    save_figure(fig, "random_source_fit_vs_heldout_iia")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paired-only",
        action="store_true",
        help="Generate only the paired trained-versus-random difference figure.",
    )
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Generate only the random source-fit versus held-out figure.",
    )
    args = parser.parse_args()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    if not args.source_only:
        plot_paired_state_difference()
    if not args.paired_only:
        plot_random_source_to_heldout()
    print(
        "Source-side IIA values were taken from "
        "mapping evaluation/random_adaptation_controls_results.md."
    )


if __name__ == "__main__":
    main()
