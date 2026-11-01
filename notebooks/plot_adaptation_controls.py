#!/usr/bin/env python
"""Plot confirmatory results from the shared-mapping adaptation controls."""

import json
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = (
    ROOT
    / "Results"
    / "SharedMapping"
    / "full"
    / "adaptation_controls_v2"
    / "results.json"
)
OUTPUT_DIR = ROOT / "figures" / "adaptation_controls"

CONDITION_LABELS = {
    "no_shared": "No shared mapping",
    "single_model": "Single-model mapping",
    "shared_fixed": "Shared mapping (fixed alignment)",
    "shared_joint": "Shared mapping (joint alignment)",
}
COLORS = {
    "no_shared": "#555555",
    "single_model": "#0072B2",
    "shared_fixed": "#D55E00",
    "shared_joint": "#009E73",
}
FOLD_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
ADAPT_SIZES = [0, 256, 1024, 4096, 10000]
CAPTION = (
    "Figure X: Held-out interchange intervention accuracy (IIA) after adapting "
    "the alignment matrix on varying numbers of labeled intervention examples. Results show "
    "the four confirmatory leave-one-model-out folds (held-out seeds 3837, 9097, "
    "2635, and 5137); the exploratory fold (seed 4287) is excluded. Each fold "
    "is averaged over two optimization repeats. In panel A, thin colored lines "
    "show individual folds and thick condition-colored lines show the mean across "
    "folds. Panel B shows paired condition differences within each fold and "
    "optimization repeat, averaged over repeats; thin fold-colored lines show "
    "fold-level differences and thick contrast-colored lines show their mean. "
    "Fold colors are consistent across panels. Positive differences favor the first-named "
    "condition. No confidence intervals are shown. All mappings are frozen during "
    "held-out adaptation; only the held-out orthogonal alignment is updated."
)


def load_confirmatory_rows():
    if not RESULTS_PATH.is_file():
        raise FileNotFoundError(f"Adaptation-control results not found: {RESULTS_PATH}")

    with RESULTS_PATH.open(encoding="utf-8") as result_file:
        payload = json.load(result_file)

    config = payload["config"]
    folds = config["confirmatory_folds"]
    repeats = config["inits"]
    conditions = config["conditions"]
    rows = [
        row
        for row in payload["results"]
        if row["fold"] in folds and row["state"] == "trained"
    ]

    expected = len(folds) * len(repeats) * len(conditions) * len(ADAPT_SIZES)
    if len(rows) != expected:
        raise ValueError(
            f"Expected {expected} confirmatory trained measurements, found {len(rows)}"
        )

    indexed = {
        (
            row["fold"],
            row["optimization_repeat"],
            row["condition"],
            row["adapt_examples"],
        ): row["iia"]
        for row in rows
    }
    heldout_by_fold = {row["fold"]: row["heldout"] for row in rows}
    expected_keys = {
        (fold, repeat, condition, size)
        for fold in folds
        for repeat in repeats
        for condition in conditions
        for size in ADAPT_SIZES
    }
    if indexed.keys() != expected_keys:
        raise ValueError("Results have missing or duplicate fold/condition measurements")

    return folds, repeats, indexed, heldout_by_fold


def fold_means(folds, repeats, indexed, condition, size):
    return np.asarray(
        [
            np.mean(
                [
                    indexed[(fold, repeat, condition, size)]
                    for repeat in repeats
                ]
            )
            for fold in folds
        ]
    )


def available_path(path):
    if not path.exists():
        return path
    version = 2
    while True:
        candidate = path.with_name(f"{path.stem}_v{version}{path.suffix}")
        if not candidate.exists():
            return candidate
        version += 1


def save_figure_without_overwrite(fig, png_path, pdf_path):
    png_path = available_path(png_path)
    pdf_path = available_path(pdf_path)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path, pdf_path


def plot_mean_range_band():
    folds, repeats, indexed, _ = load_confirmatory_rows()
    x = np.arange(len(ADAPT_SIZES))
    x_tick_labels = ["0", "256", "1,024", "4,096", "10,000"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for condition, label in CONDITION_LABELS.items():
        values = np.stack(
            [
                fold_means(folds, repeats, indexed, condition, size)
                for size in ADAPT_SIZES
            ],
            axis=1,
        )
        mean = values.mean(axis=0)
        ax.fill_between(
            x,
            values.min(axis=0),
            values.max(axis=0),
            color=COLORS[condition],
            alpha=0.14,
            linewidth=0,
        )
        ax.plot(
            x,
            mean,
            marker="o",
            linewidth=2.4,
            markersize=6,
            color=COLORS[condition],
            label=label,
        )

    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1, zorder=0)
    ax.set(
        title="Mean held-out IIA across confirmatory folds",
        xlabel="Adaptation examples",
        ylabel="Mean interchange intervention accuracy (IIA)",
        ylim=(0.45, 1.01),
        xlim=(-0.1, len(x) - 0.9),
    )
    ax.set_xticks(x, x_tick_labels)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    caption = (
        "Mean held-out interchange intervention accuracy (IIA) across the four "
        "confirmatory leave-one-model-out folds, averaged over two optimization "
        "repeats per fold. The exploratory fold (held-out seed 4287) is excluded. "
        "Solid lines show the mean across folds; shaded bands show the fold-wise "
        "minimum-to-maximum range, not a confidence interval. The dashed line "
        "indicates chance-level performance. All mappings remain frozen during "
        "held-out adaptation; only the held-out orthogonal alignment is updated."
    )
    png_path, pdf_path = save_figure_without_overwrite(
        fig,
        OUTPUT_DIR / "adaptation_controls_condition_means_range_band.png",
        OUTPUT_DIR / "adaptation_controls_condition_means_range_band.pdf",
    )
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Caption: {caption}")


def plot():
    folds, repeats, indexed, heldout_by_fold = load_confirmatory_rows()
    fold_colors = {
        fold: FOLD_COLORS[index % len(FOLD_COLORS)]
        for index, fold in enumerate(folds)
    }
    x = np.arange(len(ADAPT_SIZES))
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

    fig = plt.figure(figsize=(11.5, 9.4))
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.05], hspace=0.38, wspace=0.28)
    axes_a = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    ax_delta = fig.add_subplot(grid[2, :])
    x_tick_labels = ["0", "256", "1,024", "4,096", "10,000"]

    for panel_index, (condition, label) in enumerate(CONDITION_LABELS.items(), start=1):
        ax = axes_a[panel_index - 1]
        values = np.stack(
            [
                fold_means(folds, repeats, indexed, condition, size)
                for size in ADAPT_SIZES
            ],
            axis=1,
        )
        for fold_index, fold_values in enumerate(values):
            ax.plot(
                x,
                fold_values,
                color=fold_colors[folds[fold_index]],
                alpha=0.7,
                linewidth=1.25,
                marker="o",
                markersize=3,
                zorder=1,
            )
        ax.plot(
            x,
            values.mean(axis=0),
            color=COLORS[condition],
            linewidth=2.6,
            marker="o",
            markersize=5,
            label="Mean across folds",
            zorder=3,
        )
        ax.axhline(0.5, color="#777777", linestyle="--", linewidth=0.9, zorder=0)
        ax.set(
            title=f"A{panel_index}. {label}",
            xlabel="Adaptation examples",
            ylabel="IIA",
            ylim=(0.45, 1.01),
            xlim=(-0.1, len(x) - 0.9),
        )
        ax.set_xticks(x, x_tick_labels)
        ax.grid(axis="y", color="#dddddd", linewidth=0.65)
        ax.spines[["top", "right"]].set_visible(False)

    contrasts = [
        ("single_model", "no_shared", "Single-model − no shared", "#0072B2"),
        ("shared_fixed", "single_model", "Shared (fixed) − single-model", "#D55E00"),
        ("shared_joint", "single_model", "Shared (joint) − single-model", "#009E73"),
    ]
    contrast_styles = ["-", "--", ":"]
    for contrast_index, (first, second, label, color) in enumerate(contrasts):
        paired = np.stack(
            [
                [
                    np.mean(
                        [
                            indexed[(fold, repeat, first, size)]
                            - indexed[(fold, repeat, second, size)]
                            for repeat in repeats
                        ]
                    )
                    for size in ADAPT_SIZES
                ]
                for fold in folds
            ]
        )
        delta_mean = paired.mean(axis=0)
        for fold_index, fold_values in enumerate(paired):
            ax_delta.plot(
                x,
                fold_values,
                color=fold_colors[folds[fold_index]],
                alpha=0.58,
                linewidth=1.0,
                linestyle=contrast_styles[contrast_index],
                marker=".",
                markersize=4,
                zorder=1,
            )
        ax_delta.plot(
            x,
            delta_mean,
            marker="o",
            linewidth=2.4,
            markersize=5,
            color=color,
            label=label,
            zorder=3,
        )

    ax_delta.axhline(0, color="#555555", linestyle="--", linewidth=1, zorder=0)
    ax_delta.set(
        title="B. Paired differences between conditions",
        xlabel="Adaptation examples",
        ylabel="Difference in IIA",
        xlim=(-0.1, len(x) - 0.9),
    )
    ax_delta.set_xticks(x, x_tick_labels)
    ax_delta.yaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(xmax=1, decimals=0)
    )
    ax_delta.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax_delta.spines[["top", "right"]].set_visible(False)
    ax_delta.legend(frameon=False, loc="best")

    fold_handles = [
        plt.Line2D(
            [0],
            [0],
            color=fold_colors[fold],
            marker="o",
            linewidth=1.4,
            markersize=4,
            label=f"Held-out seed {heldout_by_fold[fold]}",
        )
        for fold in folds
    ]
    mean_handle = plt.Line2D(
        [0],
        [0],
        color="#222222",
        marker="o",
        linewidth=2.6,
        markersize=5,
        label="Fold mean",
    )
    fig.legend(
        handles=fold_handles + [mean_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=5,
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(
        left=0.08, right=0.98, bottom=0.07, top=0.88, hspace=0.42, wspace=0.3
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path, pdf_path = save_figure_without_overwrite(
        fig,
        OUTPUT_DIR / "adaptation_controls_confirmatory.png",
        OUTPUT_DIR / "adaptation_controls_confirmatory.pdf",
    )
    plt.close(fig)

    fig_means, ax_means = plt.subplots(figsize=(7.5, 4.8))
    for condition, label in CONDITION_LABELS.items():
        values = np.stack(
            [
                fold_means(folds, repeats, indexed, condition, size)
                for size in ADAPT_SIZES
            ],
            axis=1,
        )
        for fold_values in values:
            ax_means.plot(
                x,
                fold_values,
                color=COLORS[condition],
                alpha=0.24,
                linewidth=1.1,
                marker=".",
                markersize=4,
                zorder=1,
            )
        ax_means.plot(
            x,
            values.mean(axis=0),
            marker="o",
            linewidth=2.4,
            markersize=6,
            color=COLORS[condition],
            label=label,
            zorder=3,
        )

    ax_means.axhline(0.5, color="#777777", linestyle="--", linewidth=1, zorder=0)
    ax_means.set(
        title="Mean held-out IIA across confirmatory folds",
        xlabel="Adaptation examples",
        ylabel="Mean interchange intervention accuracy (IIA)",
        ylim=(0.45, 1.01),
        xlim=(-0.1, len(x) - 0.9),
    )
    ax_means.set_xticks(x, x_tick_labels)
    ax_means.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax_means.spines[["top", "right"]].set_visible(False)
    ax_means.legend(frameon=False, loc="upper left")
    fig_means.tight_layout()

    means_png_path = OUTPUT_DIR / "adaptation_controls_condition_means.png"
    means_pdf_path = OUTPUT_DIR / "adaptation_controls_condition_means.pdf"
    means_png_path, means_pdf_path = save_figure_without_overwrite(
        fig_means, means_png_path, means_pdf_path
    )
    plt.close(fig_means)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {means_png_path}")
    print(f"Saved: {means_pdf_path}")
    print(
        "Mean-figure caption: Mean held-out interchange intervention accuracy (IIA) "
        "across the four confirmatory leave-one-model-out folds, averaged over two "
        "optimization repeats per fold. The exploratory fold (held-out seed 4287) "
        "is excluded. Thick lines show the mean across folds; thin lines show "
        "individual fold trajectories. The dashed line indicates chance-level "
        "performance. All mappings remain frozen during held-out adaptation; only "
        "the held-out orthogonal alignment is updated."
    )
    print(f"Caption: {CAPTION}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--range-band-only",
        action="store_true",
        help="Generate only the separate mean figure with fold-wise range bands.",
    )
    args = parser.parse_args()
    if args.range_band_only:
        plot_mean_range_band()
    else:
        plot()
