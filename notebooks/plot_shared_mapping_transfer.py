#!/usr/bin/env python
"""Plot held-out transfer results for the shared-mapping main experiment."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "Results" / "SharedMapping" / "full"
OUTPUT_DIR = ROOT / "figures" / "shared_mapping_transfer"
ALIGNMENTS = [
    ("permutation", "Permutation alignment"),
    ("orthogonal", "Orthogonal alignment"),
]
STATES = [
    ("trained", "Task-trained MLPs"),
    ("random", "Randomly initialized MLPs"),
]
METHODS = [
    ("alignment_only", "Alignment only", "#555555"),
    ("shared_linear", "Shared linear map", "#0072B2"),
    ("shared_nonlinear", "Shared nonlinear map", "#D55E00"),
    ("single_model_transfer", "Single-model map transfer", "#009E73"),
]
TARGETS = [4287, 3837, 9097, 2635, 5137]
REFERENCE_BY_TARGET = {
    4287: 3837,
    3837: 4287,
    9097: 4287,
    2635: 4287,
    5137: 4287,
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


def load_results(align):
    path = RESULT_ROOT / align / "results.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing main-experiment results: {path}")

    rows = []
    with path.open(encoding="utf-8") as result_file:
        for line in result_file:
            rows.append(json.loads(line))

    expected = {}
    for state, _ in STATES:
        for method, _, _ in METHODS:
            role = "unseen" if method == "single_model_transfer" else "heldout"
            selected = [
                row
                for row in rows
                if row.get("state") == state
                and row.get("condition") == method
                and row.get("role") == role
                and (
                    method != "single_model_transfer"
                    or row.get("map_seed") == REFERENCE_BY_TARGET[row["target"]]
                )
            ]
            by_target = {}
            for row in selected:
                by_target.setdefault(row["target"], []).append(row["iia"])

            if set(by_target) != set(TARGETS):
                raise ValueError(
                    f"{align}/{state}/{method}: expected targets {TARGETS}, "
                    f"found {sorted(by_target)}"
                )
            if method == "alignment_only":
                if any(len(values) != 1 for values in by_target.values()):
                    raise ValueError(f"Unexpected repeated alignment-only results: {align}")
            elif any(len(values) != 2 for values in by_target.values()):
                raise ValueError(
                    f"Expected two mapping initializations per target: "
                    f"{align}/{state}/{method}"
                )

            expected[(state, method)] = {
                target: float(np.mean(by_target[target])) for target in TARGETS
            }
    return expected


def plot():
    data = {align: load_results(align) for align, _ in ALIGNMENTS}
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

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True, sharey=True)
    for row_index, (state, state_label) in enumerate(STATES):
        for column_index, (align, align_label) in enumerate(ALIGNMENTS):
            ax = axes[row_index, column_index]
            state_data = data[align]
            for method_index, (method, label, color) in enumerate(METHODS):
                target_values = state_data[(state, method)]
                values = np.asarray([target_values[target] for target in TARGETS])
                jitter = np.linspace(-0.12, 0.12, len(TARGETS))
                ax.scatter(
                    method_index + jitter,
                    values,
                    s=24,
                    color=color,
                    alpha=0.56,
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=2,
                )
                mean = values.mean()
                low = values.min()
                high = values.max()
                ax.errorbar(
                    method_index,
                    mean,
                    yerr=np.asarray([[mean - low], [high - mean]]),
                    fmt="D",
                    markersize=5,
                    color="#111111",
                    markerfacecolor="white",
                    markeredgewidth=1.3,
                    ecolor="#111111",
                    elinewidth=1.2,
                    capsize=3,
                    zorder=4,
                )

            ax.axhline(0.5, color="#777777", linestyle="--", linewidth=0.9, zorder=0)
            ax.set_title(f"{state_label} — {align_label}")
            ax.set_ylim(0.45, 0.72)
            ax.set_xlim(-0.5, len(METHODS) - 0.5)
            ax.set_xticks(
                np.arange(len(METHODS)),
                [label for _, label, _ in METHODS],
                rotation=18,
                ha="right",
            )
            ax.grid(axis="y", color="#dddddd", linewidth=0.65)
            ax.spines[["top", "right"]].set_visible(False)

        axes[row_index, 0].set_ylabel("Held-out IIA")

    method_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=color,
            markersize=6,
            label=label,
        )
        for _, label, color in METHODS
    ]
    mean_range_handle = plt.Line2D(
        [0],
        [0],
        marker="D",
        linestyle="-",
        color="#111111",
        markerfacecolor="white",
        markersize=5,
        label="Mean and min–max range",
    )
    fig.legend(
        handles=method_handles + [mean_range_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=5,
        frameon=False,
    )
    fig.suptitle(
        "Held-out transfer with alignment estimated from normal activity",
        y=1.03,
        fontsize=13,
    )
    fig.text(
        0.5,
        0.015,
        "Colored dots: held-out model seeds (4287, 3837, 9097, 2635, 5137); "
        "black diamonds and whiskers: mean and min–max across seeds. "
        "Whiskers are descriptive ranges, not confidence intervals. "
        "Dashed line: chance level.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.91))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUTPUT_DIR / "shared_mapping_heldout_transfer"
    png_path = next_available(base.with_suffix(".png"))
    pdf_path = next_available(base.with_suffix(".pdf"))
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    caption = (
        "Held-out interchange intervention accuracy (IIA) for models aligned "
        "using normal-activity calibration only, shown by mapping method, model "
        "state, and alignment family. Columns compare permutation and orthogonal "
        "alignment; rows compare task-trained and randomly initialized MLPs. "
        "Each colored point is one held-out model seed, averaged over two map "
        "initializations where applicable. Black diamonds show the mean and "
        "whiskers show the across-seed minimum-to-maximum range (not confidence "
        "intervals). The dashed line marks chance-level performance. Intervention "
        "labels from held-out models were not used to tune their alignments or "
        "mappings. Individual self-fit IIA is omitted because it is a reference "
        "fit on each evaluated model rather than a held-out transfer condition."
    )
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Caption: {caption}")


if __name__ == "__main__":
    plot()
