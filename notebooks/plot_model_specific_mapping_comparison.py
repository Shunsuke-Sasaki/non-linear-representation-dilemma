#!/usr/bin/env python
"""Compare model-specific mapping IIA for trained and random MLPs."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "Results" / "SharedMapping" / "full"
OUTPUT_DIR = ROOT / "figures" / "poster"
RESULT_PATHS = {
    "Task-trained MLPs": (
        "trained",
        RESULT_ROOT / "adaptation_controls_v2" / "results.json",
    ),
    "Randomly initialized MLPs": (
        "random",
        RESULT_ROOT / "adaptation_controls_random" / "results.json",
    ),
}
CONFIRMATORY_SEEDS = [3837, 9097, 2635, 5137]
REPEATS = [0, 1]
GROUP_COLORS = ["#0072B2", "#D55E00"]
SEED_COLORS = ["#5E3C99", "#1B9E77", "#D95F02", "#7570B3"]


def load_model_specific_iia(path, state):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {state} results: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload["config"]
    if config["confirmatory_folds"] != [1, 2, 3, 4] or config["inits"] != REPEATS:
        raise ValueError(f"Unexpected confirmatory folds/repeats in {path}")

    values = {}
    for row in payload["results"]:
        if row["state"] != state or row["heldout"] not in CONFIRMATORY_SEEDS:
            continue
        key = (row["heldout"], row["optimization_repeat"])
        value = row["individual_self_iia"]
        if key in values and not np.isclose(values[key], value):
            raise ValueError(f"Inconsistent model-specific IIA for {state}, {key}")
        values[key] = value

    expected = {
        (seed, repeat) for seed in CONFIRMATORY_SEEDS for repeat in REPEATS
    }
    if values.keys() != expected:
        raise ValueError(
            f"Expected {len(expected)} confirmatory model-specific IIA values "
            f"for {state}, found {len(values)}"
        )
    return values


def main():
    data = {
        group: load_model_specific_iia(path, state)
        for group, (state, path) in RESULT_PATHS.items()
    }
    group_names = list(RESULT_PATHS)
    x_positions = np.arange(len(group_names))
    seed_legend_handles = []

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.8, 5.4))

    for seed_index, seed in enumerate(CONFIRMATORY_SEEDS):
        seed_color = SEED_COLORS[seed_index]
        means = [
            np.mean([data[state][(seed, repeat)] for repeat in REPEATS])
            for state in group_names
        ]
        ax.plot(
            x_positions,
            means,
            color=seed_color,
            linewidth=1.3,
            alpha=0.75,
            zorder=1,
        )
        for group_index, (state, x_position) in enumerate(
            zip(group_names, x_positions)
        ):
            repeat_offsets = [-0.045, 0.045]
            repeat_values = [data[state][(seed, repeat)] for repeat in REPEATS]
            ax.scatter(
                x_position + np.asarray(repeat_offsets),
                repeat_values,
                color=GROUP_COLORS[group_index],
                alpha=0.38,
                s=32,
                edgecolor="white",
                linewidth=0.4,
                zorder=2,
            )
            ax.scatter(
                x_position,
                means[group_index],
                color=GROUP_COLORS[group_index],
                edgecolor=seed_color,
                linewidth=1.5,
                s=86,
                marker="o",
                zorder=3,
            )
        seed_legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=seed_color,
                linewidth=1.5,
                marker="o",
                markerfacecolor="white",
                markersize=6,
                label=f"Held-out seed {seed}",
            )
        )

    ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1, zorder=0)
    ax.set_xticks(x_positions, group_names)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("Interchange intervention accuracy (IIA)")
    ax.set_title("Model-specific nonlinear mapping")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    group_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="#555555",
            markersize=7,
            label=group,
        )
        for group, color in zip(group_names, GROUP_COLORS)
    ]
    group_legend = ax.legend(
        handles=group_handles,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        title="Model group",
    )
    ax.add_artist(group_legend)
    ax.legend(
        handles=seed_legend_handles,
        loc="lower left",
        frameon=True,
        framealpha=0.95,
        title="Paired held-out seed",
    )
    fig.text(
        0.5,
        0.015,
        "Large points: mean over two mapping-optimization repeats; small points: "
        "individual repeats. Lines pair the same held-out seed across groups. "
        "Dashed line marks chance level. Exploratory seed 4287 is excluded.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_path = OUTPUT_DIR / "model_specific_mapping_trained_vs_random_v2"
    png_path = base_path.with_suffix(".png")
    pdf_path = base_path.with_suffix(".pdf")
    if png_path.exists() or pdf_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing figure: {base_path} "
            "(choose a new output name if regeneration is needed)"
        )
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    caption = (
        "Held-out IIA using a nonlinear mapping trained specifically for each "
        "model, compared between task-trained and randomly initialized MLPs. "
        "Large points show the mean across two mapping-optimization repeats; "
        "small points show the individual repeats. Lines pair the same "
        "held-out seed across model groups. The dashed line indicates chance "
        "performance. The exploratory seed (4287) is excluded."
    )
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Caption: {caption}")


if __name__ == "__main__":
    main()
