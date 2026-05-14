"""Visualize saved champion genomes for the simple survival simulation."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np


GENE_LABELS = [
    "food pull",
    "hunger food",
    "near aggression",
    "randomness",
    "obstacle penalty",
    "far aggression",
    "wall repulsion",
    "mountain pull",
    "river pull",
    "oversize aversion",
]

TEAM_COLORS = {
    "red": (1.0, 0.2, 0.2),
    "yellow": (1.0, 0.72, 0.05),
    "purple": (0.7, 0.3, 1.0),
}


def load_genomes(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    teams = data.get("teams", [])
    if not teams:
        raise ValueError(f"{path} does not contain any saved teams")

    names = [team["team"] for team in teams]
    scores = [float(team["best_score"]) for team in teams]
    genomes = np.array([team["genome"] for team in teams], dtype=float)

    if genomes.ndim != 2 or genomes.shape[1] != len(GENE_LABELS):
        raise ValueError(
            f"expected genomes with {len(GENE_LABELS)} genes, got shape {genomes.shape}"
        )

    return data, names, scores, genomes


def plot_genomes(data, names, scores, genomes):
    import matplotlib.pyplot as plt

    max_abs = max(2.0, float(np.max(np.abs(genomes))))

    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.25])

    ax_heat = fig.add_subplot(grid[0, :])
    heat = ax_heat.imshow(genomes, cmap="coolwarm", vmin=-max_abs, vmax=max_abs, aspect="auto")
    ax_heat.set_title("Champion genome heatmap")
    ax_heat.set_xticks(range(len(GENE_LABELS)), GENE_LABELS, rotation=35, ha="right")
    ax_heat.set_yticks(range(len(names)), names)
    ax_heat.set_xlabel("gene")

    for row in range(genomes.shape[0]):
        for col in range(genomes.shape[1]):
            value = genomes[row, col]
            text_color = "white" if abs(value) > max_abs * 0.55 else "black"
            ax_heat.text(col, row, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    cbar = fig.colorbar(heat, ax=ax_heat, shrink=0.9)
    cbar.set_label("gene value")

    for idx, (name, score, genome) in enumerate(zip(names, scores, genomes)):
        ax = fig.add_subplot(grid[1, idx])
        color = TEAM_COLORS.get(name, "tab:blue")
        ax.bar(range(len(genome)), genome, color=color)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylim(-max_abs, max_abs)
        ax.set_title(f"{name} champion | score {score:.1f}")
        ax.set_xticks(range(len(GENE_LABELS)), range(len(GENE_LABELS)))
        ax.set_xlabel("gene index")
        if idx == 0:
            ax.set_ylabel("value")
        ax.grid(axis="y", alpha=0.25)

    generation = data.get("generation", "?")
    saved_at = data.get("saved_at", "unknown time")
    fig.suptitle(f"Survival Trio Bez champion genomes | generation {generation} | saved {saved_at}")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "genomes_file",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("outputs") / "best_genomes.json",
        help="Path to best_genomes.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional image path to save the visualization, for example genomes.png",
    )
    args = parser.parse_args()

    if args.output:
        matplotlib.use("Agg")

    data, names, scores, genomes = load_genomes(args.genomes_file)
    fig = plot_genomes(data, names, scores, genomes)

    if args.output:
        fig.savefig(args.output, dpi=160)
        print(f"saved genome visualization to {args.output}")
    else:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
