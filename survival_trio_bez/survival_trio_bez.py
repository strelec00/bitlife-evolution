"""Three-team survival sim with biomes, size penalties, and co-evolved brains.

This file is the runnable matplotlib UI. Core configuration lives in
`survival_common.py`, world rules live in `survival_world.py`, and genetic
training lives in `survival_training.py`.
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from survival_common import (
    BOARD,
    EMPTY_COLOR_ARR,
    FOOD_COLOR_ARR,
    FRAME_INTERVAL,
    GENERATIONS,
    LABELS,
    MOUNTAIN_COLOR_ARR,
    RIVER_COLOR_ARR,
    STEPS_PER_FRAME,
    TEAM_COLOR_ARR,
    TEAM_COLORS,
)
from survival_training import Trainer


def render_world(w):
    img = np.broadcast_to(EMPTY_COLOR_ARR, (BOARD, BOARD, 3)).copy()
    if w.mountains:
        rs, cs = zip(*w.mountains)
        img[list(rs), list(cs)] = MOUNTAIN_COLOR_ARR
    if w.rivers:
        rs, cs = zip(*w.rivers)
        img[list(rs), list(cs)] = RIVER_COLOR_ARR
    if w.food_cells:
        rs, cs = zip(*w.food_cells)
        img[list(rs), list(cs)] = FOOD_COLOR_ARR
    for ti, cr in enumerate(w.creatures):
        if not cr.alive or not cr.cells:
            continue
        rs, cs = zip(*cr.cells)
        img[list(rs), list(cs)] = TEAM_COLOR_ARR[ti]
    return img


def main():
    trainer = Trainer()

    fig, (ax_board, ax_fit) = plt.subplots(
        1, 2, figsize=(13, 7), gridspec_kw={"width_ratios": [1, 1]}
    )
    ax_board.set_xticks([])
    ax_board.set_yticks([])
    im = ax_board.imshow(render_world(trainer.world), interpolation="nearest")
    title = ax_board.set_title("gen 0 demo - step 0")

    ax_fit.set_xlim(0, GENERATIONS)
    ax_fit.set_xlabel("generation")
    ax_fit.set_ylabel("fitness")
    ax_fit.set_title("learning curves")
    ax_fit.grid(alpha=0.3)
    fit_lines_best = [
        ax_fit.plot([], [], color=TEAM_COLORS[t], linewidth=2, label=f"{LABELS[t]} best")[0]
        for t in range(3)
    ]
    fit_lines_mean = [
        ax_fit.plot(
            [], [], color=TEAM_COLORS[t], linewidth=1, alpha=0.4,
            linestyle="--", label=f"{LABELS[t]} mean"
        )[0]
        for t in range(3)
    ]
    ax_fit.legend(ncol=3, fontsize=8, loc="upper left")

    status = fig.text(0.5, 0.02, "", ha="center", fontsize=9)
    anim_ref = {"anim": None}

    def update_fit_lines():
        for t in range(3):
            x = list(range(len(trainer.history_best[t])))
            fit_lines_best[t].set_data(x, trainer.history_best[t])
            fit_lines_mean[t].set_data(x, trainer.history_mean[t])
        ax_fit.relim()
        ax_fit.autoscale_view()

    def update(_):
        w = trainer.world
        if not w.done:
            for _ in range(STEPS_PER_FRAME):
                if w.done:
                    break
                w.step()
        if w.done:
            if trainer.gen < GENERATIONS:
                trainer.evaluate_and_breed()
                update_fit_lines()
                trainer.start_demo()
            else:
                if not trainer.best_genomes_saved:
                    trainer.save_best_genomes()
                anim_ref["anim"].event_source.stop()
                title.set_text(f"training complete - final demo ended ({w.end_reason})")
                return [im, title, status]

        im.set_data(render_world(trainer.world))
        parts = []
        for i, cr in enumerate(trainer.world.creatures):
            if cr.alive:
                parts.append(f"{LABELS[i]}: size={len(cr.cells)} h={cr.hunger}")
            else:
                parts.append(f"{LABELS[i]}: DEAD")
        title.set_text(
            f"gen {trainer.demo_gen} demo - step {trainer.world.step_no}"
            f"  |  food={len(trainer.world.food_cells)}"
        )
        status.set_text("   ".join(parts))
        return [im, title, status]

    anim_ref["anim"] = animation.FuncAnimation(
        fig, update, interval=FRAME_INTERVAL, blit=False, repeat=False,
        cache_frame_data=False
    )
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.show()


if __name__ == "__main__":
    main()
