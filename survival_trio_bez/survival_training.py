"""Genetic training loop for the simple survival simulation."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from survival_common import (
    ELITE,
    GENOME_SIZE,
    LABELS,
    MUT_RATE,
    MUT_SIGMA,
    POP,
    TOURNAMENT_K,
    rng,
)
from survival_world import World


def random_genome(rng_):
    g = rng_.uniform(-1.0, 1.0, GENOME_SIZE)
    g[0] = rng_.uniform(0.3, 1.5)   # food pull positive
    g[2] = rng_.uniform(0.0, 1.5)   # aggression near
    g[5] = rng_.uniform(0.0, 1.0)   # aggression second
    g[6] = rng_.uniform(0.0, 1.0)   # wall repulsion
    g[7] = rng_.uniform(-0.5, 1.0)  # mountain pull mostly positive
    g[8] = rng_.uniform(-0.5, 1.0)  # river pull mostly positive
    g[9] = rng_.uniform(0.0, 1.0)   # oversize aversion positive
    return g


def mutate(g, rng_):
    out = g.copy()
    mask = rng_.random(GENOME_SIZE) < MUT_RATE
    n = int(mask.sum())
    if n:
        out[mask] += rng_.normal(0.0, MUT_SIGMA, n)
    return np.clip(out, -2.0, 2.0)


def crossover(a, b, rng_):
    pt = int(rng_.integers(1, GENOME_SIZE))
    return np.concatenate([a[:pt], b[pt:]])


def tournament(pop, scores, rng_):
    idx = rng_.choice(len(pop), TOURNAMENT_K, replace=False)
    return pop[max(idx, key=lambda i: scores[i])].copy()


class Trainer:
    def __init__(self):
        self.pops = [[random_genome(rng) for _ in range(POP)] for _ in range(3)]
        self.champions = [self.pops[t][0].copy() for t in range(3)]
        self.champ_score = [-np.inf] * 3
        self.gen = 0
        self.history_best = [[], [], []]
        self.history_mean = [[], [], []]
        self.demo_gen = 0
        self.world = World([c.copy() for c in self.champions], rng)
        self.best_genomes_saved = False

    def start_demo(self):
        self.demo_gen = self.gen
        self.world = World([c.copy() for c in self.champions], rng)

    def save_best_genomes(self, path=None):
        if path is None:
            path = Path(__file__).with_name("outputs") / "best_genomes.json"
        else:
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "generation": self.gen,
            "teams": [
                {
                    "team": LABELS[t],
                    "best_score": self.champ_score[t],
                    "genome": self.champions[t].tolist(),
                }
                for t in range(3)
            ],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.best_genomes_saved = True
        print(f"saved best genomes to {path}")
        return path

    def evaluate_and_breed(self):
        idxs = [rng.permutation(POP) for _ in range(3)]
        scores = [np.zeros(POP) for _ in range(3)]
        for k in range(POP):
            triple = [self.pops[t][idxs[t][k]] for t in range(3)]
            w = World(triple, rng)
            while not w.done:
                w.step()
            fit = w.fitness()
            for t in range(3):
                scores[t][idxs[t][k]] = fit[t]

        for t in range(3):
            self.history_best[t].append(float(scores[t].max()))
            self.history_mean[t].append(float(scores[t].mean()))
            best_idx = int(np.argmax(scores[t]))
            if scores[t][best_idx] > self.champ_score[t]:
                self.champ_score[t] = float(scores[t][best_idx])
                self.champions[t] = self.pops[t][best_idx].copy()

        print("gen {:3d} | ".format(self.gen) + " | ".join(
            "{:6s} best={:7.1f} mean={:7.1f}".format(
                LABELS[t], self.history_best[t][-1], self.history_mean[t][-1])
            for t in range(3)))

        for t in range(3):
            elite_idx = sorted(range(POP), key=lambda i: scores[t][i], reverse=True)[:ELITE]
            new_pop = [self.pops[t][i].copy() for i in elite_idx]
            while len(new_pop) < POP:
                a = tournament(self.pops[t], scores[t], rng)
                b = tournament(self.pops[t], scores[t], rng)
                child = mutate(crossover(a, b, rng), rng)
                new_pop.append(child)
            self.pops[t] = new_pop

        self.gen += 1
