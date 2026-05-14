"""Shared configuration and constants for the simple survival simulation."""

import numpy as np

SEED = None

BOARD = 100
INIT_BODY_SIZE = 4
MAX_HUNGER = 40
STARVE_PERIOD = 5
FOOD_TARGET = 50
SPAWN_TRIES = 3
STALEMATE_CAP = 500

# Biomes
N_MOUNTAIN_PATCHES = 5
N_RIVER_PATCHES = 4
BIOME_PATCH_RADIUS = (2, 4)

# Oversize behavior
BIG_SIZE_THRESHOLD = 25
SIZE_SLOWDOWN_STEP = 5
OVERSIZE_COMBAT_FACTOR = 0.5

# Genome / GA
GENOME_SIZE = 10
POP = 24
GENERATIONS = 20
TOURNAMENT_K = 4
ELITE = 2
MUT_SIGMA = 0.2
MUT_RATE = 0.5

# Fitness
W_KILL = 2.0
W_FOOD = 1.0
W_SIZE = 0.5
W_SURV = 1.0
WIN_BONUS = 200.0

WALL_REPULSION_RANGE = 4

# Render timing
FRAME_INTERVAL = 50
STEPS_PER_FRAME = 10

TEAM_COLORS = [
    (1.0, 0.2, 0.2),   # red
    (1.0, 0.85, 0.1),  # yellow
    (0.7, 0.3, 1.0),   # purple
]
FOOD_COLOR = (0.2, 0.85, 0.2)
MOUNTAIN_COLOR = (0.55, 0.55, 0.58)
RIVER_COLOR = (0.35, 0.7, 0.95)
EMPTY_COLOR = (0.97, 0.97, 0.95)
LABELS = ["red", "yellow", "purple"]

CARDINALS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
CORNERS = [(2, 2), (2, BOARD - 5), (BOARD - 5, BOARD // 2 - 1)]

EMPTY_COLOR_ARR = np.array(EMPTY_COLOR)
FOOD_COLOR_ARR = np.array(FOOD_COLOR)
MOUNTAIN_COLOR_ARR = np.array(MOUNTAIN_COLOR)
RIVER_COLOR_ARR = np.array(RIVER_COLOR)
TEAM_COLOR_ARR = np.array(TEAM_COLORS)

_seed_seq = np.random.SeedSequence(SEED)
print(f"[seed] {_seed_seq.entropy}")
rng = np.random.default_rng(_seed_seq)
