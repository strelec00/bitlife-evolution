"""Three-species ecosystem sim with herbivores, carnivores, omnivores.

Each species owns multiple INDIVIDUAL organisms sharing one brain (genome).
Organisms gain ATOMS by eating; spend atoms on body cells and parts:
  - Eye   (15 atoms): removes sight cap. Default sight = SIGHT_RADIUS cells.
  - Gun   (30 atoms): each turn, shoots 1 cell off nearest visible enemy.
  - Spike (12 atoms): attackers eating your cells lose 1 cell back.

Diet rules:
  - Herbivore (RED, 6 orgs * 3 cells): eats only food (plants).
  - Carnivore (YELLOW, 2 orgs * 5 cells): eats only enemy organism cells (melee).
  - Omnivore  (PURPLE, 3 orgs * 4 cells): eats both.

Group defense:
  - For each cell, count same-species adjacent ally cells (8-conn).
  - Cell dies in melee combat only if enemy attackers > ally count.
  - Big herbivore packs become very tanky once they fatten up.

Map: 100x100, persistent biomes:
  - Mountain (gray): each body cell on mountain counts +1 in combat-size
    comparison (offense + defense for melee).
  - River (cyan): if any body cell touches river, hunger does NOT tick.

Each tick per organism, brain picks one action: BUILD part / GROW body /
SHOOT (if has gun + visible enemy) / MOVE (default). Eat & combat resolve
automatically after movement.

Match ends when <=1 species alive OR STALEMATE_CAP reached.

Fitness per species:
    W_SURV * sum(org_steps_alive) + W_KILL * kills_dealt
    + W_FOOD * food_eaten + W_ATOMS * atoms_collected
    + W_SIZE * total_cells_at_end + (WIN_BONUS if last species standing)

Survival weight dominates.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# ----- config -----
SEED = None

BOARD = 70
FOOD_TARGET = 100
SPAWN_TRIES = 8
STALEMATE_CAP = 600
SIGHT_RADIUS = 22

# species & diets
N_SPECIES = 3
LABELS = ["red", "yellow", "purple"]
DIETS = ["herb", "carn", "omni"]
INIT_ORGS = [6, 2, 3]
INIT_CELLS = [3, 5, 4]
SPAWN_CENTERS = [(8, 8), (8, BOARD - 8), (BOARD - 8, BOARD // 2)]

# atoms / parts
INIT_ATOMS = 10
ATOM_CAP = 200
EYE_COST = 15
GUN_COST = 30
SPIKE_COST = 12
BODY_COST = 5
SHOOT_RANGE = 18  # max manhattan distance for gun fire

# hunger / oversize / biomes
MAX_HUNGER = 150
STARVE_PERIOD = 7
BIG_SIZE_THRESHOLD = 20
SIZE_SLOWDOWN_STEP = 5
OVERSIZE_COMBAT_FACTOR = 0.5
N_MOUNTAIN_PATCHES = 8
N_RIVER_PATCHES = 0
BIOME_PATCH_RADIUS = (2, 5)

# brain
GENOME_SIZE = 14
WALL_REPULSION_RANGE = 5

# GA
POP = 16
GENERATIONS = 50
TOURNAMENT_K = 4
ELITE = 2
MUT_SIGMA = 0.3
MUT_RATE = 0.5

# fitness weights
W_SURV = 1.0
W_KILL = 2.0
W_FOOD = 1.0
W_ATOMS = 0.5
W_SIZE = 0.3
WIN_BONUS = 300.0

# render
FRAME_INTERVAL = 30
STEPS_PER_FRAME = 5

SPECIES_COLORS = [
    (1.0, 0.2, 0.2),   # red
    (1.0, 0.85, 0.1),  # yellow
    (0.7, 0.3, 1.0),   # purple
]
FOOD_COLOR = (0.2, 0.85, 0.2)
MOUNTAIN_COLOR = (0.55, 0.55, 0.58)
RIVER_COLOR = (0.35, 0.7, 0.95)
EMPTY_COLOR = (0.97, 0.97, 0.95)

CARDINALS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
EIGHT = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)]

EMPTY_COLOR_ARR = np.array(EMPTY_COLOR)
FOOD_COLOR_ARR = np.array(FOOD_COLOR)
MOUNTAIN_COLOR_ARR = np.array(MOUNTAIN_COLOR)
RIVER_COLOR_ARR = np.array(RIVER_COLOR)
SPECIES_COLOR_ARR = np.array(SPECIES_COLORS)

_seed_seq = np.random.SeedSequence(SEED)
print(f"[seed] {_seed_seq.entropy}")
rng = np.random.default_rng(_seed_seq)


# ----- organism -----

class Organism:
    __slots__ = (
        "species_id", "cells", "atoms", "hunger", "alive",
        "has_eye", "has_gun", "has_spike", "move_skip_counter",
        "last_move_dir",
    )

    def __init__(self, species_id, cells):
        self.species_id = species_id
        self.cells = set(cells)
        # carns get extra starting atoms so they can buy a gun fast
        self.atoms = INIT_ATOMS + (25 if DIETS[species_id] == "carn" else 0)
        self.hunger = 0
        self.alive = True
        # carnivores hatch with an eye built-in (predator sight)
        self.has_eye = (DIETS[species_id] == "carn")
        self.has_gun = False
        self.has_spike = False
        self.move_skip_counter = 0
        self.last_move_dir = (0, 0)

    def centroid(self):
        n = len(self.cells)
        if n == 0:
            return 0.0, 0.0
        rs = 0
        cs = 0
        for r, c in self.cells:
            rs += r
            cs += c
        return rs / n, cs / n


# ----- helpers -----

def _free_cell(rng_, occupied):
    for _ in range(80):
        p = (int(rng_.integers(0, BOARD)), int(rng_.integers(0, BOARD)))
        if p not in occupied:
            return p
    return None


def _spawn_biomes(rng_):
    forbidden = set()
    for (r0, c0) in SPAWN_CENTERS:
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                forbidden.add((r0 + dr, c0 + dc))

    def _patch(center, radius, exclude):
        out = set()
        cr_, cc_ = center
        r2 = radius * radius
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr * dr + dc * dc <= r2:
                    nr, nc = cr_ + dr, cc_ + dc
                    if 0 <= nr < BOARD and 0 <= nc < BOARD and (nr, nc) not in exclude:
                        out.add((nr, nc))
        return out

    mountains = set()
    for _ in range(N_MOUNTAIN_PATCHES):
        cr_r = int(rng_.integers(7, BOARD - 7))
        cr_c = int(rng_.integers(7, BOARD - 7))
        rad = int(rng_.integers(BIOME_PATCH_RADIUS[0], BIOME_PATCH_RADIUS[1] + 1))
        mountains |= _patch((cr_r, cr_c), rad, forbidden)
    rivers = set()
    for _ in range(N_RIVER_PATCHES):
        cr_r = int(rng_.integers(7, BOARD - 7))
        cr_c = int(rng_.integers(7, BOARD - 7))
        rad = int(rng_.integers(BIOME_PATCH_RADIUS[0], BIOME_PATCH_RADIUS[1] + 1))
        rivers |= _patch((cr_r, cr_c), rad, forbidden | mountains)
    return mountains, rivers


def _filter_radius(targets, centroid, radius):
    cr_r, cr_c = centroid
    return {t for t in targets if abs(t[0] - cr_r) + abs(t[1] - cr_c) <= radius}


def _nearest_to(centroid, targets):
    if not targets:
        return None, 10**9
    cr_r, cr_c = centroid
    best, best_d = None, 10**9
    for tc in targets:
        d = abs(tc[0] - cr_r) + abs(tc[1] - cr_c)
        if d < best_d:
            best_d, best = d, tc
    return best, best_d


def _decide_move(genome, org, world, snapshot_others_cells):
    """Return target new_cells set (movement plan). Caller resolves conflicts."""
    g = genome
    centroid = org.centroid()
    centroid_r, centroid_c = centroid
    size_self = len(org.cells)

    # diet-aware behavior gates: brain only weighs signals it can actually use
    diet = DIETS[org.species_id]
    eats_food = diet in ("herb", "omni")
    eats_meat = diet in ("carn", "omni")

    # perception (uses per-step species cache; sight cap unless has eye)
    if org.has_eye:
        visible_food = world.food_cells
    else:
        visible_food = _filter_radius(world.food_cells, centroid, SIGHT_RADIUS)

    enemy_cells_all = set()
    for sid in range(N_SPECIES):
        if sid != org.species_id:
            enemy_cells_all |= world._species_cells[sid]
    visible_enemy = enemy_cells_all if org.has_eye else _filter_radius(enemy_cells_all, centroid, SIGHT_RADIUS)
    enemy_t, _ = _nearest_to(centroid, visible_enemy)
    nearest_enemy_org_size = 1
    nearest_enemy_diet = None
    if enemy_t is not None:
        for o in world.alive_orgs:
            if o.species_id != org.species_id and enemy_t in o.cells:
                nearest_enemy_org_size = len(o.cells)
                nearest_enemy_diet = DIETS[o.species_id]
                break

    ally_cells = world._species_cells[org.species_id] - org.cells
    visible_ally = ally_cells if org.has_eye else _filter_radius(ally_cells, centroid, SIGHT_RADIUS)
    ally_t, _ = _nearest_to(centroid, visible_ally)

    # biomes always perceived (terrain)
    food_t, _ = _nearest_to(centroid, visible_food)
    mtn_t, _ = _nearest_to(centroid, world.mountains)
    riv_t, _ = _nearest_to(centroid, world.rivers)

    pref_r = pref_c = 0.0

    # food pull — only if diet eats food
    if food_t is not None and eats_food:
        hunger_factor = 1.0 + g[5] * (org.hunger / MAX_HUNGER)
        oversize_factor = 1.0
        if size_self > BIG_SIZE_THRESHOLD and g[11] > 0:
            excess_norm = (size_self - BIG_SIZE_THRESHOLD) / BIG_SIZE_THRESHOLD
            oversize_factor = max(0.0, 1.0 - g[11] * excess_norm)
        scale = g[0] * hunger_factor * oversize_factor
        pref_r += scale * np.sign(food_t[0] - centroid_r)
        pref_c += scale * np.sign(food_t[1] - centroid_c)

    # prey chase — only if diet eats meat
    # flee — only if WE cannot eat the threat (prey species fleeing predator)
    if enemy_t is not None:
        ratio = size_self / max(1, nearest_enemy_org_size)
        if eats_meat:
            # chase: scaled by aggression weight; positive when bigger
            chase = g[1] * (ratio - 1.0)
            # baseline pull toward prey when we are predator (so we hunt even if same-size)
            chase += 0.6 * g[1]
            pref_r += chase * np.sign(enemy_t[0] - centroid_r)
            pref_c += chase * np.sign(enemy_t[1] - centroid_c)
        # flee from meat-eaters when we cannot eat them back, or when we are smaller
        threat = (nearest_enemy_diet in ("carn", "omni"))
        if threat and (not eats_meat or ratio < 1.2):
            flee_strength = max(g[3], 0.5)  # baseline flee even if gene says low
            flee_strength *= max(0.5, 1.6 - ratio)  # flee harder when much smaller
            pref_r -= flee_strength * np.sign(enemy_t[0] - centroid_r)
            pref_c -= flee_strength * np.sign(enemy_t[1] - centroid_c)

    # ally cohesion
    if ally_t is not None:
        pref_r += g[2] * np.sign(ally_t[0] - centroid_r)
        pref_c += g[2] * np.sign(ally_t[1] - centroid_c)

    # mountain pull
    if mtn_t is not None:
        pref_r += g[12] * np.sign(mtn_t[0] - centroid_r)
        pref_c += g[12] * np.sign(mtn_t[1] - centroid_c)

    # river pull
    if riv_t is not None:
        pref_r += g[13] * np.sign(riv_t[0] - centroid_r)
        pref_c += g[13] * np.sign(riv_t[1] - centroid_c)

    # wall repulsion
    if g[10] != 0.0:
        dist_top = centroid_r
        dist_bot = (BOARD - 1) - centroid_r
        dist_lft = centroid_c
        dist_rgt = (BOARD - 1) - centroid_c
        if dist_top < WALL_REPULSION_RANGE:
            pref_r += g[10] * (WALL_REPULSION_RANGE - dist_top)
        if dist_bot < WALL_REPULSION_RANGE:
            pref_r -= g[10] * (WALL_REPULSION_RANGE - dist_bot)
        if dist_lft < WALL_REPULSION_RANGE:
            pref_c += g[10] * (WALL_REPULSION_RANGE - dist_lft)
        if dist_rgt < WALL_REPULSION_RANGE:
            pref_c -= g[10] * (WALL_REPULSION_RANGE - dist_rgt)

    # noise
    pref_r += g[4] * (world.rng.random() - 0.5) * 2
    pref_c += g[4] * (world.rng.random() - 0.5) * 2

    last_dr, last_dc = org.last_move_dir
    scored = []
    for dr, dc in CARDINALS:
        s = dr * pref_r + dc * pref_c
        if (dr, dc) == (0, 0):
            s -= 0.1
        elif (dr, dc) == (last_dr, last_dc):
            s += 0.4   # inertia: keep going same way
        elif (dr, dc) == (-last_dr, -last_dc):
            s -= 0.5   # don't immediately reverse (kills the vibration)
        s += world.rng.random() * 1e-3
        scored.append((s, dr, dc))
    scored.sort(reverse=True)

    for _, dr, dc in scored:
        new_cells = {(r + dr, c + dc) for (r, c) in org.cells}
        if any(r < 0 or r >= BOARD or c < 0 or c >= BOARD for r, c in new_cells):
            continue
        if new_cells & snapshot_others_cells:
            continue
        return new_cells, (dr, dc)
    return set(org.cells), (0, 0)


def _grow_org(org, food_cells, occupied_others, rng_):
    cands = []
    for (r, c) in org.cells:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD and 0 <= nc < BOARD and (nr, nc) not in org.cells:
                cands.append((nr, nc))
    if not cands:
        return False
    cands = [x for x in cands if x not in occupied_others and x not in food_cells]
    if not cands:
        return False
    org.cells.add(cands[int(rng_.integers(0, len(cands)))])
    return True


def _shrink_org(org):
    if not org.cells:
        org.alive = False
        return
    cr_r, cr_c = org.centroid()
    far = max(org.cells, key=lambda x: (x[0] - cr_r) ** 2 + (x[1] - cr_c) ** 2)
    org.cells.discard(far)
    if not org.cells:
        org.alive = False


def _effective_combat_size(cells, mountains):
    raw = len(cells)
    if raw == 0:
        return 0.0
    bonus = sum(1 for c in cells if c in mountains)
    if raw > BIG_SIZE_THRESHOLD:
        size = BIG_SIZE_THRESHOLD + (raw - BIG_SIZE_THRESHOLD) * OVERSIZE_COMBAT_FACTOR
    else:
        size = raw
    return size + bonus


# ----- world -----

class World:
    def __init__(self, genomes, rng_):
        self.rng = rng_
        self.genomes = genomes
        self.mountains, self.rivers = _spawn_biomes(rng_)
        self.organisms = self._spawn_organisms(rng_)
        self.alive_orgs = list(self.organisms)
        self.food_cells = set()
        # pre-populate food so brains have visible targets from step 0
        occupied = set()
        for o in self.alive_orgs:
            occupied |= o.cells
        while len(self.food_cells) < FOOD_TARGET:
            f = _free_cell(rng_, occupied | self.food_cells)
            if f is None:
                break
            self.food_cells.add(f)
        self.step_no = 0
        self.food_eaten = [0] * N_SPECIES
        self.kills_dealt = [0] * N_SPECIES
        self.atoms_collected = [0] * N_SPECIES
        self.survival_steps = [0] * N_SPECIES
        self.parts_built = [0] * N_SPECIES
        self.done = False
        self.end_reason = None
        # render caches (biomes never change)
        self._mtn_idx = (
            (list(zip(*self.mountains))[0], list(zip(*self.mountains))[1])
            if self.mountains else None
        )
        self._riv_idx = (
            (list(zip(*self.rivers))[0], list(zip(*self.rivers))[1])
            if self.rivers else None
        )
        # per-step caches (rebuilt each step)
        self._species_cells = [set() for _ in range(N_SPECIES)]

    def _build_species_cache(self):
        for s in self._species_cells:
            s.clear()
        for o in self.alive_orgs:
            self._species_cells[o.species_id] |= o.cells

    def _spawn_organisms(self, rng_):
        orgs = []
        occupied = set()
        for sid in range(N_SPECIES):
            cr0, cc0 = SPAWN_CENTERS[sid]
            n_orgs = INIT_ORGS[sid]
            n_cells = INIT_CELLS[sid]
            for o_idx in range(n_orgs):
                dx = (o_idx % 3) * 5
                dy = (o_idx // 3) * 5
                # distribute around center based on which corner
                if cc0 > BOARD // 2:
                    sx, sy = cr0 + dy, cc0 - dx
                else:
                    sx, sy = cr0 + dy, cc0 + dx
                sx = max(2, min(BOARD - 3, sx))
                sy = max(2, min(BOARD - 3, sy))
                cells = self._make_org_body(rng_, (sx, sy), n_cells, occupied)
                if cells:
                    occupied |= cells
                    orgs.append(Organism(sid, cells))
        return orgs

    def _make_org_body(self, rng_, start, n_cells, occupied):
        if start in occupied:
            for _ in range(20):
                start = (
                    max(2, min(BOARD - 3, start[0] + int(rng_.integers(-2, 3)))),
                    max(2, min(BOARD - 3, start[1] + int(rng_.integers(-2, 3)))),
                )
                if start not in occupied:
                    break
            else:
                return set()
        body = {start}
        while len(body) < n_cells:
            cands = []
            for c in body:
                for d in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nb = (c[0] + d[0], c[1] + d[1])
                    if 0 <= nb[0] < BOARD and 0 <= nb[1] < BOARD and nb not in body and nb not in occupied:
                        cands.append(nb)
            if not cands:
                break
            body.add(cands[int(rng_.integers(0, len(cands)))])
        return body

    def _refresh_alive(self):
        self.alive_orgs = [o for o in self.organisms if o.alive]

    def _spawn_food(self):
        occupied = set(self.food_cells)
        for o in self.alive_orgs:
            occupied |= o.cells
        for _ in range(SPAWN_TRIES):
            if len(self.food_cells) >= FOOD_TARGET:
                break
            f = _free_cell(self.rng, occupied)
            if f:
                self.food_cells.add(f)
                occupied.add(f)

    def _decide_action(self, org):
        """Diet-driven action priority.
        Carnivores: hunt-first (move/shoot). Build gun ASAP. Don't waste atoms growing.
        Herb/omni:  build/grow first to get tanky and armed, then move/shoot."""
        g = self.genomes[org.species_id]
        diet = DIETS[org.species_id]

        if diet == "carn":
            # priority: gun build > shoot in range > spike if cheap > MOVE (hunt)
            if not org.has_gun and org.atoms >= GUN_COST:
                return ("build", "gun")
            if org.has_gun:
                target = self._nearest_visible_enemy_cell(org)
                if target is not None and self._in_shoot_range(org, target):
                    return ("shoot", target)
            if not org.has_spike and org.atoms >= SPIKE_COST + max(0.0, g[8]) * 15:
                return ("build", "spike")
            # otherwise hunt — even with no atoms or out-of-range gun, MOVE toward prey
            return ("move",)

        # herb / omni: build defenses, grow body, then act
        if not org.has_eye and org.atoms >= EYE_COST + max(0.0, g[6]) * 30:
            return ("build", "eye")
        if not org.has_spike and org.atoms >= SPIKE_COST + max(0.0, g[8]) * 20:
            return ("build", "spike")
        if not org.has_gun and org.atoms >= GUN_COST + max(0.0, g[7]) * 30:
            return ("build", "gun")
        size_cap = BIG_SIZE_THRESHOLD - int(max(0.0, g[11]) * 5)
        size_cap = max(5, size_cap)
        grow_thresh = BODY_COST + max(0.0, g[9]) * 20
        if len(org.cells) < size_cap and org.atoms >= grow_thresh:
            return ("grow",)
        if org.has_gun:
            target = self._nearest_visible_enemy_cell(org)
            if target is not None and self._in_shoot_range(org, target):
                return ("shoot", target)
        return ("move",)

    def _in_shoot_range(self, org, cell):
        cr_r, cr_c = org.centroid()
        return abs(cell[0] - cr_r) + abs(cell[1] - cr_c) <= SHOOT_RANGE

    def _nearest_visible_enemy_cell(self, org):
        centroid = org.centroid()
        cr_r, cr_c = centroid
        # use cached enemy set (cells across all other species)
        enemy = set()
        for sid in range(N_SPECIES):
            if sid != org.species_id:
                enemy |= self._species_cells[sid]
        if not enemy:
            return None
        if org.has_eye:
            best, best_d = None, 10**9
            for c in enemy:
                d = abs(c[0] - cr_r) + abs(c[1] - cr_c)
                if d < best_d:
                    best_d, best = d, c
            return best
        best, best_d = None, 10**9
        for c in enemy:
            d = abs(c[0] - cr_r) + abs(c[1] - cr_c)
            if d <= SIGHT_RADIUS and d < best_d:
                best_d, best = d, c
        return best

    def _do_build(self, org, part):
        if part == "eye" and not org.has_eye:
            org.atoms -= EYE_COST
            org.has_eye = True
            self.parts_built[org.species_id] += 1
        elif part == "gun" and not org.has_gun:
            org.atoms -= GUN_COST
            org.has_gun = True
            self.parts_built[org.species_id] += 1
        elif part == "spike" and not org.has_spike:
            org.atoms -= SPIKE_COST
            org.has_spike = True
            self.parts_built[org.species_id] += 1

    def _do_grow(self, org):
        occupied = set()
        for x in self.alive_orgs:
            if x is not org:
                occupied |= x.cells
        if _grow_org(org, self.food_cells, occupied | org.cells, self.rng):
            org.atoms -= BODY_COST

    def _do_shoot(self, shooter, target_cell):
        for o in self.alive_orgs:
            if o.species_id == shooter.species_id or o is shooter:
                continue
            if target_cell in o.cells:
                o.cells.discard(target_cell)
                shooter.atoms = min(ATOM_CAP, shooter.atoms + 2)
                self.kills_dealt[shooter.species_id] += 1
                self.atoms_collected[shooter.species_id] += 2
                shooter.hunger = 0  # eating prey resets hunger
                # auto-grow shooter (gun kill = eating)
                others_cells = set()
                for x in self.alive_orgs:
                    if x is not shooter and x is not o:
                        others_cells |= x.cells
                others_cells |= o.cells  # remaining victim cells still occupy space
                _grow_org(shooter, self.food_cells, others_cells | shooter.cells, self.rng)
                if not o.cells:
                    o.alive = False
                return

    def _move_phase(self, movers):
        if not movers:
            return
        # snapshot all current alive cells (for plan validity)
        all_cells = set()
        for o in self.alive_orgs:
            all_cells |= o.cells

        # account for slowdown: oversized orgs skip some frames
        active_movers = []
        for o in movers:
            size = len(o.cells)
            period = 1
            if size > BIG_SIZE_THRESHOLD:
                period = 1 + (size - BIG_SIZE_THRESHOLD) // SIZE_SLOWDOWN_STEP
            if o.move_skip_counter < period - 1:
                o.move_skip_counter += 1
            else:
                o.move_skip_counter = 0
                active_movers.append(o)

        if not active_movers:
            return

        self.rng.shuffle(active_movers)

        plans = []
        for o in active_movers:
            others = all_cells - o.cells
            plan, plan_dir = _decide_move(self.genomes[o.species_id], o, self, others)
            plans.append((o, plan, plan_dir))

        # conflict resolution: bigger orgs claim first
        plans.sort(key=lambda x: -len(x[0].cells))
        claimed = set()
        active_set = {id(o) for o in active_movers}
        for o in self.alive_orgs:
            if id(o) not in active_set:
                claimed |= o.cells
        for org, plan, plan_dir in plans:
            if plan & claimed:
                claimed |= org.cells  # revert
                org.last_move_dir = (0, 0)  # break inertia so next tick tries new dir
            else:
                org.cells = set(plan)
                org.last_move_dir = plan_dir
                claimed |= org.cells

    def _eat_phase(self):
        for o in self.alive_orgs:
            diet = DIETS[o.species_id]
            if diet not in ("herb", "omni"):
                continue
            eaten = o.cells & self.food_cells
            if not eaten:
                continue
            self.food_cells -= eaten
            gained = len(eaten)
            o.atoms = min(ATOM_CAP, o.atoms + gained)
            o.hunger = 0
            self.food_eaten[o.species_id] += gained
            self.atoms_collected[o.species_id] += gained
            # auto-grow body: +1 cell per food eaten
            others_cells = set()
            for x in self.alive_orgs:
                if x is not o:
                    others_cells |= x.cells
            for _ in range(gained):
                _grow_org(o, self.food_cells, others_cells | o.cells, self.rng)

    def _combat_phase(self):
        if len(self.alive_orgs) < 2:
            return
        # cell ownership
        cell_owner = {}
        for o in self.alive_orgs:
            for c in o.cells:
                cell_owner[c] = o
        if not cell_owner:
            return
        # group defense: count adjacent cells from OTHER same-species organisms
        # (own-body density does not grant bonus; pack cohesion does)
        ally_count = {}
        for c, owner in cell_owner.items():
            count = 0
            for dr, dc in EIGHT:
                nb = (c[0] + dr, c[1] + dc)
                ob = cell_owner.get(nb)
                if ob is not None and ob is not owner and ob.species_id == owner.species_id:
                    count += 1
            ally_count[c] = count

        # attacker count per cell (only carn/omni species can deal melee)
        hit_count = {}
        first_attacker = {}
        for c, owner in cell_owner.items():
            attackers = []
            for dr, dc in EIGHT:
                nb = (c[0] + dr, c[1] + dc)
                ob = cell_owner.get(nb)
                if ob is None or ob.species_id == owner.species_id:
                    continue
                if DIETS[ob.species_id] not in ("carn", "omni"):
                    continue
                attackers.append(ob)
            if attackers:
                hit_count[c] = len(attackers)
                first_attacker[c] = attackers[0]

        # apply: cell dies if hits > ally_count + 1 (baseline 1 + group bonus)
        # so an isolated cell needs >=2 attackers to die; pack cells need more
        spike_counter = {}
        attacker_growth = {}  # attacker -> count of kills this combat phase
        for vc, hits in hit_count.items():
            defense = ally_count[vc] + 1
            if hits > defense:
                victim = cell_owner[vc]
                attacker = first_attacker[vc]
                attacker.atoms = min(ATOM_CAP, attacker.atoms + 2)
                self.kills_dealt[attacker.species_id] += 1
                self.atoms_collected[attacker.species_id] += 2
                attacker_growth[id(attacker)] = attacker_growth.get(id(attacker), [attacker, 0])
                attacker_growth[id(attacker)][1] += 1
                if victim.has_spike:
                    spike_counter[id(attacker)] = spike_counter.get(id(attacker), [attacker, 0])
                    spike_counter[id(attacker)][1] += 1
                victim.cells.discard(vc)
                if not victim.cells:
                    victim.alive = False

        # auto-grow attackers per kill (carnivore eating prey)
        for atk_id, (atk, n) in attacker_growth.items():
            if not atk.alive:
                continue
            others = set()
            for x in self.alive_orgs:
                if x is not atk:
                    others |= x.cells
            for _ in range(n):
                _grow_org(atk, self.food_cells, others | atk.cells, self.rng)

        for atk_id, (atk, n) in spike_counter.items():
            if not atk.alive:
                continue
            for _ in range(n):
                _shrink_org(atk)
                if not atk.alive:
                    break

    def _hunger_phase(self):
        for o in self.alive_orgs:
            if not o.alive:
                continue
            on_river = self.rivers and any(c in self.rivers for c in o.cells)
            if on_river:
                continue
            o.hunger += 1
            if o.hunger > MAX_HUNGER and (o.hunger - MAX_HUNGER) % STARVE_PERIOD == 0:
                _shrink_org(o)

    def _alive_species_count(self):
        seen = set()
        for o in self.alive_orgs:
            if o.alive:
                seen.add(o.species_id)
        return len(seen), seen

    def step(self):
        if self.done:
            return
        self._spawn_food()
        self._refresh_alive()

        if not self.alive_orgs:
            self.done = True
            self.end_reason = "wipeout"
            return

        self._build_species_cache()

        # Decide actions for all alive orgs
        actions = [(o, self._decide_action(o)) for o in self.alive_orgs]

        # Phase: builds (free)
        for o, act in actions:
            if act[0] == "build":
                self._do_build(o, act[1])
        # Phase: grows
        for o, act in actions:
            if act[0] == "grow":
                self._do_grow(o)
        # Phase: shoots
        for o, act in actions:
            if act[0] == "shoot":
                self._do_shoot(o, act[1])

        # Refresh after shoots may have killed orgs
        self._refresh_alive()
        self._build_species_cache()

        # Phase: moves
        movers = [o for o, act in actions if act[0] == "move" and o.alive]
        self._move_phase(movers)

        # Phase: eat
        self._refresh_alive()
        self._eat_phase()

        # Phase: combat
        self._refresh_alive()
        self._combat_phase()

        # Phase: hunger
        self._refresh_alive()
        self._hunger_phase()

        self._refresh_alive()
        # bookkeeping
        for o in self.alive_orgs:
            self.survival_steps[o.species_id] += 1

        self.step_no += 1
        n_alive, _ = self._alive_species_count()
        if n_alive <= 1:
            self.done = True
            self.end_reason = "winner" if n_alive == 1 else "wipeout"
        elif self.step_no >= STALEMATE_CAP:
            self.done = True
            self.end_reason = "stalemate"

    def fitness(self):
        winner_sid = None
        if self.end_reason == "winner":
            _, alive_sids = self._alive_species_count()
            if alive_sids:
                winner_sid = next(iter(alive_sids))
        out = []
        for sid in range(N_SPECIES):
            cells_alive = sum(len(o.cells) for o in self.organisms if o.alive and o.species_id == sid)
            bonus = WIN_BONUS if sid == winner_sid else 0.0
            out.append(
                W_SURV * self.survival_steps[sid]
                + W_KILL * self.kills_dealt[sid]
                + W_FOOD * self.food_eaten[sid]
                + W_ATOMS * self.atoms_collected[sid]
                + W_SIZE * cells_alive
                + bonus
            )
        return out


# ----- GA -----

def random_genome(rng_):
    g = rng_.uniform(-1.0, 1.0, GENOME_SIZE)
    g[0] = rng_.uniform(0.7, 1.8)   # food pull (strong default — pursue food)
    g[1] = rng_.uniform(0.3, 1.2)   # prey pull
    g[2] = rng_.uniform(0.0, 1.0)   # ally cohesion
    g[3] = rng_.uniform(0.0, 1.0)   # flee
    g[5] = rng_.uniform(0.0, 1.0)   # hunger urgency
    g[6] = rng_.uniform(0.0, 1.0)   # eye threshold
    g[7] = rng_.uniform(0.0, 1.0)   # gun threshold
    g[8] = rng_.uniform(0.0, 1.0)   # spike threshold
    g[9] = rng_.uniform(0.0, 1.0)   # body grow threshold
    g[10] = rng_.uniform(0.0, 1.0)  # wall repulsion
    g[11] = rng_.uniform(0.0, 1.0)  # oversize aversion
    g[12] = rng_.uniform(-0.3, 1.0)  # mountain pull
    g[13] = rng_.uniform(-0.3, 1.0)  # river pull
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


# ----- trainer -----

class Trainer:
    def __init__(self):
        self.pops = [[random_genome(rng) for _ in range(POP)] for _ in range(N_SPECIES)]
        self.champions = [self.pops[t][0].copy() for t in range(N_SPECIES)]
        self.champ_score = [-np.inf] * N_SPECIES
        self.gen = 0
        self.history_best = [[] for _ in range(N_SPECIES)]
        self.history_mean = [[] for _ in range(N_SPECIES)]
        self.demo_gen = 0
        self.world = World([c.copy() for c in self.champions], rng)

    def start_demo(self):
        self.demo_gen = self.gen
        self.world = World([c.copy() for c in self.champions], rng)

    def evaluate_and_breed(self):
        idxs = [rng.permutation(POP) for _ in range(N_SPECIES)]
        scores = [np.zeros(POP) for _ in range(N_SPECIES)]
        for k in range(POP):
            triple = [self.pops[t][idxs[t][k]] for t in range(N_SPECIES)]
            w = World(triple, rng)
            while not w.done:
                w.step()
            fit = w.fitness()
            for t in range(N_SPECIES):
                scores[t][idxs[t][k]] = fit[t]

        for t in range(N_SPECIES):
            self.history_best[t].append(float(scores[t].max()))
            self.history_mean[t].append(float(scores[t].mean()))
            best_idx = int(np.argmax(scores[t]))
            if scores[t][best_idx] > self.champ_score[t]:
                self.champ_score[t] = float(scores[t][best_idx])
                self.champions[t] = self.pops[t][best_idx].copy()

        print("gen {:3d} | ".format(self.gen) + " | ".join(
            "{:6s} best={:8.1f} mean={:8.1f}".format(
                LABELS[t], self.history_best[t][-1], self.history_mean[t][-1])
            for t in range(N_SPECIES)))

        for t in range(N_SPECIES):
            elite_idx = sorted(range(POP), key=lambda i: scores[t][i], reverse=True)[:ELITE]
            new_pop = [self.pops[t][i].copy() for i in elite_idx]
            while len(new_pop) < POP:
                a = tournament(self.pops[t], scores[t], rng)
                b = tournament(self.pops[t], scores[t], rng)
                child = mutate(crossover(a, b, rng), rng)
                new_pop.append(child)
            self.pops[t] = new_pop

        self.gen += 1


trainer = Trainer()


# ----- render -----

def render_world(w):
    img = np.broadcast_to(EMPTY_COLOR_ARR, (BOARD, BOARD, 3)).copy()
    if w._mtn_idx is not None:
        img[w._mtn_idx[0], w._mtn_idx[1]] = MOUNTAIN_COLOR_ARR
    if w._riv_idx is not None:
        img[w._riv_idx[0], w._riv_idx[1]] = RIVER_COLOR_ARR
    if w.food_cells:
        rs, cs = zip(*w.food_cells)
        img[list(rs), list(cs)] = FOOD_COLOR_ARR
    for o in w.alive_orgs:
        if not o.cells:
            continue
        rs, cs = zip(*o.cells)
        base = SPECIES_COLOR_ARR[o.species_id]
        n_parts = int(o.has_eye) + int(o.has_gun) + int(o.has_spike)
        color = np.clip(base + 0.05 * n_parts, 0, 1) if n_parts else base
        img[list(rs), list(cs)] = color
    return img


# ----- figure -----

fig, (ax_board, ax_fit) = plt.subplots(1, 2, figsize=(14, 7),
                                       gridspec_kw={"width_ratios": [1.1, 1]})
ax_board.set_xticks([])
ax_board.set_yticks([])
im = ax_board.imshow(render_world(trainer.world), interpolation="nearest")
title = ax_board.set_title("gen 0 demo — step 0")

ax_fit.set_xlim(0, GENERATIONS)
ax_fit.set_xlabel("generation")
ax_fit.set_ylabel("fitness")
ax_fit.set_title("learning curves")
ax_fit.grid(alpha=0.3)
fit_lines_best = [
    ax_fit.plot([], [], color=SPECIES_COLORS[t], linewidth=2,
                label=f"{LABELS[t]} ({DIETS[t]}) best")[0]
    for t in range(N_SPECIES)
]
fit_lines_mean = [
    ax_fit.plot([], [], color=SPECIES_COLORS[t], linewidth=1, alpha=0.4, linestyle="--",
                label=f"{LABELS[t]} mean")[0]
    for t in range(N_SPECIES)
]
ax_fit.legend(ncol=3, fontsize=7, loc="upper left")

status = fig.text(0.5, 0.015, "", ha="center", fontsize=8)


def update_fit_lines():
    for t in range(N_SPECIES):
        x = list(range(len(trainer.history_best[t])))
        fit_lines_best[t].set_data(x, trainer.history_best[t])
        fit_lines_mean[t].set_data(x, trainer.history_mean[t])
    ax_fit.relim()
    ax_fit.autoscale_view()


def species_summary(w):
    parts_total = [0] * N_SPECIES  # eye/gun/spike counts
    eye_n = [0] * N_SPECIES
    gun_n = [0] * N_SPECIES
    spk_n = [0] * N_SPECIES
    cells_n = [0] * N_SPECIES
    org_n = [0] * N_SPECIES
    atoms_n = [0] * N_SPECIES
    for o in w.organisms:
        if not o.alive:
            continue
        sid = o.species_id
        org_n[sid] += 1
        cells_n[sid] += len(o.cells)
        atoms_n[sid] += o.atoms
        eye_n[sid] += int(o.has_eye)
        gun_n[sid] += int(o.has_gun)
        spk_n[sid] += int(o.has_spike)
    parts = [(eye_n[t], gun_n[t], spk_n[t]) for t in range(N_SPECIES)]
    return org_n, cells_n, atoms_n, parts


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
            anim.event_source.stop()
            title.set_text(f"training complete — final demo ({w.end_reason})")
            return [im, title, status]

    im.set_data(render_world(trainer.world))
    org_n, cells_n, atoms_n, parts = species_summary(trainer.world)
    parts_str = lambda p: f"E{p[0]}G{p[1]}S{p[2]}"
    parts_lines = "  ".join(
        f"{LABELS[i]}({DIETS[i]}): orgs={org_n[i]} cells={cells_n[i]} "
        f"atm={atoms_n[i]} {parts_str(parts[i])}"
        for i in range(N_SPECIES)
    )
    title.set_text(
        f"gen {trainer.demo_gen} demo — step {trainer.world.step_no}"
        f"  |  food={len(trainer.world.food_cells)}"
    )
    status.set_text(parts_lines)
    return [im, title, status]


anim = animation.FuncAnimation(
    fig, update, interval=FRAME_INTERVAL, blit=False, repeat=False, cache_frame_data=False
)
plt.tight_layout(rect=(0, 0.04, 1, 1))
plt.show()
