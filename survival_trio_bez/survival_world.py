"""World rules for the simple three-team survival simulation."""

import numpy as np

from survival_common import (
    BIG_SIZE_THRESHOLD,
    BIOME_PATCH_RADIUS,
    BOARD,
    CARDINALS,
    CORNERS,
    FOOD_TARGET,
    INIT_BODY_SIZE,
    MAX_HUNGER,
    N_MOUNTAIN_PATCHES,
    N_RIVER_PATCHES,
    OVERSIZE_COMBAT_FACTOR,
    SIZE_SLOWDOWN_STEP,
    SPAWN_TRIES,
    STALEMATE_CAP,
    STARVE_PERIOD,
    W_FOOD,
    W_KILL,
    W_SIZE,
    W_SURV,
    WALL_REPULSION_RANGE,
    WIN_BONUS,
)


class Creature:
    __slots__ = ("team_id", "cells", "hunger", "alive", "move_skip_counter")

    def __init__(self, team_id, cells):
        self.team_id = team_id
        self.cells = set(cells)
        self.hunger = 0
        self.alive = True
        self.move_skip_counter = 0

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


def _make_body(rng_, corner):
    r0, c0 = corner
    body = set()
    while len(body) < INIT_BODY_SIZE:
        body.add((r0 + int(rng_.integers(0, 3)), c0 + int(rng_.integers(0, 3))))
    return body


def _free_cell(rng_, occupied):
    for _ in range(60):
        p = (int(rng_.integers(0, BOARD)), int(rng_.integers(0, BOARD)))
        if p not in occupied:
            return p
    return None


def _spawn_biomes(rng_):
    """Spawn mountain and river patches, avoiding starting corners."""
    forbidden = set()
    for (r0, c0) in CORNERS:
        for dr in range(INIT_BODY_SIZE + 2):
            for dc in range(INIT_BODY_SIZE + 2):
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
        cr_r = int(rng_.integers(6, BOARD - 6))
        cr_c = int(rng_.integers(6, BOARD - 6))
        rad = int(rng_.integers(BIOME_PATCH_RADIUS[0], BIOME_PATCH_RADIUS[1] + 1))
        mountains |= _patch((cr_r, cr_c), rad, forbidden)
    rivers = set()
    for _ in range(N_RIVER_PATCHES):
        cr_r = int(rng_.integers(6, BOARD - 6))
        cr_c = int(rng_.integers(6, BOARD - 6))
        rad = int(rng_.integers(BIOME_PATCH_RADIUS[0], BIOME_PATCH_RADIUS[1] + 1))
        rivers |= _patch((cr_r, cr_c), rad, forbidden | mountains)
    return mountains, rivers


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


def _two_nearest_enemies(centroid, others_by_team):
    out = []
    for ob in others_by_team.values():
        if not ob:
            continue
        et, ed = _nearest_to(centroid, ob)
        out.append((ed, et, len(ob)))
    out.sort()
    return [(t, s) for _, t, s in out[:2]]


def _decide_move(genome, cr, others_by_team, food_cells, mountains, rivers, rng_):
    centroid = cr.centroid()
    centroid_r, centroid_c = centroid
    size_self = len(cr.cells)

    food_t, _ = _nearest_to(centroid, food_cells) if food_cells else (None, 0)
    enemies = _two_nearest_enemies(centroid, others_by_team)
    mtn_t, _ = _nearest_to(centroid, mountains) if mountains else (None, 0)
    riv_t, _ = _nearest_to(centroid, rivers) if rivers else (None, 0)

    pref_r = pref_c = 0.0

    if food_t is not None:
        hunger_factor = 1.0 + genome[1] * (cr.hunger / MAX_HUNGER)
        oversize_factor = 1.0
        if size_self > BIG_SIZE_THRESHOLD and genome[9] > 0:
            excess_norm = (size_self - BIG_SIZE_THRESHOLD) / BIG_SIZE_THRESHOLD
            oversize_factor = max(0.0, 1.0 - genome[9] * excess_norm)
        scale = genome[0] * hunger_factor * oversize_factor
        pref_r += scale * np.sign(food_t[0] - centroid_r)
        pref_c += scale * np.sign(food_t[1] - centroid_c)

    if len(enemies) >= 1:
        et, esize = enemies[0]
        ratio = size_self / max(1, esize)
        strength = genome[2] * (ratio - 1.0)
        pref_r += strength * np.sign(et[0] - centroid_r)
        pref_c += strength * np.sign(et[1] - centroid_c)

    if len(enemies) >= 2:
        et2, esize2 = enemies[1]
        ratio2 = size_self / max(1, esize2)
        strength2 = genome[5] * (ratio2 - 1.0)
        pref_r += strength2 * np.sign(et2[0] - centroid_r)
        pref_c += strength2 * np.sign(et2[1] - centroid_c)

    if mtn_t is not None:
        pref_r += genome[7] * np.sign(mtn_t[0] - centroid_r)
        pref_c += genome[7] * np.sign(mtn_t[1] - centroid_c)

    if riv_t is not None:
        pref_r += genome[8] * np.sign(riv_t[0] - centroid_r)
        pref_c += genome[8] * np.sign(riv_t[1] - centroid_c)

    if genome[6] != 0.0:
        dist_top = centroid_r
        dist_bot = (BOARD - 1) - centroid_r
        dist_lft = centroid_c
        dist_rgt = (BOARD - 1) - centroid_c
        if dist_top < WALL_REPULSION_RANGE:
            pref_r += genome[6] * (WALL_REPULSION_RANGE - dist_top)
        if dist_bot < WALL_REPULSION_RANGE:
            pref_r -= genome[6] * (WALL_REPULSION_RANGE - dist_bot)
        if dist_lft < WALL_REPULSION_RANGE:
            pref_c += genome[6] * (WALL_REPULSION_RANGE - dist_lft)
        if dist_rgt < WALL_REPULSION_RANGE:
            pref_c -= genome[6] * (WALL_REPULSION_RANGE - dist_rgt)

    pref_r += genome[3] * (rng_.random() - 0.5) * 2
    pref_c += genome[3] * (rng_.random() - 0.5) * 2

    others = set()
    for ob in others_by_team.values():
        others |= ob

    scored = []
    for dr, dc in CARDINALS:
        s = dr * pref_r + dc * pref_c
        if (dr, dc) == (0, 0):
            s -= genome[4]
        s += rng_.random() * 1e-3
        scored.append((s, dr, dc))
    scored.sort(reverse=True)

    for _, dr, dc in scored:
        new_cells = {(r + dr, c + dc) for (r, c) in cr.cells}
        if any(r < 0 or r >= BOARD or c < 0 or c >= BOARD for r, c in new_cells):
            continue
        if new_cells & others:
            continue
        return new_cells
    return set(cr.cells)


def _grow(cr, food_cells, occupied_by_others, rng_):
    cands = []
    for (r, c) in cr.cells:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD and 0 <= nc < BOARD and (nr, nc) not in cr.cells:
                cands.append((nr, nc))
    if not cands:
        return
    cands = [x for x in cands if x not in occupied_by_others and x not in food_cells]
    if not cands:
        return
    cr.cells.add(cands[int(rng_.integers(0, len(cands)))])


def _shrink(cr):
    if not cr.cells:
        cr.alive = False
        return
    cr_r, cr_c = cr.centroid()
    far = max(cr.cells, key=lambda x: (x[0] - cr_r) ** 2 + (x[1] - cr_c) ** 2)
    cr.cells.discard(far)
    if not cr.cells:
        cr.alive = False


def _effective_combat_size(cells, mountains):
    raw = len(cells)
    if raw == 0:
        return 0.0
    mountain_bonus = sum(1 for c in cells if c in mountains)
    if raw > BIG_SIZE_THRESHOLD:
        excess = raw - BIG_SIZE_THRESHOLD
        size = BIG_SIZE_THRESHOLD + excess * OVERSIZE_COMBAT_FACTOR
    else:
        size = raw
    return size + mountain_bonus


def _dilate8(grid):
    out = grid.copy()
    out[1:, :] |= grid[:-1, :]
    out[:-1, :] |= grid[1:, :]
    out[:, 1:] |= grid[:, :-1]
    out[:, :-1] |= grid[:, 1:]
    out[1:, 1:] |= grid[:-1, :-1]
    out[:-1, :-1] |= grid[1:, 1:]
    out[1:, :-1] |= grid[:-1, 1:]
    out[:-1, 1:] |= grid[1:, :-1]
    return out


class World:
    def __init__(self, genomes, rng_):
        self.rng = rng_
        self.genomes = genomes
        self.mountains, self.rivers = _spawn_biomes(rng_)
        self.creatures = [Creature(i + 1, _make_body(rng_, CORNERS[i])) for i in range(3)]
        self.food_cells = set()
        self.step_no = 0
        self.food_eaten = [0, 0, 0]
        self.kills_dealt = [0.0, 0.0, 0.0]
        self.max_size = [INIT_BODY_SIZE] * 3
        self.survival_steps = [0, 0, 0]
        self.done = False
        self.end_reason = None

    def _spawn_food(self):
        occupied = set(self.food_cells)
        for cr in self.creatures:
            if cr.alive:
                occupied |= cr.cells
        for _ in range(SPAWN_TRIES):
            if len(self.food_cells) >= FOOD_TARGET:
                break
            f = _free_cell(self.rng, occupied)
            if f:
                self.food_cells.add(f)
                occupied.add(f)

    def _move_phase(self):
        alive = [cr for cr in self.creatures if cr.alive]
        if not alive:
            return

        movers = []
        for cr in alive:
            size = len(cr.cells)
            period = 1
            if size > BIG_SIZE_THRESHOLD:
                period = 1 + (size - BIG_SIZE_THRESHOLD) // SIZE_SLOWDOWN_STEP
            if cr.move_skip_counter < period - 1:
                cr.move_skip_counter += 1
            else:
                cr.move_skip_counter = 0
                movers.append(cr)

        if not movers:
            return

        snapshot = {cr.team_id: frozenset(cr.cells) for cr in alive}
        order = list(movers)
        self.rng.shuffle(order)

        plans = {}
        for cr in order:
            tid = cr.team_id
            others_by_team = {ot: snapshot[ot] for ot in snapshot if ot != tid}
            plans[tid] = _decide_move(
                self.genomes[tid - 1], cr, others_by_team,
                self.food_cells, self.mountains, self.rivers, self.rng,
            )

        size_order = sorted(plans.keys(), key=lambda t: -len(snapshot[t]))
        claimed = set()
        non_mover_cells = set()
        for cr in alive:
            if cr.team_id not in plans:
                non_mover_cells |= snapshot[cr.team_id]
        claimed |= non_mover_cells

        final_plan = {}
        for tid in size_order:
            plan = plans[tid]
            if plan & claimed:
                plan = set(snapshot[tid])
            final_plan[tid] = plan
            claimed |= plan

        team_to_creature = {cr.team_id: cr for cr in alive}
        for tid, cells in final_plan.items():
            team_to_creature[tid].cells = set(cells)

    def _eat_phase(self):
        for i, cr in enumerate(self.creatures):
            if not cr.alive:
                continue
            eaten = cr.cells & self.food_cells
            if not eaten:
                continue
            self.food_cells -= eaten
            cr.hunger = 0
            self.food_eaten[i] += len(eaten)
            others_cells = set()
            for j, oc in enumerate(self.creatures):
                if oc.alive and j != i:
                    others_cells |= oc.cells
            for _ in range(len(eaten)):
                _grow(cr, self.food_cells, others_cells | cr.cells, self.rng)

    def _combat_phase(self):
        alive_creatures = [cr for cr in self.creatures if cr.alive]
        if len(alive_creatures) < 2:
            return

        bodies = {cr.team_id: cr.cells for cr in alive_creatures}
        sizes = {tid: _effective_combat_size(b, self.mountains) for tid, b in bodies.items()}

        grids = {}
        for tid, body in bodies.items():
            g = np.zeros((BOARD, BOARD), dtype=bool)
            for (r, c) in body:
                g[r, c] = True
            grids[tid] = g
        dilations = {tid: _dilate8(g) for tid, g in grids.items()}

        team_to_creature = {cr.team_id: cr for cr in alive_creatures}

        for victim_tid in list(bodies.keys()):
            larger = [t for t in bodies if t != victim_tid and sizes[t] > sizes[victim_tid]]
            if not larger:
                continue
            v_grid = grids[victim_tid]
            attribution = np.full((BOARD, BOARD), -1, dtype=np.int8)
            death_mask = np.zeros((BOARD, BOARD), dtype=bool)
            for atk_tid in sorted(larger):
                contact = v_grid & dilations[atk_tid]
                new_kill = contact & (attribution == -1)
                attribution[new_kill] = atk_tid
                death_mask |= contact

            if not death_mask.any():
                continue

            dead_rs, dead_cs = np.where(death_mask)
            lost_cells = set(zip(dead_rs.tolist(), dead_cs.tolist()))
            v_cr = team_to_creature[victim_tid]
            v_cr.cells -= lost_cells
            if not v_cr.cells:
                v_cr.alive = False

            for atk_tid in sorted(larger):
                kill_count = int((attribution == atk_tid).sum())
                if kill_count == 0:
                    continue
                self.kills_dealt[atk_tid - 1] += kill_count
                a_cr = team_to_creature[atk_tid]
                if not a_cr.alive:
                    continue
                occupied_others = set()
                for ocr in self.creatures:
                    if ocr.alive and ocr.team_id != atk_tid:
                        occupied_others |= ocr.cells
                for _ in range(kill_count):
                    _grow(a_cr, self.food_cells, occupied_others | a_cr.cells, self.rng)

    def _hunger_phase(self):
        for cr in self.creatures:
            if not cr.alive:
                continue
            on_river = self.rivers and any(c in self.rivers for c in cr.cells)
            if on_river:
                continue
            cr.hunger += 1
            if cr.hunger > MAX_HUNGER and (cr.hunger - MAX_HUNGER) % STARVE_PERIOD == 0:
                _shrink(cr)

    def step(self):
        if self.done:
            return
        self._spawn_food()
        self._move_phase()
        self._eat_phase()
        self._combat_phase()
        self._hunger_phase()

        for i, cr in enumerate(self.creatures):
            if cr.alive:
                self.survival_steps[i] += 1
                if len(cr.cells) > self.max_size[i]:
                    self.max_size[i] = len(cr.cells)

        self.step_no += 1
        alive_teams = sum(1 for cr in self.creatures if cr.alive)
        if alive_teams <= 1:
            self.done = True
            self.end_reason = "winner" if alive_teams == 1 else "wipeout"
        elif self.step_no >= STALEMATE_CAP:
            self.done = True
            self.end_reason = "stalemate"

    def fitness(self):
        out = []
        winner_idx = None
        if self.end_reason == "winner":
            for i in range(3):
                if self.creatures[i].alive:
                    winner_idx = i
                    break
        for i in range(3):
            cr = self.creatures[i]
            final_size = len(cr.cells) if cr.alive else 0
            bonus = WIN_BONUS if i == winner_idx else 0.0
            out.append(
                W_KILL * self.kills_dealt[i]
                + W_FOOD * self.food_eaten[i]
                + W_SIZE * final_size
                + W_SURV * self.survival_steps[i]
                + bonus
            )
        return out
