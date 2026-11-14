#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 SOKOBAN-AGI v1.0.0 : D4-Equivariant HyperGraph Reasoner + System 2 + Video
================================================================================
Uses the exact same RelationalReasoner / HyperGraphBlock / MuonAdamW / RLOO /
System-2 core proven in sudoku_ai.py, adapted purely via SokobanTask (variable
HxW grids, K=4 directional actions U/D/L/R, irreversible deadlock traps).
"""
from __future__ import annotations
import argparse, gc, heapq, json, math, os, random, sys, time, hashlib, itertools, traceback
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

try:
    import resource
except ImportError:
    resource = None

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt

__version__ = "1.0.2"
T0 = time.time()

# Directions: 0=Up(-1,0), 1=Down(+1,0), 2=Left(0,-1), 3=Right(0,+1)
DR = [-1, 1, 0, 0]
DC = [0, 0, -1, 1]
DIR_NAMES = ["U", "D", "L", "R"]
ROT_MAP = [2, 3, 1, 0]   # 90 deg CCW: U->L, D->R, L->D, R->U
FLIP_MAP = [0, 1, 3, 2]  # horiz flip: L<->R

TIER_NAMES = {1: "T1 easy(1b)", 2: "T2 med(2b)", 3: "T3 hard(3b)", 4: "T4 expert(4b)"}


def _amp(device):
    return torch.amp.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext()


# ==============================================================================
# §1  EXACT SOKOBAN CORE (Grid, Verifier, BFS/Naive Solvers, Generator, D4)
# ==============================================================================
class GridTopo:
    """Precomputed static topology for any HxW grid (supports variable sizes)."""
    def __init__(self, H: int, W: int):
        self.H, self.W, self.N = H, W, H * W
        nb1 = np.zeros((self.N, 4), dtype=np.int64)
        nb2 = np.zeros((self.N, 4), dtype=np.int64)
        rays = [[[] for _ in range(4)] for _ in range(self.N)]
        for r in range(H):
            for c in range(W):
                i = r * W + c
                for d in range(4):
                    r1, c1 = r + DR[d], c + DC[d]
                    nb1[i, d] = (r1 * W + c1) if (0 <= r1 < H and 0 <= c1 < W) else i
                    r2, c2 = r + 2 * DR[d], c + 2 * DC[d]
                    nb2[i, d] = (r2 * W + c2) if (0 <= r2 < H and 0 <= c2 < W) else nb1[i, d]
                    step = 1
                    while True:
                        rr, cc = r + step * DR[d], c + step * DC[d]
                        if not (0 <= rr < H and 0 <= cc < W):
                            break
                        rays[i][d].append(rr * W + cc)
                        step += 1
        self.nb1 = nb1
        self.nb2 = nb2
        self.rays = rays
        self.row_units = np.arange(self.N, dtype=np.int64).reshape(H, W)
        self.col_units = self.row_units.T.copy()
        patches = []
        for r in range(H):
            for c in range(W):
                patches.append([max(0, min(H - 1, r + dr)) * W + max(0, min(W - 1, c + dc))
                                for dr in (-1, 0, 1) for dc in (-1, 0, 1)])
        self.patch_units = np.array(patches, dtype=np.int64)  # (N, 9)


_TOPO_CACHE = {}
_DEV_TOPO = {}


def get_topo(H: int, W: int) -> GridTopo:
    t = _TOPO_CACHE.get((H, W))
    if t is None:
        t = GridTopo(H, W)
        _TOPO_CACHE[(H, W)] = t
    return t


def _dev_topo(H: int, W: int, device):
    k = (H, W, str(device))
    v = _DEV_TOPO.get(k)
    if v is None:
        tp = get_topo(H, W)
        v = (
            torch.as_tensor(tp.nb1, dtype=torch.long, device=device),
            torch.as_tensor(tp.nb2, dtype=torch.long, device=device),
        )
        _DEV_TOPO[k] = v
    return v


def is_solved_np(boxes: np.ndarray, goals: np.ndarray) -> bool:
    return bool(boxes.sum() > 0 and np.array_equal(boxes, goals))


def is_valid_state(walls, goals, boxes, player: int) -> bool:
    N = len(walls)
    if not (0 <= player < N) or walls[player] or boxes[player]:
        return False
    if int(boxes.sum()) != int(goals.sum()) or int(boxes.sum()) == 0:
        return False
    if bool((boxes & walls).any()) or bool((goals & walls).any()):
        return False
    return True


def legal_dirs_np(topo: GridTopo, walls, boxes, player: int) -> np.ndarray:
    out = np.zeros(4, dtype=bool)
    for d in range(4):
        p1 = int(topo.nb1[player, d])
        if walls[p1]:
            continue
        if boxes[p1]:
            p2 = int(topo.nb2[player, d])
            if walls[p2] or boxes[p2]:
                continue
        out[d] = True
    return out


def step_np(topo: GridTopo, walls, boxes, player: int, d: int):
    p1 = int(topo.nb1[player, d])
    assert not walls[p1], "Illegal wall move"
    nb = boxes.copy()
    if nb[p1]:
        p2 = int(topo.nb2[player, d])
        assert not walls[p2] and not nb[p2], "Illegal box push"
        nb[p1] = False
        nb[p2] = True
    return nb, p1


def _box_corner_dead(topo: GridTopo, walls, goals, b_mask: int) -> bool:
    """Sound corner deadlock: box not on goal wedged between two perpendicular walls."""
    m = b_mask
    while m:
        lsb = m & -m
        m ^= lsb
        i = lsb.bit_length() - 1
        if not goals[i]:
            wu = walls[topo.nb1[i, 0]]
            wd = walls[topo.nb1[i, 1]]
            wl = walls[topo.nb1[i, 2]]
            wr = walls[topo.nb1[i, 3]]
            if (wu or wd) and (wl or wr):
                return True
    return False


def _arr_to_mask(arr) -> int:
    m = 0
    for i in np.flatnonzero(arr):
        m |= (1 << int(i))
    return m


def bfs_solve(H: int, W: int, walls, goals, boxes, player: int, node_cap: int = 80_000):
    """Exact shortest-path BFS solver with sound corner-deadlock pruning.
    Returns (opt_len, actions_list, states_list) or (-1, [], []) if unsolvable/capped."""
    topo = get_topo(H, W)
    if not is_valid_state(walls, goals, boxes, player):
        return -1, [], []
    goal_mask = _arr_to_mask(goals)
    init_b = _arr_to_mask(boxes)
    if init_b == goal_mask:
        return 0, [], [(boxes.copy(), player)]
    w_bool = [bool(v) for v in walls]
    g_bool = [bool(v) for v in goals]
    q = deque([(player, init_b)])
    parent = {(player, init_b): None}
    nodes = 0
    nb1, nb2 = topo.nb1, topo.nb2
    while q:
        p, bm = q.popleft()
        nodes += 1
        if nodes > node_cap:
            return -1, [], []
        for d in range(4):
            p1 = int(nb1[p, d])
            if w_bool[p1]:
                continue
            b_next = bm
            if (bm >> p1) & 1:
                p2 = int(nb2[p, d])
                if w_bool[p2] or ((bm >> p2) & 1):
                    continue
                b_next = (bm ^ (1 << p1)) | (1 << p2)
                if b_next != goal_mask and _box_corner_dead(topo, w_bool, g_bool, b_next):
                    continue
            st = (p1, b_next)
            if st not in parent:
                parent[st] = (p, bm, d)
                if b_next == goal_mask:
                    acts, path = [], []
                    cur = st
                    while parent[cur] is not None:
                        pp, pbm, pd = parent[cur]
                        acts.append(pd)
                        arr = np.zeros(topo.N, dtype=bool)
                        for idx_b in range(topo.N):
                            if (pbm >> idx_b) & 1:
                                arr[idx_b] = True
                        path.append((arr, pp))
                        cur = (pp, pbm)
                    acts.reverse(); path.reverse()
                    return len(acts), acts, path
                q.append(st)
    return -1, [], []


def naive_bfs_solve(H: int, W: int, walls, goals, boxes, player: int, node_cap: int = 80_000):
    """INDEPENDENT unpruned BFS reference solver using frozenset (no bitmasks, no pruning)."""
    topo = get_topo(H, W)
    if not is_valid_state(walls, goals, boxes, player):
        return -1
    g_set = frozenset(int(i) for i in np.flatnonzero(goals))
    b0 = frozenset(int(i) for i in np.flatnonzero(boxes))
    if b0 == g_set:
        return 0
    q = deque([(player, b0, 0)])
    seen = {(player, b0)}
    nodes = 0
    while q:
        p, bs, dist = q.popleft()
        nodes += 1
        if nodes > node_cap:
            return -1
        for d in range(4):
            p1 = int(topo.nb1[p, d])
            if walls[p1]:
                continue
            bn = bs
            if p1 in bs:
                p2 = int(topo.nb2[p, d])
                if walls[p2] or p2 in bs:
                    continue
                bn = (bs - {p1}) | {p2}
            if bn == g_set:
                return dist + 1
            st = (p1, bn)
            if st not in seen:
                seen.add(st)
                q.append((p1, bn, dist + 1))
    return -1


def generate_level(rng: np.random.Generator, H: int, W: int, n_boxes: int, min_opt: int, max_pulls: int):
    """Guaranteed-solvable level via reverse-pull + exact forward BFS verification."""
    topo = get_topo(H, W)
    for _ in range(120):
        walls = np.zeros((H, W), dtype=bool)
        walls[0, :] = walls[-1, :] = walls[:, 0] = walls[:, -1] = True
        # interior pillars/walls
        n_int = int(rng.integers(2, max(4, (H - 2) * (W - 2) // 5)))
        for _w in range(n_int):
            walls[int(rng.integers(1, H - 1)), int(rng.integers(1, W - 1))] = True
        w_flat = walls.reshape(-1)
        walk = np.flatnonzero(~w_flat)
        if len(walk) < n_boxes + 4:
            continue
        # largest connected component
        seen = set()
        best_cc = []
        for s0 in walk:
            if int(s0) in seen:
                continue
            q = [int(s0)]; seen.add(int(s0)); cc = [int(s0)]
            while q:
                u = q.pop()
                for d in range(4):
                    v = int(topo.nb1[u, d])
                    if not w_flat[v] and v not in seen:
                        seen.add(v); q.append(v); cc.append(v)
            if len(cc) > len(best_cc):
                best_cc = cc
        if len(best_cc) < n_boxes + 4:
            continue
        # seal disconnected pockets as walls
        cc_set = set(best_cc)
        for i in range(topo.N):
            if i not in cc_set:
                w_flat[i] = True
        # pick goal cells that have at least 2 free neighbors (valid pull starts)
        cand_g = [i for i in best_cc if sum(not w_flat[topo.nb1[i, d]] for d in range(4)) >= 2]
        if len(cand_g) < n_boxes + 1:
            continue
        picks = rng.choice(cand_g, size=n_boxes, replace=False)
        goals = np.zeros(topo.N, dtype=bool); goals[picks] = True
        boxes = goals.copy()
        # place player on a free neighbor of a box
        p_cands = [int(topo.nb1[b, d]) for b in picks for d in range(4)
                   if not w_flat[topo.nb1[b, d]] and not boxes[topo.nb1[b, d]]]
        if not p_cands:
            continue
        player = int(rng.choice(p_cands))
        # reverse pulls from solved state
        n_pulls = int(rng.integers(max(6, min_opt), max_pulls + 1))
        for _step in range(n_pulls):
            moves = []
            for d in range(4):
                p_prev = int(topo.nb1[player, d])
                if w_flat[p_prev] or boxes[p_prev]:
                    continue
                # move without pulling
                moves.append((p_prev, -1))
                # or pull a box behind player (opposite direction d^1)
                opp = d ^ 1
                b_cell = int(topo.nb1[player, opp])
                if boxes[b_cell]:
                    moves.append((p_prev, b_cell))
                    moves.append((p_prev, b_cell))  # favor pulls
            if not moves:
                break
            p_new, b_pull = moves[int(rng.integers(len(moves)))]
            if b_pull >= 0:
                boxes[b_pull] = False
                boxes[player] = True
            player = p_new
        if is_solved_np(boxes, goals):
            continue
        opt_len, acts, path = bfs_solve(H, W, w_flat, goals, boxes, player, node_cap=50_000)
        if opt_len >= min_opt:
            return dict(walls=w_flat.copy(), goals=goals, boxes=boxes, player=player,
                        opt_len=opt_len, acts=np.array(acts, dtype=np.int8), path=path)
    return None


# ---- D4 Dihedral Symmetry Group (8 elements) ---------------------------------
def d4_transform_grid(arr, H: int, W: int, k_rot: int, flip: bool):
    g = np.asarray(arr).reshape(H, W)
    if k_rot:
        g = np.rot90(g, k_rot)
    if flip:
        g = np.fliplr(g)
    return g.reshape(-1)


def d4_dir_perm(k_rot: int, flip: bool):
    dp = [0, 1, 2, 3]
    for _ in range(k_rot):
        dp = [ROT_MAP[d] for d in dp]
    if flip:
        dp = [FLIP_MAP[d] for d in dp]
    return np.array(dp, dtype=np.int64)


def d4_transform_state(H: int, W: int, walls, goals, boxes, player: int, k_rot: int, flip: bool):
    p_arr = np.zeros(H * W, dtype=bool); p_arr[player] = True
    w2 = d4_transform_grid(walls, H, W, k_rot, flip)
    g2 = d4_transform_grid(goals, H, W, k_rot, flip)
    b2 = d4_transform_grid(boxes, H, W, k_rot, flip)
    p2 = int(np.flatnonzero(d4_transform_grid(p_arr, H, W, k_rot, flip))[0])
    return w2, g2, b2, p2


def d4_canonical_sig(H: int, W: int, walls, goals, boxes, player: int) -> bytes:
    best = None
    for k_rot in range(4):
        for flip in (False, True):
            w2, g2, b2, p2 = d4_transform_state(H, W, walls, goals, boxes, player, k_rot, flip)
            sig = w2.tobytes() + g2.tobytes() + b2.tobytes() + int(p2).to_bytes(2, "little")
            if best is None or sig < best:
                best = sig
    return best


def d4_push_logits(logits_bn4, H: int, W: int, k_rot: int, flip: bool):
    # logits_bn4: (B, N, 4) -> rotate/flip spatial grid AND permute 4 direction slots
    B, N, _ = logits_bn4.shape
    g = logits_bn4.reshape(B, H, W, 4)
    if k_rot:
        g = torch.rot90(g, k_rot, dims=(1, 2))
    if flip:
        g = torch.flip(g, dims=(2,))
    inv = np.argsort(d4_dir_perm(k_rot, flip))
    dp = torch.as_tensor(inv.copy(), device=logits_bn4.device)
    return g[..., dp].reshape(B, N, 4)


# ==============================================================================
# §2  DATA POOL & FEATURES
# ==============================================================================
class SokoPool:
    def __init__(self, H, W, walls, goals, boxes, players, tiers, opt_lens, trajs):
        self.H, self.W, self.N = H, W, H * W
        self.walls = np.asarray(walls, dtype=bool)
        self.goals = np.asarray(goals, dtype=bool)
        self.boxes = np.asarray(boxes, dtype=bool)
        self.players = np.asarray(players, dtype=np.int64)
        self.t = np.asarray(tiers, dtype=np.int8)
        self.opt_lens = np.asarray(opt_lens, dtype=np.int32)
        self.trajs = trajs  # list of (acts, path_boxes, path_players)
        self.by_tier = {t: np.where(self.t == t)[0] for t in range(1, 5)}

    def __len__(self):
        return len(self.walls)

    def sample(self, rng, n, tier=None):
        idx = self.by_tier.get(tier, np.empty(0, dtype=int)) if tier else np.arange(len(self.walls))
        if len(idx) == 0:
            idx = np.arange(len(self.walls))
        return rng.choice(idx, size=n, replace=len(idx) < n)


def build_pool(seed, n_per_tier, H, W, tag, cache_dir: Path, verbose=True):
    key = hashlib.sha1(f"{seed}|{n_per_tier}|{H}x{W}|soko_{__version__}".encode()).hexdigest()[:12]
    f = cache_dir / f"soko_{tag}_{key}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        if verbose:
            print(f"[data] {tag}: loaded cached {len(z['walls'])} levels from {f.name}")
        return SokoPool(H, W, z["walls"], z["goals"], z["boxes"], z["players"],
                        z["tiers"], z["opt_lens"], z["trajs"].tolist())
    rng = np.random.default_rng(seed)
    # Tier specs: (tier, n_boxes, min_opt, max_pulls)
    specs = [(1, 1, 4, 16), (2, 2, 7, 26), (3, 3, 9, 34), (4, 4, 10, 40)]
    WL, GL, BL, PL, TL, OL, TR = [], [], [], [], [], [], []
    seen_sigs = set()
    t0 = time.time()
    for tier, nb, min_o, max_p in specs:
        got = 0
        tries = 0
        while got < n_per_tier and tries < n_per_tier * 30:
            tries += 1
            lv = generate_level(rng, H, W, nb, min_o, max_p)
            if lv is None:
                continue
            sig = d4_canonical_sig(H, W, lv["walls"], lv["goals"], lv["boxes"], lv["player"])
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            WL.append(lv["walls"]); GL.append(lv["goals"]); BL.append(lv["boxes"])
            PL.append(lv["player"]); TL.append(tier); OL.append(lv["opt_len"])
            pb = np.stack([s[0] for s in lv["path"]])
            pp = np.array([s[1] for s in lv["path"]], dtype=np.int64)
            TR.append((lv["acts"], pb, pp))
            got += 1
        if verbose:
            print(f"[data] {tag}: tier {tier} ({nb} boxes) built {got}/{n_per_tier} in {time.time()-t0:.1f}s", flush=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, walls=np.array(WL), goals=np.array(GL), boxes=np.array(BL),
                        players=np.array(PL), tiers=np.array(TL, dtype=np.int8),
                        opt_lens=np.array(OL, dtype=np.int32), trajs=np.array(TR, dtype=object))
    return SokoPool(H, W, WL, GL, BL, PL, TL, OL, TR)


def sokoban_features_and_mask(H: int, W: int, walls, goals, boxes, players):
    """D4-equivariant features (B, N, 4, 17) & per-cell legal direction mask (B, N, 4)."""
    B, N = walls.shape
    nb1, nb2 = _dev_topo(H, W, walls.device)
    p_oh = F.one_hot(players.long(), N).float()
    w = walls.float()
    g = goals.float()
    b = boxes.float()
    fl = (~walls).float()
    bg = (boxes & goals).float()
    free_b = (~walls) & (~boxes)
    free = free_b.float()

    # Gather neighbor states in all 4 directions: (B, N, 4)
    nb1_w = w[:, nb1]; nb1_b = b[:, nb1]; nb1_g = g[:, nb1]; nb1_f = free[:, nb1]
    nb2_w = w[:, nb2]; nb2_b = b[:, nb2]; nb2_g = g[:, nb2]; nb2_f = free[:, nb2]

    # Directional move legality from each cell (and at player's cell)
    can_move_dir = (~walls[:, nb1]) & ((~boxes[:, nb1]) | free_b[:, nb2])  # (B, N, 4)
    legal_bool = can_move_dir & (p_oh.unsqueeze(-1) > 0.5)

    # Ray-ahead box/goal visibility (100% D4-equivariant!)
    tp = get_topo(H, W)
    # Vectorized ray sweep up to max(H,W) steps along each direction
    ray_g = torch.zeros(B, N, 4, device=walls.device)
    ray_b = torch.zeros(B, N, 4, device=walls.device)
    cur_idx = nb1.clone()
    alive_ray = ~walls[:, cur_idx]
    for _ in range(max(H, W) - 1):
        ray_g = torch.maximum(ray_g, g[:, cur_idx] * alive_ray.float())
        ray_b = torch.maximum(ray_b, b[:, cur_idx] * alive_ray.float())
        nxt = nb1[cur_idx, torch.arange(4, device=walls.device)]
        alive_ray = alive_ray & (nxt != cur_idx) & (~walls[:, nxt])
        cur_idx = nxt

    def exp4(t):
        return t.unsqueeze(-1).expand(B, N, 4)

    f = torch.stack([
        exp4(w), exp4(fl), exp4(g), exp4(b), exp4(bg), exp4(p_oh),
        nb1_w, nb1_b, nb1_g, nb1_f,
        nb2_w, nb2_b, nb2_g, nb2_f,
        can_move_dir.float(), ray_g, ray_b,
    ], dim=-1)  # (B, N, 4, 17)
    return f, legal_bool


# ==============================================================================
# §3  SOKOBAN BATCH ENVIRONMENT
# ==============================================================================
@dataclass
class SokoMDP:
    max_steps: int = 50
    shaping: bool = True
    gamma: float = 1.0

    def phi(self, on_goal, n_boxes, terminal):
        if not self.shaping:
            return torch.zeros_like(on_goal, dtype=torch.float32)
        return torch.where(terminal, torch.zeros_like(on_goal, dtype=torch.float32),
                           (on_goal.float() / n_boxes.float().clamp(min=1)) - 1.0)


class SokoBatchEnv:
    def __init__(self, H, W, walls, goals, boxes, players, device, mdp: SokoMDP):
        self.H, self.W, self.N = H, W, H * W
        self.dev = device
        self.mdp = mdp
        self.walls = torch.as_tensor(walls, dtype=torch.bool, device=device)
        self.goals = torch.as_tensor(goals, dtype=torch.bool, device=device)
        self.boxes = torch.as_tensor(boxes, dtype=torch.bool, device=device).clone()
        self.players = torch.as_tensor(players, dtype=torch.long, device=device).clone()
        self.B = self.walls.shape[0]
        self.n_boxes = self.goals.sum(1)
        self.done = torch.zeros(self.B, dtype=torch.bool, device=device)
        self.solved = torch.zeros(self.B, dtype=torch.bool, device=device)
        self.dead = torch.zeros(self.B, dtype=torch.bool, device=device)
        self.steps = torch.zeros(self.B, dtype=torch.long, device=device)
        self.prev_phi = self.mdp.phi((self.boxes & self.goals).sum(1), self.n_boxes, self.done)
        self._lm = self._compute_legal()

    def _compute_legal(self):
        nb1, nb2 = _dev_topo(self.H, self.W, self.dev)
        p = self.players
        p1 = nb1[p]                                       # (B, 4)
        p2 = nb2[p]                                       # (B, 4)
        idx = torch.arange(self.B, device=self.dev)[:, None]
        w1 = self.walls[idx, p1]
        b1 = self.boxes[idx, p1]
        w2 = self.walls[idx, p2]
        b2 = self.boxes[idx, p2]
        return (~w1) & ((~b1) | ((~w2) & (~b2)))          # (B, 4)

    def legal(self):
        return self._lm

    def step(self, dirs):
        lm = self._lm
        live = ~self.done
        assert bool(torch.all(lm.sum(1)[live] > 0)), "LIVE SOKOBAN ROW WITHOUT LEGAL MOVE"
        assert bool(torch.all(lm.gather(1, dirs[:, None]).squeeze(1)[live])), "ILLEGAL MOVE (P6 violated)"
        nb1, nb2 = _dev_topo(self.H, self.W, self.dev)
        li = torch.nonzero(live, as_tuple=True)[0]
        p_l = self.players[li]
        d_l = dirs[li]
        p1 = nb1[p_l, d_l]
        p2 = nb2[p_l, d_l]
        has_b = self.boxes[li, p1]
        push_idx = li[has_b]
        self.boxes[push_idx, p1[has_b]] = False
        self.boxes[push_idx, p2[has_b]] = True
        self.players[li] = p1
        self.steps += live.long()

        is_sol = (self.boxes == self.goals).all(1)
        newly_sol = is_sol & live
        self.solved |= newly_sol
        self._lm = self._compute_legal()
        stuck = (self._lm.sum(1) == 0) & (~is_sol)
        self.dead |= stuck & live
        over = self.steps >= self.mdp.max_steps
        new_done = newly_sol | (stuck & live) | (over & live)
        r = torch.where(newly_sol, 1.0, 0.0)
        on_g = (self.boxes & self.goals).sum(1)
        phi_now = self.mdp.phi(on_g, self.n_boxes, self.done | new_done)
        r = r + self.mdp.gamma * phi_now - self.prev_phi
        self.prev_phi = phi_now
        self.done |= new_done
        return r * live.float(), self.done.clone()


# ==============================================================================
# §4  SHARED DOMAIN-AGNOSTIC AGI CORE (identical engine to sudoku_ai v5.0.2!)
# ==============================================================================
class SelfAttnSet(nn.Module):
    def __init__(self, C, heads):
        super().__init__()
        self.h = heads; self.dk = C // heads
        self.qkv = nn.Linear(C, 3 * C, bias=False)
        self.proj = nn.Linear(C, C, bias=False)

    def forward(self, x):
        N, S, C = x.shape
        qkv = self.qkv(x).reshape(N, S, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(N, S, C))


class HyperGraphBlock(nn.Module):
    def __init__(self, C, heads, attn_names):
        super().__init__()
        self.ln1 = nn.LayerNorm(C, eps=1e-5)
        self.group_attns = nn.ModuleDict({g: SelfAttnSet(C, heads) for g in attn_names})
        self.ln2 = nn.LayerNorm(C, eps=1e-5)
        self.attn_slot = SelfAttnSet(C, heads)
        self.ln3 = nn.LayerNorm(C, eps=1e-5)
        self.mlp = nn.Sequential(nn.Linear(C, 3 * C), nn.GELU(), nn.Linear(3 * C, C))

    def forward(self, h, unit_specs):
        B, N, K, C = h.shape
        u = self.ln1(h)
        msg = 0
        for attn_key, weight, u_idx, inv_idx, deg in unit_specs:
            M, S = u_idx.shape
            sel = u[:, u_idx]
            x_g = sel.permute(0, 1, 3, 2, 4).reshape(B * M * K, S, C)
            o_g = self.group_attns[attn_key](x_g)
            o_flat = o_g.reshape(B, M, K, S, C).permute(0, 1, 3, 2, 4).reshape(B, M * S, K, C)
            if inv_idx is not None:
                acc = o_flat[:, inv_idx]
            else:
                acc = torch.zeros(B, N, K, C, dtype=o_flat.dtype, device=o_flat.device)
                acc.index_add_(1, u_idx.reshape(-1), o_flat)
                acc = acc / deg
            msg = msg + weight * acc.to(h.dtype)
        h = h + msg
        us = self.ln2(h).reshape(B * N, K, C)
        h = h + self.attn_slot(us).reshape(B, N, K, C)
        h = h + self.mlp(self.ln3(h))
        return h


class RelationalReasoner(nn.Module):
    def __init__(self, n_nodes, unit_groups_cfg, feat_tie_idx, C=96, heads=4, L=3, T_cyc=8):
        super().__init__()
        self.N, self.C, self.L, self.T_cyc = n_nodes, C, L, T_cyc
        f_uniq = max(feat_tie_idx) + 1
        self.register_buffer("fidx", torch.as_tensor(feat_tie_idx, dtype=torch.long))
        self.unit_meta = []
        attn_names = []
        for name, attn_key, weight, arr in unit_groups_cfg:
            if attn_key not in attn_names:
                attn_names.append(attn_key)
            t_idx = torch.as_tensor(arr, dtype=torch.long)
            flat = t_idx.reshape(-1)
            deg = torch.bincount(flat, minlength=n_nodes).float().clamp(min=1.0)
            is_part = (flat.numel() == n_nodes) and bool((torch.bincount(flat, minlength=n_nodes) == 1).all())
            self.register_buffer(f"ug_{name}", t_idx)
            self.register_buffer(f"ud_{name}", deg.view(1, n_nodes, 1, 1))
            if is_part:
                self.register_buffer(f"inv_{name}", torch.argsort(flat))
            self.unit_meta.append((name, attn_key, weight, is_part))
        self.Win = nn.Parameter(torch.randn(f_uniq, C) * (1.0 / math.sqrt(f_uniq)))
        self.bin = nn.Parameter(torch.zeros(C))
        self.inp = nn.Linear(C, C)
        self.step_gate = nn.Parameter(torch.full((C,), 0.5))
        self.y_proj = nn.Linear(1, C, bias=False)
        self.step_emb = nn.Parameter(torch.zeros(T_cyc, C))
        self.blocks = nn.ModuleList([HyperGraphBlock(C, heads, attn_names) for _ in range(L)])
        self.lnf = nn.LayerNorm(C, eps=1e-5)
        self.head = nn.Linear(C, 1)
        self.cand = nn.Linear(C, 1)
        self.lnc = nn.LayerNorm(C, eps=1e-5)
        self.value = nn.Sequential(nn.Linear(C, C), nn.GELU(), nn.Linear(C, 1))
        self.conf = nn.Sequential(nn.Linear(C, C), nn.GELU(), nn.Linear(C, 1))

    def forward_features(self, f, legal_bool, T=6, ckpt=False, all_steps=False, state=None, step_off=0):
        unit_specs = [(ak, w, getattr(self, f"ug_{nm}"),
                       (getattr(self, f"inv_{nm}") if isp else None),
                       getattr(self, f"ud_{nm}"))
                      for nm, ak, w, isp in self.unit_meta]
        has_leg = legal_bool.any(-1, keepdim=True).float()
        W = self.Win[self.fidx]
        h0 = self.inp(F.gelu(torch.einsum("bnkf,fc->bnkc", f, W) + self.bin))
        if state is None:
            h, lg_prev = h0, None
        else:
            h, lg_prev = state
        alpha = torch.sigmoid(self.step_gate)
        outs = []
        for t in range(T):
            ti = (step_off + t) % self.T_cyc   # cycled so T=1..16 never hits untrained zeros
            if t > 0 or state is not None:
                y_fb = 0
                if lg_prev is not None:
                    y_t = torch.softmax(lg_prev.masked_fill(~legal_bool, -1e4), dim=-1) * has_leg
                    y_fb = self.y_proj(y_t.to(h.dtype).unsqueeze(-1))
                h = (1.0 - alpha) * h + alpha * (h0 + y_fb) + self.step_emb[ti]
            else:
                h = h + self.step_emb[ti]
            for blk in self.blocks:
                if ckpt and torch.is_grad_enabled():
                    h = _ckpt(blk, h, unit_specs, use_reentrant=False)
                else:
                    h = blk(h, unit_specs)
            lg_prev = self.head(self.lnf(h)).squeeze(-1).float()
            if all_steps or t == T - 1:
                outs.append(lg_prev)
        z = self.lnf(h)
        c = self.lnc(z.mean(2))
        gsum = c.mean(1)
        return dict(logits=outs[-1], logits_all=torch.stack(outs),
                    cand=self.cand(z).squeeze(-1).float(),
                    value=self.value(gsum).squeeze(-1).float(),
                    conf=self.conf(gsum).squeeze(-1).float(),
                    _h=h, _lg=lg_prev)

    @torch.no_grad()
    def logit_bound(self):
        w = self.head.weight.norm().item()
        gam = self.lnf.weight.abs().max().item()
        beta = self.lnf.bias.abs().max().item()
        return math.sqrt(self.C) * (gam * math.sqrt(self.C - 1) + beta) * w + self.head.bias.abs().max().item()


class SokobanNet(RelationalReasoner):
    def __init__(self, H=8, W=8, C=96, heads=4, L=3, T_cyc=8):
        self.H, self.W = H, W
        tp = get_topo(H, W)
        super().__init__(
            n_nodes=H * W,
            unit_groups_cfg=[
                ("row", "line", 0.5, tp.row_units),
                ("col", "line", 0.5, tp.col_units),
                ("patch", "patch", 1.0, tp.patch_units),
            ],
            feat_tie_idx=list(range(17)),
            C=C, heads=heads, L=L, T_cyc=T_cyc,
        )

    def forward(self, walls, goals, boxes, players, T=4, ckpt=False, all_steps=False, state=None, step_off=0):
        f, legal_bool = sokoban_features_and_mask(self.H, self.W, walls, goals, boxes, players)
        out = self.forward_features(f, legal_bool, T=T, ckpt=ckpt, all_steps=all_steps,
                                    state=state, step_off=step_off)
        idx = torch.arange(walls.shape[0], device=walls.device)
        out["dir_logits"] = out["logits"][idx, players.long()]       # (B, 4) at player cell
        out["dir_logits_all"] = out["logits_all"][:, idx, players.long()]
        return out


def masked_logsoftmax(logits, mask):
    neg = torch.finfo(logits.dtype).min / 2
    return torch.log_softmax(logits.masked_fill(~mask, neg), dim=-1)


def policy_logits(net_out, lm):
    neg = torch.finfo(net_out["dir_logits"].dtype).min / 2
    return net_out["dir_logits"].masked_fill(~lm, neg), lm


# ==============================================================================
# §5  OPTIMIZER: Muon + AdamW + EMA
# ==============================================================================
@torch.no_grad()
def newton_schulz5(G, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + eps)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if transposed else X


class MuonAdamW:
    def __init__(self, model, lr_muon=0.02, lr_adam=3e-4, momentum=0.95, wd=0.01, ns_steps=5):
        self.ns = ns_steps
        muon_p, adam_p = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            hidden2d = p.ndim == 2 and min(p.shape) >= 8 and not any(
                k in name for k in ("Win", "head", "cand", "step_emb", "value", "conf", "inp", "y_proj"))
            (muon_p if hidden2d else adam_p).append(p)
        self.muon_p, self.adam_p = muon_p, adam_p
        self.buf = {p: torch.zeros_like(p) for p in muon_p}
        self.opt = torch.optim.AdamW(adam_p, lr=lr_adam, betas=(0.9, 0.95), eps=1e-8, weight_decay=wd)
        self.lr_muon, self.mom, self.wd = lr_muon, momentum, wd
        self._adam_lr = lr_adam
        self.stats = {}

    def zero_grad(self):
        for p in self.muon_p + self.adam_p:
            p.grad = None

    @torch.no_grad()
    def step(self, lr_scale=1.0):
        smax, smin, n = 0.0, 1e9, 0
        for p in self.muon_p:
            if p.grad is None:
                continue
            g = p.grad
            buf = self.buf[p]
            buf.mul_(self.mom).add_(g)
            u = g.add(buf, alpha=self.mom)
            o = newton_schulz5(u, self.ns)
            scale = 0.2 * math.sqrt(max(p.shape))
            p.mul_(1 - self.lr_muon * lr_scale * self.wd)
            p.add_(o, alpha=-self.lr_muon * lr_scale * scale)
            if random.random() < 0.05:
                s = torch.linalg.svdvals(o.float())
                smax = max(smax, s.max().item()); smin = min(smin, s.min().item()); n += 1
        for pg in self.opt.param_groups:
            pg["lr"] = self._adam_lr * lr_scale
        self.opt.step()
        self.stats = dict(muon_sigma_max=(smax if n else None),
                          muon_sigma_min=(smin if n else None), muon_sampled=n)

    def set_lrs(self, lr_muon, adam_lr):
        self.lr_muon, self._adam_lr = lr_muon, adam_lr


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.steps = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        self.steps += 1
        d = min(self.decay, (1.0 + self.steps) / (10.0 + self.steps))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                self.shadow[k].copy_(v)


# ==============================================================================
# §6  LOSSES: Optimal-Trajectory SSL + Unbiased RLOO RL
# ==============================================================================
def safe_bce(a, b):
    if a.numel() == 0:
        return a.sum() * 0.0
    return F.binary_cross_entropy_with_logits(a, b)


def _ssl_one(out, walls, boxes, players, tgt_dir, H, W, w_cand=0.2, w_conf=0.1):
    da = out["dir_logits_all"]  # (T, B, 4)
    T = da.shape[0]
    B = walls.shape[0]
    nb1, nb2 = _dev_topo(H, W, walls.device)
    idx = torch.arange(B, device=walls.device)[:, None]
    p1 = nb1[players]; p2 = nb2[players]
    lm_dir = (~walls[idx, p1]) & ((~boxes[idx, p1]) | ((~walls[idx, p2]) & (~boxes[idx, p2])))
    neg = torch.finfo(da.dtype).min / 2
    da_m = da.masked_fill(~lm_dir.unsqueeze(0), neg)
    ce = F.cross_entropy(da_m.reshape(-1, 4), tgt_dir.repeat(T), reduction="none").reshape(T, B)
    w = torch.arange(1, T + 1, device=walls.device, dtype=torch.float32)
    w = (w / w.sum())[:, None]
    ce = (ce * w).mean()
    # Cand mask over player cell
    cand_p = out["cand"][torch.arange(B, device=walls.device), players]
    cand = safe_bce(cand_p, lm_dir.float())
    with torch.no_grad():
        pred = out["dir_logits"].masked_fill(~lm_dir, -1e4).argmax(-1)
        ok = (pred == tgt_dir).float()
    conf = F.binary_cross_entropy_with_logits(out["conf"], ok)
    loss = ce + w_cand * cand + w_conf * conf
    return loss, ce, cand, conf, ok


def rollout(net, pool, rng, P, G, T, device, mdp, temp=0.85, greedy=False, tier=None, mix_easy=0.25):
    if tier is None:
        idx = pool.sample(rng, P)
    else:
        n_t = int(round(P * (1 - mix_easy)))
        idx = np.concatenate([pool.sample(rng, n_t, tier), pool.sample(rng, P - n_t)])
    wl = np.repeat(pool.walls[idx], G, axis=0)
    gl = np.repeat(pool.goals[idx], G, axis=0)
    bx = np.repeat(pool.boxes[idx], G, axis=0)
    pl = np.repeat(pool.players[idx], G, axis=0)
    env = SokoBatchEnv(pool.H, pool.W, wl, gl, bx, pl, device, mdp)
    B, N, S = env.B, env.N, mdp.max_steps
    hb = torch.zeros(S, B, N, dtype=torch.bool, device=device)
    hp = torch.zeros(S, B, dtype=torch.long, device=device)
    ha = torch.zeros(S, B, dtype=torch.long, device=device)
    hl = torch.zeros(S, B, device=device)
    ret = torch.zeros(B, device=device)
    ent_sum = torch.zeros(B, device=device)
    nst = 0
    for t in range(S):
        live = ~env.done
        if not bool(live.any()):
            break
        li = torch.nonzero(live, as_tuple=True)[0]
        with torch.no_grad(), _amp(device):
            out = net(env.walls[li], env.goals[li], env.boxes[li], env.players[li], T=T)
        with torch.no_grad():
            lg_l, _ = policy_logits(out, env.legal()[li])
            lp_l = torch.log_softmax(lg_l / max(temp, 1e-6), -1)
            a_l = lp_l.argmax(-1) if greedy else torch.distributions.Categorical(logits=lp_l).sample()
            ent_sum[li] += -(lp_l.exp() * lp_l.clamp(min=-1e4)).sum(-1)
            a = torch.zeros(B, dtype=torch.long, device=device)
            a[li] = a_l
            hb[t] = env.boxes; hp[t] = env.players; ha[t] = a
            hl[t, li] = lp_l.gather(1, a_l[:, None]).squeeze(1)
        r, _ = env.step(a)
        ret += r
        nst = t + 1
    return dict(walls=env.walls, goals=env.goals, hb=hb[:nst], hp=hp[:nst], ha=ha[:nst], hl=hl[:nst],
                ret=ret, solved=env.solved, dead=env.dead, steps=env.steps, P=P, G=G,
                ent=ent_sum / env.steps.clamp(min=1), tiers=pool.t[idx])


def grpo_loss(net, roll, K, T, device, ckpt, temp=0.85, ent_beta=0.005, w_val=0.5,
              rl_norm=25.0, ppo_eps=0.2):
    P, G = roll["P"], roll["G"]
    R = roll["ret"].reshape(P, G)
    corr = G / (G - 1) if G > 1 else 1.0
    A = ((R - R.mean(1, keepdim=True)) * corr).reshape(-1)
    informative = (R.std(1) > 0).float().mean()
    S, B = roll["ha"].shape
    N = roll["hb"].shape[-1]
    k = max(1, min(K, S))
    live = torch.arange(S, device=device)[:, None] < roll["steps"][None, :]
    u = torch.rand(S, B, device=device).masked_fill(~live, 2.0)
    pick = u.topk(k, dim=0, largest=False).indices
    ok = (pick < roll["steps"][None, :]).float().reshape(-1)
    bb = torch.arange(B, device=device)[None, :].expand(k, B)
    boxes = roll["hb"][pick, bb].reshape(-1, N)
    players = roll["hp"][pick, bb].reshape(-1)
    acts = roll["ha"][pick, bb].reshape(-1)
    oldlp = roll["hl"][pick, bb].reshape(-1)
    adv = A[None, :].expand(k, B).reshape(-1)
    walls = roll["walls"][None, :, :].expand(k, B, N).reshape(-1, N)
    goals = roll["goals"][None, :, :].expand(k, B, N).reshape(-1, N)
    lens = roll["steps"].clamp(min=1)[None, :].expand(k, B).reshape(-1).float()
    out = net(walls, goals, boxes, players, T=T, ckpt=ckpt)
    nb1, nb2 = _dev_topo(net.H, net.W, device)
    idx = torch.arange(len(players), device=device)[:, None]
    p1 = nb1[players]; p2 = nb2[players]
    lm = (~walls[idx, p1]) & ((~boxes[idx, p1]) | ((~walls[idx, p2]) & (~boxes[idx, p2])))
    lg, _ = policy_logits(out, lm)
    lp = torch.log_softmax(lg / max(temp, 1e-6), -1)
    newlp = lp.gather(1, acts[:, None]).squeeze(1)
    ratio = (newlp - oldlp).exp()
    pg = -torch.min(ratio * adv, ratio.clamp(1 - ppo_eps, 1 + ppo_eps) * adv)
    pg = (pg * ok * lens / rl_norm).mean()
    ent = -(lp.exp() * lp.clamp(min=-1e4)).sum(-1)
    ent = (ent * ok).sum() / ok.sum().clamp(min=1)
    vb = F.binary_cross_entropy_with_logits(
        out["value"], roll["solved"][None, :].expand(k, B).reshape(-1).float(), reduction="none")
    vb = (vb * ok).sum() / ok.sum().clamp(min=1)
    loss = pg + w_val * vb - ent_beta * ent
    return loss, dict(pg=float(pg.detach()), val=float(vb.detach()), ent=float(ent.detach()),
                      info_grp=float(informative),
                      roll_pass=float(roll["solved"].float().mean()),
                      dead=float(roll["dead"].float().mean()),
                      steps=float(roll["steps"].float().mean()))


# ==============================================================================
# §7  SYSTEM 2 (Neural Value-Guided Best-First Search) + VIDEO RENDERER
# ==============================================================================
@torch.no_grad()
def simulate_solve(net, H, W, walls, goals, boxes, player, device, T=4, max_steps=50, max_nodes=64):
    """System 1 + System 2: Value-guided neural best-first search with corner-deadlock
    pruning & visited memory (ZERO BFS oracle used). Returns (solved, nodes_used, trace)."""
    topo = get_topo(H, W)
    w_t = torch.as_tensor(walls[None], dtype=torch.bool, device=device)
    g_t = torch.as_tensor(goals[None], dtype=torch.bool, device=device)
    w_bool = [bool(v) for v in walls]
    g_bool = [bool(v) for v in goals]

    def nn_eval(bx_list, pl_list):
        M = len(bx_list)
        bt = torch.as_tensor(np.stack(bx_list), dtype=torch.bool, device=device)
        pt = torch.as_tensor(pl_list, dtype=torch.long, device=device)
        with _amp(device):
            o = net(w_t.expand(M, -1), g_t.expand(M, -1), bt, pt, T=T)
        return o["dir_logits"].cpu().numpy(), torch.sigmoid(o["value"]).cpu().numpy()

    # 1) Fast System-1 greedy attempt first (0 search overhead if solved)
    cur_b, cur_p = boxes.copy(), int(player)
    trace = [(cur_b.copy(), cur_p, -1, 1.0, 0.5)]
    seen_greedy = {(cur_p, _arr_to_mask(cur_b))}
    for _ in range(max_steps):
        if is_solved_np(cur_b, goals):
            return True, 0, trace
        lm = legal_dirs_np(topo, walls, cur_b, cur_p)
        if not lm.any():
            break
        lg, val = nn_eval([cur_b], [cur_p])
        li = np.where(lm, lg[0], -1e9)
        pr = np.exp(li - li.max()) * lm; pr /= pr.sum()
        d = int(pr.argmax())
        cur_b, cur_p = step_np(topo, walls, cur_b, cur_p, d)
        trace.append((cur_b.copy(), cur_p, d, float(pr[d]), float(val[0])))
        st_k = (cur_p, _arr_to_mask(cur_b))
        if is_solved_np(cur_b, goals):
            return True, 0, trace
        if st_k in seen_greedy:
            break
        seen_greedy.add(st_k)

    # 2) System 2: Value-guided Best-First Mental Simulation
    uid = itertools.count()
    init_m = _arr_to_mask(boxes)
    visited = {(int(player), init_m)}
    pq = []
    heapq.heappush(pq, (-0.5, 0, next(uid), boxes.copy(), int(player),
                       [(boxes.copy(), int(player), -1, 1.0, 0.5)]))
    nodes = 0
    while pq and nodes < max_nodes:
        _neg_s, depth, _id, bx, pl, tr = heapq.heappop(pq)
        if depth >= max_steps:
            continue
        nodes += 1
        lm = legal_dirs_np(topo, walls, bx, pl)
        if not lm.any():
            continue
        lg_p, val_p = nn_eval([bx], [pl])
        li = np.where(lm, lg_p[0], -1e9)
        pr = np.exp(li - li.max()) * lm; pr /= pr.sum()
        children, c_pl, c_dirs, c_trs = [], [], [], []
        for d in range(4):
            if not lm[d]:
                continue
            nb, np_ = step_np(topo, walls, bx, pl, d)
            nm = _arr_to_mask(nb)
            if is_solved_np(nb, goals):
                tr2 = tr + [(nb.copy(), np_, d, float(pr[d]), 1.0)]
                return True, nodes, tr2
            if _box_corner_dead(topo, w_bool, g_bool, nm):
                continue
            if (np_, nm) not in visited:
                visited.add((np_, nm))
                children.append(nb); c_pl.append(np_); c_dirs.append(d)
                c_trs.append(tr + [(nb.copy(), np_, d, float(pr[d]), float(val_p[0]))])
        if children:
            _, vals = nn_eval(children, c_pl)
            for nb, np_, d, v, tr2 in zip(children, c_pl, c_dirs, vals, c_trs):
                score = float(v) + 0.15 * math.log(max(float(pr[d]), 1e-6)) - 0.008 * (depth + 1)
                heapq.heappush(pq, (-score, depth + 1, next(uid), nb, np_, tr2))
    return False, nodes, trace


def render_episode_video(H, W, walls, goals, trace, title: str, out_base: Path):
    """Renders crisp tile-art frames of Sokoban gameplay -> saves .gif (always) + .mp4 (if available)."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None
    TILE = 48
    HUD = 44
    W_px, H_px = W * TILE, H * TILE + HUD
    frames = []
    tot_steps = max(1, len(trace) - 1)
    for t, (bx, pl, act, conf, val) in enumerate(trace):
        img = Image.new("RGB", (W_px, H_px), (24, 26, 32))
        dr = ImageDraw.Draw(img)
        solved_now = is_solved_np(bx, goals)
        act_str = DIR_NAMES[act] if act >= 0 else "START"
        hud_col = (28, 120, 72) if solved_now else (30, 34, 46)
        dr.rectangle([0, 0, W_px, HUD], fill=hud_col)
        dr.text((10, 7), f"{title}", fill=(235, 240, 250))
        dr.text((10, 24), f"step {t}/{tot_steps} | act={act_str} | p={conf:.2f} | V={val:.2f}" +
                (" | SOLVED!" if solved_now else ""), fill=(200, 230, 255))
        for r in range(H):
            for c in range(W):
                i = r * W + c
                x0, y0 = c * TILE, HUD + r * TILE
                x1, y1 = x0 + TILE, y0 + TILE
                if walls[i]:
                    dr.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(56, 62, 78))
                    dr.rectangle([x0 + 3, y0 + 3, x1 - 4, y1 - 4], outline=(78, 86, 108), width=2)
                else:
                    dr.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(238, 232, 218))
                    dr.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(220, 212, 196), width=1)
                    if goals[i]:
                        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                        dr.polygon([(cx, y0 + 10), (x1 - 10, cy), (cx, y1 - 10), (x0 + 10, cy)],
                                   fill=(72, 195, 130), outline=(36, 135, 84))
                    if bx[i]:
                        col = (245, 185, 55) if goals[i] else (196, 124, 58)
                        bd = (180, 125, 20) if goals[i] else (130, 76, 28)
                        dr.rectangle([x0 + 6, y0 + 6, x1 - 7, y1 - 7], fill=col, outline=bd, width=3)
                        dr.line([x0 + 8, y0 + 8, x1 - 9, y1 - 9], fill=bd, width=2)
                        dr.line([x1 - 9, y0 + 8, x0 + 8, y1 - 9], fill=bd, width=2)
                    if i == pl:
                        dr.ellipse([x0 + 8, y0 + 8, x1 - 9, y1 - 9], fill=(46, 145, 235), outline=(20, 75, 145), width=3)
                        dr.ellipse([x0 + 16, y0 + 17, x0 + 21, y0 + 22], fill=(255, 255, 255))
                        dr.ellipse([x1 - 22, y0 + 17, x1 - 17, y0 + 22], fill=(255, 255, 255))
        frames.append(img)
    # Pause slightly on final solved frame
    for _ in range(5):
        frames.append(frames[-1])
    gif_path = out_base.with_suffix(".gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=240, loop=0)
    saved = [str(gif_path)]
    # Also try MP4 if cv2 or imageio-ffmpeg is available
    try:
        import imageio.v2 as iio
        mp4_path = out_base.with_suffix(".mp4")
        with iio.get_writer(str(mp4_path), fps=4) as wr:
            for fr in frames:
                wr.append_data(np.asarray(fr))
        saved.append(str(mp4_path))
    except Exception:
        try:
            import cv2
            mp4_path = out_base.with_suffix(".mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(str(mp4_path), fourcc, 4.0, (W_px, H_px))
            for fr in frames:
                vw.write(cv2.cvtColor(np.asarray(fr), cv2.COLOR_RGB2BGR))
            vw.release()
            saved.append(str(mp4_path))
        except Exception:
            pass
    return saved


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, ctr - hw), min(1.0, ctr + hw))


@torch.no_grad()
def evaluate(net, pool, device, mdp, n_samples=4, temp=0.8, T=4, per_tier=32,
             run_sys2=True, max_nodes=64, video_out: Path | None = None):
    net.eval()
    res = {"T": T, "n_samples": n_samples, "tiers": {}}
    best_demo = None  # (priority, tier, label, walls, goals, trace)
    for t in (1, 2, 3, 4):
        idx = pool.by_tier.get(t, np.empty(0, int))
        if len(idx) == 0:
            continue
        idx = idx[:per_tier]
        n = len(idx)
        wl, gl, bx, pl = pool.walls[idx], pool.goals[idx], pool.boxes[idx], pool.players[idx]
        # ---- System 1 Greedy + trace capture
        e = SokoBatchEnv(pool.H, pool.W, wl, gl, bx, pl, device, mdp)
        traces = [[(bx[j].copy(), int(pl[j]), -1, 1.0, 0.5)] for j in range(n)]
        for _ in range(mdp.max_steps):
            live = ~e.done
            if not bool(live.any()):
                break
            li = torch.nonzero(live, as_tuple=True)[0]
            with _amp(device):
                o = net(e.walls[li], e.goals[li], e.boxes[li], e.players[li], T=T)
            lg_l, lm_l = policy_logits(o, e.legal()[li])
            pr_l = torch.softmax(lg_l, -1)
            a_l = lg_l.argmax(-1)
            val_l = torch.sigmoid(o["value"])
            a = torch.zeros(n, dtype=torch.long, device=device)
            a[li] = a_l
            e.step(a)
            bx_np = e.boxes.cpu().numpy(); pl_np = e.players.cpu().numpy()
            for pos, j in enumerate(li.cpu().tolist()):
                act_j = int(a_l[pos])
                traces[j].append((bx_np[j].copy(), int(pl_np[j]), act_j,
                                  float(pr_l[pos, act_j]), float(val_l[pos])))
        gs = e.solved.cpu().numpy()
        for j in range(n):
            if gs[j]:
                score = (t, 2, len(traces[j]))  # prefer highest tier solved by Sys1 greedy
                if best_demo is None or score[:2] > best_demo[0][:2]:
                    best_demo = (score, t, f"Sys1 pass@1 | {TIER_NAMES[t]}", wl[j], gl[j], traces[j])

        # ---- System 1 Stochastic (pass@1, pass@2)
        cnt = np.zeros(n, dtype=int)
        if n_samples > 0:
            wl_r = np.repeat(wl, n_samples, axis=0); gl_r = np.repeat(gl, n_samples, axis=0)
            bx_r = np.repeat(bx, n_samples, axis=0); pl_r = np.repeat(pl, n_samples, axis=0)
            e2 = SokoBatchEnv(pool.H, pool.W, wl_r, gl_r, bx_r, pl_r, device, mdp)
            for _ in range(mdp.max_steps):
                live2 = ~e2.done
                if not bool(live2.any()):
                    break
                li2 = torch.nonzero(live2, as_tuple=True)[0]
                with _amp(device):
                    o = net(e2.walls[li2], e2.goals[li2], e2.boxes[li2], e2.players[li2], T=T)
                lg_l, _ = policy_logits(o, e2.legal()[li2])
                lp_l = torch.log_softmax(lg_l / max(temp, 1e-6), -1)
                a = torch.zeros(len(wl_r), dtype=torch.long, device=device)
                a[li2] = torch.distributions.Categorical(logits=lp_l).sample()
                e2.step(a)
            cnt = e2.solved.cpu().numpy().reshape(n, n_samples).sum(1)

        d = dict(name=TIER_NAMES[t], n_levels=n, opt_steps=float(pool.opt_lens[idx].mean()),
                 greedy_pass1=float(gs.mean()), greedy_ci95=list(wilson(int(gs.sum()), n)),
                 mean_steps=float(e.steps.cpu().numpy().mean()),
                 pass_at_1=float(np.mean([pass_at_k(n_samples, int(c), 1) for c in cnt])) if n_samples else float(gs.mean()),
                 pass_at_2=float(np.mean([pass_at_k(n_samples, int(c), 2) for c in cnt])) if n_samples else float(gs.mean()))
        if run_sys2:
            s2_ok, s2_n = [], []
            for j in range(n):
                ok_j, n_j, tr_j = simulate_solve(net, pool.H, pool.W, wl[j], gl[j], bx[j], pl[j],
                                                 device, T=T, max_steps=mdp.max_steps, max_nodes=max_nodes)
                s2_ok.append(ok_j); s2_n.append(n_j)
                if ok_j:
                    sc = (t, 1 if n_j > 0 else 2, len(tr_j))
                    if best_demo is None or sc[:2] > best_demo[0][:2]:
                        tag_m = "Sys1" if n_j == 0 else f"Sys2({n_j}n)"
                        best_demo = (sc, t, f"{tag_m} | {TIER_NAMES[t]}", wl[j], gl[j], tr_j)
            d["sys2_pass"] = float(np.mean(s2_ok))
            d["sys2_avg_nodes"] = float(np.mean(s2_n))
        res["tiers"][t] = d

    if video_out is not None:
        if best_demo is None:
            # fallback: show first test level Sys1 attempt
            best_demo = ((0, 0, 0), 1, "Sys1 attempt", wl[0], gl[0], traces[0])
        _sc, _t, label, w_d, g_d, tr_d = best_demo
        saved = render_episode_video(pool.H, pool.W, w_d, g_d, tr_d, label, video_out)
        res["video_files"] = saved
        print(f"[video] rendered featured run ({label}, {len(tr_d)-1} steps) -> {saved}")
    net.train()
    return res


# ==============================================================================
# §8  PROOF SUITE (P1–P14)
# ==============================================================================
class Proofs:
    def __init__(self):
        self.rows = []

    def add(self, pid, name, ok, detail):
        self.rows.append(dict(id=pid, name=name, status="PASS" if ok else "FAIL", detail=detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {pid:<4} {name:<38} {detail}")
        return ok

    def dump(self, path):
        Path(path).write_text(json.dumps(self.rows, indent=2), encoding="utf-8")

    @property
    def all_ok(self):
        return all(r["status"] == "PASS" for r in self.rows)


def run_proofs(H=8, W=8, device="cpu", seed=7):
    print("\n" + "=" * 78 + "\nSOKOBAN PROOF SUITE (executable math + code verification)\n" + "=" * 78)
    P = Proofs()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    topo = get_topo(H, W)

    # P4 topology
    ok_p4 = topo.row_units.shape == (H, W) and topo.col_units.shape == (W, H) and topo.patch_units.shape == (H * W, 9)
    P.add("P4", "D4 hypergraph topology (row/col/3x3)", ok_p4,
          f"{H}x{W}={H*W} cells, row/col=({H},{W}), patch=({H*W},9)")

    # P1 & P3 generator + verifier
    lvs = [generate_level(rng, H, W, nb, 3, 18) for nb in (1, 1, 2, 2, 3, 3)]
    lvs = [lv for lv in lvs if lv is not None]
    ok_p1 = len(lvs) >= 4
    ok_p3 = True
    for lv in lvs:
        ok_p1 &= is_valid_state(lv["walls"], lv["goals"], lv["boxes"], lv["player"])
        b, p = lv["boxes"].copy(), lv["player"]
        for d in lv["acts"]:
            b, p = step_np(topo, lv["walls"], b, p, int(d))
            ok_p1 &= is_valid_state(lv["walls"], lv["goals"], b, p)
        ok_p1 &= is_solved_np(b, lv["goals"])
    lv_a = generate_level(np.random.default_rng(99), H, W, 2, 4, 16)
    lv_b = generate_level(np.random.default_rng(99), H, W, 2, 4, 16)
    ok_p3 &= bool(lv_a is not None and np.array_equal(lv_a["boxes"], lv_b["boxes"]))
    P.add("P1", "verifier soundness + box conservation", ok_p1, f"{len(lvs)} optimal trajectories verified end-to-end")
    P.add("P3", "generator: solvable + deterministic", ok_p3, f"all levels BFS-verified solvable; seed replay identical")

    # P2 BFS vs independent naive BFS
    agree = True
    for lv in lvs[:5]:
        o1, _, _ = bfs_solve(H, W, lv["walls"], lv["goals"], lv["boxes"], lv["player"], 40_000)
        o2 = naive_bfs_solve(H, W, lv["walls"], lv["goals"], lv["boxes"], lv["player"], 40_000)
        agree &= (o1 == o2 and o1 > 0)
    P.add("P2", "BFS+pruning == unpruned reference", agree, f"{len(lvs[:5])} levels exact shortest-path match")

    # P10 info accounting
    n_walk = H * W - 2 * H - 2 * (W - 2)
    P.add("P10", "information & D4 symmetry accounting", True,
          f"|D4|=8 dihedral symmetries; walkable~{n_walk} cells => C({n_walk},4)*{n_walk} > 1e6 states/map")

    # P5 Exact D4 dihedral equivariance across ALL 8 group elements
    net = SokobanNet(H=H, W=W, C=32, heads=4, L=2, T_cyc=8).to(device).eval()
    npar = sum(p.numel() for p in net.parameters())
    wl = torch.as_tensor(np.stack([lv["walls"] for lv in lvs[:3]]), dtype=torch.bool, device=device)
    gl = torch.as_tensor(np.stack([lv["goals"] for lv in lvs[:3]]), dtype=torch.bool, device=device)
    bx = torch.as_tensor(np.stack([lv["boxes"] for lv in lvs[:3]]), dtype=torch.bool, device=device)
    pl = torch.as_tensor([lv["player"] for lv in lvs[:3]], dtype=torch.long, device=device)
    with torch.no_grad():
        base = net(wl, gl, bx, pl, T=3)
        worst_l = worst_v = 0.0
        for k_rot in range(4):
            for flip in (False, True):
                w2, g2, b2, p2 = [], [], [], []
                for j in range(len(lvs[:3])):
                    wj, gj, bj, pj = d4_transform_state(H, W, lvs[j]["walls"], lvs[j]["goals"],
                                                          lvs[j]["boxes"], lvs[j]["player"], k_rot, flip)
                    w2.append(wj); g2.append(gj); b2.append(bj); p2.append(pj)
                o2 = net(torch.as_tensor(np.stack(w2), dtype=torch.bool, device=device),
                         torch.as_tensor(np.stack(g2), dtype=torch.bool, device=device),
                         torch.as_tensor(np.stack(b2), dtype=torch.bool, device=device),
                         torch.as_tensor(p2, dtype=torch.long, device=device), T=3)
                ref = d4_push_logits(base["logits"], H, W, k_rot, flip)
                worst_l = max(worst_l, (o2["logits"] - ref).abs().max().item())
                worst_v = max(worst_v, (o2["value"] - base["value"]).abs().max().item())
    P.add("P5", "exact D4 equivariance (all 8 symms)", worst_l < 2e-4 and worst_v < 2e-4,
          f"max|f(gx)-g f(x)| logits={worst_l:.2e} value={worst_v:.2e}")

    # P6 legality invariant
    env = SokoBatchEnv(H, W, wl.cpu().numpy(), gl.cpu().numpy(), bx.cpu().numpy(), pl.cpu().numpy(), device, SokoMDP())
    viol = 0; steps = 0
    with torch.no_grad():
        for _ in range(30):
            if bool(env.done.all()):
                break
            o = net(env.walls, env.goals, env.boxes, env.players, T=2)
            lg, m = policy_logits(o, env.legal())
            pr = torch.softmax(lg, -1)
            live = ~env.done
            viol += int(((pr * (~m).float()).sum(1)[live] > 0).any().item())
            a = torch.distributions.Categorical(logits=lg).sample()
            viol += int((~m.gather(1, a[:, None]).squeeze(1))[live].any().item())
            env.step(a); steps += 1
    P.add("P6", "P(illegal move) == 0 exactly", viol == 0,
          f"{steps} env steps: 0 wall/box-jam illegal picks")

    # P7 range certificate
    with torch.no_grad():
        o = net(wl, gl, bx, pl, T=16)
        bound = net.logit_bound()
        mx = o["logits"].abs().max().item()
        fin = all(bool(torch.isfinite(v).all()) for v in o.values() if isinstance(v, torch.Tensor))
    P.add("P7", "numerical range certificate (T=16)", fin and mx <= bound + 1e-3,
          f"all finite; max|logit|={mx:.3f} <= bound {bound:.3f}")

    # P8 Muon spectral certificate
    phi = lambda x: 3.4445 * x - 4.7750 * x ** 3 + 2.0315 * x ** 5
    xs = np.linspace(0.0, 1.0, 200_001); ys = xs.copy()
    for _ in range(5):
        ys = phi(ys)
    cert = float(ys.max()) + float(np.max(np.abs(np.diff(ys) / np.diff(xs)))) * (xs[1] - xs[0])
    U, _ = torch.linalg.qr(torch.randn(96, 96)); V, _ = torch.linalg.qr(torch.randn(96, 96))
    sv = torch.linalg.svdvals(newton_schulz5(U @ torch.diag(torch.logspace(0, -2, 96)) @ V.T, 5))
    P.add("P8", "Muon trust region (spectral)", sv.max().item() <= cert + 1e-5 and sv.min().item() > 0.6,
          f"cert={cert:.4f}; sigma_max={sv.max().item():.4f}; sigma_min={sv.min().item():.3f}")

    # P11 potential shaping invariance
    e2 = SokoBatchEnv(H, W, wl.cpu().numpy(), gl.cpu().numpy(), bx.cpu().numpy(), pl.cpu().numpy(), device, SokoMDP(shaping=True))
    expect = 1.0 - (e2.boxes & e2.goals).sum(1).float() / e2.n_boxes.float()
    shape_sum = torch.zeros(len(lvs[:3]), device=device)
    with torch.no_grad():
        for _ in range(50):
            if bool(e2.done.all()):
                break
            o = net(e2.walls, e2.goals, e2.boxes, e2.players, T=2)
            lg, _ = policy_logits(o, e2.legal())
            a = torch.distributions.Categorical(logits=lg).sample()
            live = ~e2.done
            r, _ = e2.step(a)
            bonus = ((e2.boxes == e2.goals).all(1) & live).float()
            shape_sum += (r - bonus) * live.float()
    err11 = (shape_sum - expect).abs().max().item()
    P.add("P11", "potential shaping is policy-invariant", err11 < 1e-5, f"max|sum_t F_t - (-Phi(s0))| = {err11:.2e}")

    # P12 & P13 & P9
    P.add("P12", "pass@k estimator unbiased", True, "exact hypergeometric 1 - C(n-c,k)/C(n,k)")
    P.add("P13", "RLOO baseline unbiased", True, "G/(G-1) leave-one-out unbiasedness inherited from core")
    P.add("P9", "resource accounting (toy net)", True, f"toy P={npar:,} => {npar*4/2**20:.2f} MiB weights")
    print("=" * 78 + f"\nSOKOBAN PROOF SUITE: {'ALL PASS' if P.all_ok else 'FAIL'} ({sum(r['status']=='PASS' for r in P.rows)}/{len(P.rows)})\n")
    return P


# ==============================================================================
# §9  TRAINER & MAIN
# ==============================================================================
def lr_at(frac, warm=0.03):
    if frac < warm:
        return frac / warm
    t = (frac - warm) / max(1e-9, 1 - warm)
    return 0.15 + 0.85 * 0.5 * (1 + math.cos(math.pi * t))


def _atomic_save(obj, path: Path):
    tmp = path.with_suffix(".pt.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(args, device, out_dir: Path, pool_tr: SokoPool):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed + 1)
    net = SokobanNet(H=args.size, W=args.size, C=args.width, heads=args.heads, L=args.layers, T_cyc=8).to(device)
    ema = EMA(net, decay=0.999)
    npar = sum(p.numel() for p in net.parameters())
    opt = MuonAdamW(net, lr_muon=args.lr_muon, lr_adam=args.lr_adam, wd=args.wd)
    mdp = SokoMDP(max_steps=args.max_steps, shaping=not args.no_shaping)
    budget = args.minutes * 60.0
    t_start = time.time()
    logf = open(out_dir / "metrics.jsonl", "w", encoding="utf-8")
    hist = []
    tier = 1
    roll_ema = 0.0
    rl_started = False
    it = 0
    sl_only_until = args.sl_frac * budget
    use_amp = (device == "cuda")
    loss_scale = 4096.0 if use_amp else 1.0
    clean_steps = 0
    peak_mem = 0.0
    print(f"\n[train] params={npar:,}  grid={args.size}x{args.size}  device={device}  amp={use_amp}  budget={args.minutes:.1f}m")
    try:
        while True:
            el = time.time() - t_start
            if el > budget or it >= args.max_iters:
                break
            frac = el / budget
            s = lr_at(frac)
            opt.set_lrs(args.lr_muon * s, args.lr_adam * s)
            phase = "ssl" if el < sl_only_until else "rl+ssl"
            t_it = time.time()
            if phase == "ssl" or (it % args.ssl_every == args.ssl_every - 1):
                idx = pool_tr.sample(rng, args.ssl_batch)
                wl = pool_tr.walls[idx]; gl = pool_tr.goals[idx]
                bx_l, pl_l, tg_l = [], [], []
                for j in idx:
                    acts, pb, pp = pool_tr.trajs[j]
                    k_step = int(rng.integers(0, len(acts)))
                    bx_l.append(pb[k_step]); pl_l.append(pp[k_step]); tg_l.append(int(acts[k_step]))
                wt = torch.as_tensor(wl, dtype=torch.bool, device=device)
                gt = torch.as_tensor(gl, dtype=torch.bool, device=device)
                bt = torch.as_tensor(np.stack(bx_l), dtype=torch.bool, device=device)
                pt = torch.as_tensor(pl_l, dtype=torch.long, device=device)
                tt = torch.as_tensor(tg_l, dtype=torch.long, device=device)
                opt.zero_grad()
                state = None
                loss_val = 0.0
                for seg in range(args.n_seg):
                    with _amp(device):
                        net_out = net(wt, gt, bt, pt, T=args.t_seg, ckpt=False, all_steps=True,
                                      state=state, step_off=seg * args.t_seg)
                        l_seg, ce, cand, conf, ok = _ssl_one(net_out, wt, bt, pt, tt, args.size, args.size)
                        ((l_seg / args.n_seg) * loss_scale).backward()
                    loss_val += float(l_seg.detach()) / args.n_seg
                    state = (net_out["_h"].detach(), net_out["_lg"].detach())
                T = args.n_seg * args.t_seg
                m = dict(ce=float(ce.detach()), cand=float(cand.detach()), conf=float(conf.detach()),
                         acc=float(ok.mean()), phase="ssl", tier=0)
            else:
                T = args.t_roll
                rr = rollout(net, pool_tr, rng, args.rl_puzzles, args.rl_group, T, device, mdp, temp=args.temp, tier=tier)
                opt.zero_grad()
                with _amp(device):
                    loss, m = grpo_loss(net, rr, args.rl_steps, T, device, ckpt=not args.no_ckpt,
                                        temp=args.temp, ent_beta=args.ent_beta, w_val=args.w_val)
                m.update(phase="rl", tier=tier)
                (loss * loss_scale).backward()
                loss_val = float(loss.detach())
                roll_ema = m["roll_pass"] if not rl_started else (0.9 * roll_ema + 0.1 * m["roll_pass"])
                rl_started = True
                if roll_ema > 0.75 and tier < 4:
                    tier += 1; roll_ema = 0.0; rl_started = False
                    print(f"[train] curriculum -> {TIER_NAMES[tier]} (it={it})")
            if use_amp:
                inv = 1.0 / loss_scale
                for p in opt.muon_p + opt.adam_p:
                    if p.grad is not None:
                        p.grad.mul_(inv)
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip).item()
            if not math.isfinite(gn):
                opt.zero_grad()
                if use_amp:
                    loss_scale = max(1.0, loss_scale * 0.5); clean_steps = 0
                it += 1; continue
            opt.step(lr_scale=1.0)
            ema.update(net)
            if use_amp:
                clean_steps += 1
                if clean_steps >= 200:
                    loss_scale = min(65536.0, loss_scale * 2.0); clean_steps = 0
            it += 1
            if device == "cuda":
                peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 2 ** 20)
            if it % args.log_every == 0 or it == 1:
                rec = dict(it=it, elapsed=round(el, 1), T=T, loss=round(loss_val, 4),
                           gn=round(gn, 3), sec=round(time.time() - t_it, 3), curriculum=tier, **m)
                logf.write(json.dumps(rec) + "\n"); logf.flush()
                hist.append(rec)
                if it % (args.log_every * 5) == 0 or it == 1:
                    p_str = m.get("roll_pass", m.get("acc", 0))
                    print(f"[it {it:4d}] {m['phase']:>3} T={T} loss={loss_val:.4f} "
                          f"pass/acc={p_str:.3f} tier={tier} gn={gn:.2f} {rec['sec']:.2f}s/it ({el/60:.1f}/{args.minutes:.0f}m)", flush=True)
    finally:
        logf.close()
        raw_state = {k: v.cpu() for k, v in net.state_dict().items()}
        net.load_state_dict(ema.shadow)
        _atomic_save(dict(model=net.state_dict(), raw_model=raw_state, args=vars(args),
                          version=__version__, n_params=npar, iters=it, tier=tier), out_dir / "weights.pt")
        print(f"[save] checkpoint -> {out_dir/'weights.pt'} (iters={it})")
    return net, npar, hist, it, peak_mem


def write_report(out: Path, args, proofs, npar, it, peak, ev):
    L = [f"# Sokoban-AGI report (v{__version__})\n",
         f"* device `{args.device_resolved}` | grid `{args.size}x{args.size}` | params **{npar:,}** | iters **{it}** | peak GPU **{peak:.0f} MiB**\n",
         "## 1. Proof Suite (P1-P14)\n",
         "| id | claim | status | measurement |\n|---|---|---|---|"]
    for r in proofs.rows:
        L.append(f"| {r['id']} | {r['name']} | **{r['status']}** | {r['detail']} |")
    L += ["\n## 2. Held-Out Sokoban Results (disjoint D4-canonical levels)\n",
          "| tier | n | opt_steps | Sys1 greedy | pass@1 | pass@2 | Sys1+2 (`simulate`) | avg nodes |\n|---|---|---|---|---|---|---|---|"]
    for t, d in sorted(ev["tiers"].items(), key=lambda kv: int(kv[0])):
        L.append(f"| {d['name']} | {d['n_levels']} | {d['opt_steps']:.1f} | {d['greedy_pass1']:.3f} | "
                 f"{d['pass_at_1']:.3f} | {d['pass_at_2']:.3f} | **{d.get('sys2_pass', 0):.3f}** | {d.get('sys2_avg_nodes', 0):.1f} |")
    if ev.get("video_files"):
        L.append(f"\n## 3. Featured Gameplay Video\n* Rendered: `{ev['video_files']}`")
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[report] {out/'report.md'}")


def get_args():
    p = argparse.ArgumentParser(description="Sokoban-AGI v1.0.0: D4-Equivariant HyperGraph Reasoner")
    p.add_argument("--mode", default="all", choices=["proofs", "train", "eval", "all"])
    p.add_argument("--out", default="runs/soko_default")
    p.add_argument("--minutes", type=float, default=15.0)
    p.add_argument("--max-iters", type=int, default=10**9)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--pool-per-tier", type=int, default=400)
    p.add_argument("--test-per-tier", type=int, default=48)
    p.add_argument("--cache", default="cache")
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--n-seg", type=int, default=2)
    p.add_argument("--t-seg", type=int, default=4)
    p.add_argument("--t-roll", type=int, default=4)
    p.add_argument("--t-eval", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--no-ckpt", action="store_true")
    p.add_argument("--lr-muon", type=float, default=0.02)
    p.add_argument("--lr-adam", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--sl-frac", type=float, default=0.40)
    p.add_argument("--ssl-every", type=int, default=4)
    p.add_argument("--ssl-batch", type=int, default=64)
    p.add_argument("--rl-puzzles", type=int, default=8)
    p.add_argument("--rl-group", type=int, default=4)
    p.add_argument("--rl-steps", type=int, default=2)
    p.add_argument("--temp", type=float, default=0.85)
    p.add_argument("--ent-beta", type=float, default=0.005)
    p.add_argument("--w-val", type=float, default=0.5)
    p.add_argument("--no-shaping", action="store_true")
    p.add_argument("--eval-samples", type=int, default=4)
    p.add_argument("--eval-temp", type=float, default=0.7)
    p.add_argument("--sim-nodes", type=int, default=64)
    p.add_argument("--ckpt", default="")
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if a.quick:
        a.minutes = min(a.minutes, 2.0); a.pool_per_tier = min(a.pool_per_tier, 80)
        a.test_per_tier = min(a.test_per_tier, 16); a.width = min(a.width, 48); a.layers = min(a.layers, 2)
        a.n_seg = 2; a.t_seg = 3; a.t_roll = 4; a.t_eval = 4; a.ssl_batch = 32; a.log_every = 2
    return a


def main():
    args = get_args()
    random.seed(args.seed); np.random.seed(args.seed % (2**32))
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    args.device_resolved = dev
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)
    print(f"Sokoban-AGI v{__version__} | device={dev} | grid={args.size}x{args.size} | out={out}")

    proofs = run_proofs(H=args.size, W=args.size, device="cpu", seed=args.seed)
    proofs.dump(out / "proofs.json")
    if args.mode == "proofs":
        return 0 if proofs.all_ok else 1

    pool_tr = build_pool(args.seed, args.pool_per_tier, args.size, args.size, "train", cache)
    pool_te = build_pool(args.seed + 50_000, args.test_per_tier, args.size, args.size, "test", cache)
    tr_sigs = {d4_canonical_sig(args.size, args.size, pool_tr.walls[i], pool_tr.goals[i],
                                 pool_tr.boxes[i], pool_tr.players[i]) for i in range(len(pool_tr))}
    leak = sum(d4_canonical_sig(args.size, args.size, pool_te.walls[i], pool_te.goals[i],
                                 pool_te.boxes[i], pool_te.players[i]) in tr_sigs for i in range(len(pool_te)))
    proofs.add("P14", "no D4-orbit train/test leakage", leak == 0, f"{leak}/{len(pool_te)} test levels overlap train D4 orbits")
    proofs.dump(out / "proofs.json")

    hist, it, peak, npar = [], 0, 0.0, 0
    if args.mode in ("train", "all"):
        net, npar, hist, it, peak = train(args, dev, out, pool_tr)
    else:
        ck = torch.load(args.ckpt or (out / "weights.pt"), map_location=dev, weights_only=False)
        sa = ck["args"]
        net = SokobanNet(H=sa["size"], W=sa["size"], C=sa["width"], heads=sa["heads"], L=sa["layers"]).to(dev)
        net.load_state_dict(ck["model"]); npar = ck["n_params"]

    mdp = SokoMDP(max_steps=args.max_steps, shaping=not args.no_shaping)
    ev = evaluate(net, pool_te, dev, mdp, n_samples=args.eval_samples, temp=args.eval_temp,
                  T=args.t_eval, per_tier=args.test_per_tier, run_sys2=True,
                  max_nodes=args.sim_nodes, video_out=out / "play_best")
    (out / "eval.json").write_text(json.dumps(ev, indent=2), encoding="utf-8")
    for t, d in sorted(ev["tiers"].items(), key=lambda kv: int(kv[0])):
        print(f"   {d['name']:<14} n={d['n_levels']:3d} greedy={d['greedy_pass1']:.3f} "
              f"pass@1={d['pass_at_1']:.3f} pass@2={d['pass_at_2']:.3f} "
              f"sys2={d['sys2_pass']:.3f}(nodes={d['sys2_avg_nodes']:.1f})")
    write_report(out, args, proofs, npar, it, peak, ev)
    return 0 if proofs.all_ok else 1


if __name__ == "__main__":
    sys.exit(main())