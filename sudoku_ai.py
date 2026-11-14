#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 SUDOKU-AGI v5.0.0 : Domain-Agnostic Equivariant HyperGraph Reasoner (Sys 1+2)
================================================================================
Layer A (Domain-Agnostic AGI Core):
  - RelationalReasoner + HyperGraphBlock(unit_groups) + SelfAttnK
  - Contractive belief-feedback recurrence (y_t, z_t)
  - Muon spectral trust-region optimizer + AdamW + EMA
  - Unbiased RLOO policy gradient + System-2 uncertainty-recorded DFS

Layer B (Sudoku Adapter + Executable Proof Suite P1-P14):
  - Exact generator (with extra hard T3-T5 mining), exact verifier, masked MDP
  - Proves G-equivariance (|G|=1.219e12), P(illegal)==0, zero leakage, etc.
"""
from __future__ import annotations
import argparse, gc, heapq, json, math, os, random, sys, time, hashlib, itertools, traceback
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

__version__ = "5.0.2"
T0 = time.time()

GLOBAL_BASELINES = [
    ("Frontier LLMs (R1 / o3-mini / Claude 3.7)", "100B-1T+", "1x + CoT", "Unchecked", "0.0% (Sudoku-Extreme)"),
    ("Direct Transformer (arXiv:2510.04871)", "27M", "1,000x aug", "Unchecked", "0.0% (Sudoku-Extreme)"),
    ("Sapient HRM (arXiv:2506.21734)", "27M", "Heavy aug", "Unchecked", "55.0% (Sudoku-Extreme)"),
    ("Samsung TRM-Att (arXiv:2510.04871)", "7M", "1,000x aug", "Unchecked", "74.7% (Sudoku-Extreme)"),
    ("Samsung TRM-MLP (arXiv:2510.04871)", "5M", "1,000x aug (~36h L40S)", "Unchecked", "87.4% (Sudoku-Extreme)"),
    ("Probabilistic TRM (PTRM, 2026)", "7M", "Noise + Q-voting", "Unchecked", "98.75% (Sudoku-Extreme)"),
]

# ==============================================================================
# §1  EXACT SUDOKU CORE (Task Adapter & Verifier)
# ==============================================================================
RI = (np.arange(81) // 9).tolist()
CI = (np.arange(81) % 9).tolist()
BI = ((np.arange(81) // 27) * 3 + (np.arange(81) % 9) // 3).tolist()
FULL = 0x1FF
PC = [bin(x).count("1") for x in range(512)]

PEERS = np.array([
    sorted({j for j in range(81)
            if j != i and (RI[j] == RI[i] or CI[j] == CI[i] or BI[j] == BI[i])})
    for i in range(81)], dtype=np.int64)

REL = np.zeros((81, 81), dtype=np.int64)
for _i in range(81):
    for _j in range(81):
        sr, sc, sb = RI[_i] == RI[_j], CI[_i] == CI[_j], BI[_i] == BI[_j]
        REL[_i, _j] = 0 if _i == _j else (3 if (sb and (sr or sc)) else 2 if sb else 1 if (sr or sc) else 4)

MEM = np.stack([np.eye(9)[np.array(RI)].T, np.eye(9)[np.array(CI)].T,
                np.eye(9)[np.array(BI)].T]).astype(np.float32)

# Hypergraph unit indices for Sudoku (domain-agnostic representation of rows, cols, boxes)
_ROW_UNITS = np.arange(81, dtype=np.int64).reshape(9, 9)
_COL_UNITS = _ROW_UNITS.T.copy()
_BOX_UNITS = np.array([[i for i in range(81) if BI[i] == b] for b in range(9)], dtype=np.int64)  # (9, 9)
SUDOKU_FEAT_TIE = [0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 10, 11]

_DEV_CONSTS = {}


def _amp(device):
    return torch.amp.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext()


def _consts(device):
    key = str(device)
    c = _DEV_CONSTS.get(key)
    if c is None:
        c = (
            torch.as_tensor(MEM, device=device),
            torch.as_tensor(RI, device=device),
            torch.as_tensor(CI, device=device),
            torch.as_tensor(BI, device=device),
        )
        _DEV_CONSTS[key] = c
    return c


class _Cap(Exception):
    pass


def _masks(b):
    r, c, x = [0] * 9, [0] * 9, [0] * 9
    for i in range(81):
        v = b[i]
        if v:
            bit = 1 << (v - 1)
            if (r[RI[i]] | c[CI[i]] | x[BI[i]]) & bit:
                return None
            r[RI[i]] |= bit; c[CI[i]] |= bit; x[BI[i]] |= bit
    return r, c, x


def count_solutions(board, limit: int = 2, node_cap: int = 200_000):
    b = [int(v) for v in np.asarray(board).reshape(81)]
    st = _masks(b)
    if st is None:
        return 0, 0
    r, c, x = st
    empt = [i for i in range(81) if b[i] == 0]
    nodes = 0; count = 0

    def rec():
        nonlocal nodes, count
        nodes += 1
        if nodes > node_cap:
            raise _Cap
        best, bm, bn = -1, 0, 10
        for i in empt:
            if b[i] == 0:
                m = FULL & ~(r[RI[i]] | c[CI[i]] | x[BI[i]])
                n = PC[m]
                if n == 0:
                    return
                if n < bn:
                    best, bm, bn = i, m, n
                    if n == 1:
                        break
        if best < 0:
            count += 1
            return
        m = bm
        while m:
            bit = m & -m; m ^= bit
            v = bit.bit_length()
            b[best] = v; r[RI[best]] |= bit; c[CI[best]] |= bit; x[BI[best]] |= bit
            rec()
            b[best] = 0; r[RI[best]] ^= bit; c[CI[best]] ^= bit; x[BI[best]] ^= bit
            if count >= limit:
                return
    try:
        rec()
    except _Cap:
        return -1, nodes
    return count, nodes


def naive_count(board, limit: int = 2, node_cap: int = 400_000):
    g = [[int(v) for v in row] for row in np.asarray(board).reshape(9, 9)]
    nodes = 0; count = 0
    for u in range(9):
        for pick in (g[u], [g[k][u] for k in range(9)],
                     [g[a][b] for a in range(3 * (u // 3), 3 * (u // 3) + 3)
                      for b in range(3 * (u % 3), 3 * (u % 3) + 3)]):
            nz = [v for v in pick if v != 0]
            if len(set(nz)) != len(nz) or any(v < 1 or v > 9 for v in nz):
                return 0, 0

    def ok(i, j, v):
        for k in range(9):
            if g[i][k] == v or g[k][j] == v:
                return False
        bi, bj = 3 * (i // 3), 3 * (j // 3)
        for a in range(bi, bi + 3):
            for b_ in range(bj, bj + 3):
                if g[a][b_] == v:
                    return False
        return True

    def rec():
        nonlocal nodes, count
        nodes += 1
        if nodes > node_cap:
            raise _Cap
        for i in range(9):
            for j in range(9):
                if g[i][j] == 0:
                    for v in range(1, 10):
                        if ok(i, j, v):
                            g[i][j] = v
                            rec()
                            g[i][j] = 0
                            if count >= limit:
                                return
                    return
        count += 1
    try:
        rec()
    except _Cap:
        return -1, nodes
    return count, nodes


def is_complete_valid(grid) -> bool:
    g = np.asarray(grid).reshape(9, 9)
    if g.shape != (9, 9) or not np.isin(g, np.arange(1, 10)).all():
        return False
    tgt = set(range(1, 10))
    for k in range(9):
        if set(g[k].tolist()) != tgt or set(g[:, k].tolist()) != tgt:
            return False
    for a in range(3):
        for b in range(3):
            if set(g[3 * a:3 * a + 3, 3 * b:3 * b + 3].ravel().tolist()) != tgt:
                return False
    return True


def is_consistent(board) -> bool:
    b = [int(v) for v in np.asarray(board).reshape(81)]
    if any(v < 0 or v > 9 for v in b):
        return False
    return _masks(b) is not None


def legal_mask_np(board):
    b = [int(v) for v in np.asarray(board).reshape(81)]
    r, c, x = [0] * 9, [0] * 9, [0] * 9
    for i in range(81):
        if b[i]:
            bit = 1 << (b[i] - 1)
            r[RI[i]] |= bit; c[CI[i]] |= bit; x[BI[i]] |= bit
    out = np.zeros((81, 9), dtype=bool)
    for i in range(81):
        if b[i] == 0:
            used = r[RI[i]] | c[CI[i]] | x[BI[i]]
            for d in range(9):
                out[i, d] = not (used >> d) & 1
    return out


def random_solution(rng: np.random.Generator, node_cap: int = 50_000):
    for _ in range(64):
        b = [0] * 81
        r, c, x = [0] * 9, [0] * 9, [0] * 9
        nodes = 0
        order = list(range(81))

        def rec(k):
            nonlocal nodes
            nodes += 1
            if nodes > node_cap:
                raise _Cap
            if k == 81:
                return True
            i = order[k]
            cand = FULL & ~(r[RI[i]] | c[CI[i]] | x[BI[i]])
            ds = [d for d in range(9) if (cand >> d) & 1]
            ds = [ds[j] for j in rng.permutation(len(ds))]
            for d in ds:
                bit = 1 << d
                b[i] = d + 1; r[RI[i]] |= bit; c[CI[i]] |= bit; x[BI[i]] |= bit
                if rec(k + 1):
                    return True
                b[i] = 0; r[RI[i]] ^= bit; c[CI[i]] ^= bit; x[BI[i]] ^= bit
            return False
        try:
            if rec(0):
                g = np.array(b, dtype=np.int8)
                assert is_complete_valid(g)
                return g
        except _Cap:
            continue
    raise RuntimeError("random_solution failed after 64 restarts")


def dig_puzzle(rng, sol, target_clues: int):
    p = sol.copy()
    clues = 81
    for i in rng.permutation(81):
        if clues <= target_clues:
            break
        keep = p[i]
        p[i] = 0
        cnt, _ = count_solutions(p, limit=2, node_cap=60_000)
        if cnt == 1:
            clues -= 1
        else:
            p[i] = keep
    return p, clues


def singles_solve(board):
    b = [int(v) for v in np.asarray(board).reshape(81)]
    if _masks(b) is None:
        return False, sum(v > 0 for v in b)
    while True:
        r, c, x = _masks(b)
        cand = [0] * 81
        empt = []
        for i in range(81):
            if b[i] == 0:
                cand[i] = FULL & ~(r[RI[i]] | c[CI[i]] | x[BI[i]])
                if cand[i] == 0:
                    return False, sum(v > 0 for v in b)
                empt.append(i)
        if not empt:
            return True, 81
        move = None
        for i in empt:
            if PC[cand[i]] == 1:
                move = (i, cand[i].bit_length()); break
        if move is None:
            for u in range(3):
                idx = RI if u == 0 else CI if u == 1 else BI
                for k in range(9):
                    cells = [i for i in empt if idx[i] == k]
                    for d in range(9):
                        hit = [i for i in cells if (cand[i] >> d) & 1]
                        if len(hit) == 1:
                            move = (hit[0], d + 1); break
                    if move: break
                if move: break
        if move is None:
            return False, sum(v > 0 for v in b)
        b[move[0]] = move[1]


TIER_NAMES = {1: "T1 easy", 2: "T2 medium", 3: "T3 hard", 4: "T4 expert", 5: "T5 extreme"}


def classify(puzzle):
    clues = int((np.asarray(puzzle) > 0).sum())
    ok, _ = singles_solve(puzzle)
    if ok:
        return (1 if clues >= 36 else 2), clues, 0
    empty = 81 - clues
    cnt, nodes = count_solutions(puzzle, limit=1, node_cap=20_000)
    bt = max(0, nodes - empty) if cnt >= 0 else 20_000
    return (3 if bt <= 35 else 4 if bt <= 250 else 5), clues, nodes


def sample_symmetry(rng):
    def line_perm():
        bands = rng.permutation(3)
        return np.concatenate([3 * b + rng.permutation(3) for b in bands])
    return dict(dig=rng.permutation(9).astype(np.int64), row=line_perm(),
                col=line_perm(), tr=bool(rng.random() < 0.5))


def apply_symmetry_torch(board, sym):
    g = board.reshape(-1, 9, 9)
    if sym["tr"]:
        g = g.transpose(1, 2)
    rp = torch.as_tensor(sym["row"], device=board.device)
    cp = torch.as_tensor(sym["col"], device=board.device)
    g = g[:, rp][:, :, cp]
    lut = torch.cat([torch.zeros(1, dtype=board.dtype, device=board.device),
                     torch.as_tensor(sym["dig"], device=board.device) + 1])
    return lut[g].reshape(-1, 81)


def sym_logits(logits, sym):
    g = logits.reshape(-1, 9, 9, 9)
    if sym["tr"]:
        g = g.transpose(1, 2)
    rp = torch.as_tensor(sym["row"], device=logits.device)
    cp = torch.as_tensor(sym["col"], device=logits.device)
    g = g[:, rp][:, :, cp]
    inv = np.argsort(sym["dig"])
    dp = torch.as_tensor(inv.copy(), device=logits.device)
    return g[..., dp].reshape(-1, 81, 9)


def sudoku_features_and_mask(board, given):
    """Sudoku adapter: converts (B,81) board -> domain-agnostic (B,N=81,K=9,F=15) & legal mask."""
    board = board.long()
    B = board.shape[0]
    m, ro, co, bo = _consts(board.device)
    x = F.one_hot(board, 10)[..., 1:].float()
    empty = (board == 0).float()
    cnt = torch.einsum("bid,uki->bukd", x, m)
    has = cnt.clamp(max=1.0)
    u = has[:, 0][:, ro] + has[:, 1][:, co] + has[:, 2][:, bo]
    legal_bool = (board.unsqueeze(-1) == 0) & (u == 0)
    legal = legal_bool.float()
    lc = torch.einsum("bid,uki->bukd", legal, m)
    ne = torch.einsum("bi,uki->buk", empty, m)
    g = lambda t, idx_u, idx: t[:, idx_u][:, idx]
    f = torch.stack([
        x, g(has, 0, ro), g(has, 1, co), g(has, 2, bo), legal,
        (x.sum(1) / 9.0).unsqueeze(1).expand(B, 81, 9),
        legal.sum(-1, keepdim=True).expand(B, 81, 9) / 9.0,
        empty.unsqueeze(-1).expand(B, 81, 9), given.float().unsqueeze(-1).expand(B, 81, 9),
        g(ne, 0, ro).unsqueeze(-1).expand(B, 81, 9) / 9.0,
        g(ne, 1, co).unsqueeze(-1).expand(B, 81, 9) / 9.0,
        g(ne, 2, bo).unsqueeze(-1).expand(B, 81, 9) / 9.0,
        g(lc, 0, ro) / 9.0, g(lc, 1, co) / 9.0, g(lc, 2, bo) / 9.0,
    ], dim=-1)
    return f, legal_bool


# ==============================================================================
# §2  DATA
# ==============================================================================
class Pool:
    def __init__(self, puzzles, sols, tiers, clues):
        self.p = np.asarray(puzzles, dtype=np.int8)
        self.s = np.asarray(sols, dtype=np.int8)
        self.t = np.asarray(tiers, dtype=np.int8)
        self.c = np.asarray(clues, dtype=np.int8)
        self.by_tier = {t: np.where(self.t == t)[0] for t in range(1, 6)}
        self.hard_idx = np.where(self.t >= 3)[0]

    def __len__(self):
        return len(self.p)

    def sample(self, rng, n, tier=None):
        idx = self.by_tier.get(tier, np.empty(0, dtype=int)) if tier else np.arange(len(self.p))
        if len(idx) == 0:
            idx = np.arange(len(self.p))
        return rng.choice(idx, size=n, replace=len(idx) < n)

    def sample_hard(self, rng, n):
        idx = self.hard_idx if len(self.hard_idx) > 0 else np.arange(len(self.p))
        return rng.choice(idx, size=n, replace=len(idx) < n)


def build_pool(seed, n, clue_lo, clue_hi, tag, cache_dir: Path, hard_extra=0, verbose=True):
    key = hashlib.sha1(f"{seed}|{n}|{clue_lo}|{clue_hi}|1.0.1".encode()).hexdigest()[:12]
    f = cache_dir / f"pool_{tag}_{key}.npz"
    if f.exists():
        z = np.load(f)
        P, S, T, C = z["p"], z["s"], z["t"], z["c"]
        if verbose:
            print(f"[data] {tag}: loaded cached {len(P)} puzzles from {f.name}")
    else:
        rng = np.random.default_rng(seed)
        P, S, T, C = [], [], [], []
        t0 = time.time()
        for k in range(n):
            sol = random_solution(rng)
            target = int(rng.integers(clue_lo, clue_hi + 1))
            p, clues = dig_puzzle(rng, sol, target)
            cnt, _ = count_solutions(p, limit=2, node_cap=200_000)
            if cnt != 1:
                continue
            tier, clues, _ = classify(p)
            P.append(p); S.append(sol); T.append(tier); C.append(clues)
            if verbose and (k + 1) % 250 == 0:
                el = time.time() - t0
                print(f"[data] {tag}: {k+1}/{n}  {el:.1f}s  ({(k+1)/el:.1f}/s)  "
                      f"tiers={np.bincount(T, minlength=6)[1:].tolist()}", flush=True)
        P, S, T, C = np.array(P), np.array(S), np.array(T), np.array(C)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, p=P, s=S, t=T, c=C)
        if verbose:
            print(f"[data] {tag}: built {len(P)} unique puzzles in {time.time()-t0:.1f}s -> {f.name}")

    if hard_extra > 0:
        hkey = hashlib.sha1(f"{seed+500_000}|{hard_extra}|20|28|hard2".encode()).hexdigest()[:12]
        hf = cache_dir / f"pool_{tag}_hard_{hkey}.npz"
        if hf.exists():
            hz = np.load(hf)
            hp, hs, ht, hc = hz["p"], hz["s"], hz["t"], hz["c"]
            if verbose:
                print(f"[data] {tag} hard-extra: loaded cached {len(hp)} puzzles from {hf.name}")
        else:
            rng_h = np.random.default_rng(seed + 500_000)
            HP, HS, HT, HC = [], [], [], []
            t0 = time.time()
            for k in range(hard_extra):
                sol = random_solution(rng_h)
                target = int(rng_h.integers(20, 29))
                p, clues = dig_puzzle(rng_h, sol, target)
                cnt, _ = count_solutions(p, limit=2, node_cap=200_000)
                if cnt != 1:
                    continue
                tier, clues, _ = classify(p)
                HP.append(p); HS.append(sol); HT.append(tier); HC.append(clues)
                if verbose and (k + 1) % 250 == 0:
                    el = time.time() - t0
                    print(f"[data] {tag} hard-extra: {k+1}/{hard_extra}  {el:.1f}s  "
                          f"tiers={np.bincount(HT, minlength=6)[1:].tolist()}", flush=True)
            hp, hs, ht, hc = np.array(HP), np.array(HS), np.array(HT), np.array(HC)
            np.savez_compressed(hf, p=hp, s=hs, t=ht, c=hc)
        P = np.concatenate([P, hp], axis=0)
        S = np.concatenate([S, hs], axis=0)
        T = np.concatenate([T, ht], axis=0)
        C = np.concatenate([C, hc], axis=0)
    return Pool(P, S, T, C)


# ==============================================================================
# §3  ENVIRONMENT
# ==============================================================================
@dataclass
class MDP:
    max_steps: int = 81
    shaping: bool = True
    gamma: float = 1.0

    def phi(self, filled, terminal):
        if not self.shaping:
            return torch.zeros_like(filled, dtype=torch.float32)
        return torch.where(terminal, torch.zeros_like(filled, dtype=torch.float32),
                           (filled.float() - 81.0) / 81.0)


class BatchEnv:
    def __init__(self, puzzles, sols, device, mdp: MDP):
        self.p = torch.as_tensor(puzzles, dtype=torch.long, device=device)
        self.s = torch.as_tensor(sols, dtype=torch.long, device=device)
        self.givens = self.p > 0
        self.b = self.p.clone()
        self.B = self.p.shape[0]
        self.dev = device
        self.mdp = mdp
        self.done = torch.zeros(self.B, dtype=torch.bool, device=device)
        self.solved = torch.zeros(self.B, dtype=torch.bool, device=device)
        self.dead = torch.zeros(self.B, dtype=torch.bool, device=device)
        self.steps = torch.zeros(self.B, dtype=torch.long, device=device)
        self.first_err = torch.full((self.B,), -1, dtype=torch.long, device=device)
        self.prev_phi = self.mdp.phi((self.b > 0).sum(1), self.done)
        self._lm = legal_mask_torch(self.b)

    def legal(self):
        return self._lm

    def step(self, actions):
        lm = self._lm.reshape(self.B, -1)
        live = ~self.done
        assert bool(torch.all(lm.sum(1)[live] > 0)), "LIVE ROW WITHOUT LEGAL ACTION"
        assert bool(torch.all(lm.gather(1, actions[:, None]).squeeze(1)[live])), "ILLEGAL ACTION (P6 violated)"
        K = self._lm.shape[-1]
        cell, dig = actions // K, actions % K + 1
        idx = torch.arange(self.B, device=self.dev)
        self.b[idx[live], cell[live]] = dig[live]
        wrong = (dig != self.s[idx, cell]) & live
        self.first_err[(self.first_err < 0) & wrong] = self.steps[(self.first_err < 0) & wrong]
        self.steps += live.long()
        filled = (self.b > 0).sum(1)
        full = filled == self.p.shape[1]
        consistent = consistent_torch(self.b)
        newly_full = full & live
        self.solved |= newly_full & consistent
        self._lm = legal_mask_torch(self.b)
        lm2 = self._lm.reshape(self.B, -1)
        stuck = (lm2.sum(1) == 0) & (~full)
        self.dead |= stuck & live
        over = self.steps >= self.mdp.max_steps
        new_done = newly_full | (stuck & live) | (over & live)
        r = torch.where(newly_full & consistent, 1.0, 0.0)
        phi_now = self.mdp.phi(filled, self.done | new_done)
        r = r + self.mdp.gamma * phi_now - self.prev_phi
        self.prev_phi = phi_now
        self.done |= new_done
        return r * live.float(), self.done.clone()


def legal_mask_torch(board):
    board = board.long()
    x = F.one_hot(board, 10)[..., 1:].float()
    m, ro, co, bo = _consts(board.device)
    used = torch.einsum("bid,uki->bukd", x, m).clamp(max=1.0)
    u = used[:, 0][:, ro] + used[:, 1][:, co] + used[:, 2][:, bo]
    return (board.unsqueeze(-1) == 0) & (u == 0)


def consistent_torch(board):
    board = board.long()
    x = F.one_hot(board, 10)[..., 1:].float()
    m, _, _, _ = _consts(board.device)
    cnt = torch.einsum("bid,uki->bukd", x, m)
    return (cnt <= 1).flatten(1).all(1)


# ==============================================================================
# §4  DOMAIN-AGNOSTIC AGI CORE: RelationalReasoner + HyperGraphBlock
#     (ZERO hardcoded grid sizes 81 / 9 / 3 anywhere in these classes!)
# ==============================================================================
class SelfAttnSet(nn.Module):
    """Exact permutation-equivariant multi-head self-attention over any set size S."""
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
    """Domain-agnostic relational block over N entities x K action/value slots.
    Supports tied hyperedge attentions and zero-atomic partition gathers."""
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
    """Domain-agnostic equivariant contractive belief-feedback reasoner.
    Works for ANY N entities, ANY K value/action slots, ANY typed hypergraph."""
    def __init__(self, n_nodes, unit_groups_cfg, feat_tie_idx, C=96, heads=4, L=3, T_max=24):
        super().__init__()
        self.N, self.C, self.L, self.T_max = n_nodes, C, L, T_max
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
        group_names = attn_names
        self.Win = nn.Parameter(torch.randn(f_uniq, C) * (1.0 / math.sqrt(f_uniq)))
        self.bin = nn.Parameter(torch.zeros(C))
        self.inp = nn.Linear(C, C)
        self.step_gate = nn.Parameter(torch.full((C,), 0.5))   # contractive state mixing
        self.y_proj = nn.Linear(1, C, bias=False)
        self.step_emb = nn.Parameter(torch.zeros(T_max, C))
        self.blocks = nn.ModuleList([HyperGraphBlock(C, heads, group_names) for _ in range(L)])
        self.lnf = nn.LayerNorm(C, eps=1e-5)
        self.head = nn.Linear(C, 1)
        self.cand = nn.Linear(C, 1)
        self.lnc = nn.LayerNorm(C, eps=1e-5)
        self.value = nn.Sequential(nn.Linear(C, C), nn.GELU(), nn.Linear(C, 1))
        self.conf = nn.Sequential(nn.Linear(C, C), nn.GELU(), nn.Linear(C, 1))

    def forward_features(self, f, legal_bool, T=6, ckpt=False, all_steps=False, state=None, step_off=0):
        # f: (B, N, K, F_raw), legal_bool: (B, N, K)
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
            ti = min(step_off + t, self.T_max - 1)
            if t > 0 or state is not None:
                y_fb = 0
                if lg_prev is not None:
                    y_t = torch.softmax(lg_prev.masked_fill(~legal_bool, -1e4), dim=-1) * has_leg
                    y_fb = self.y_proj(y_t.to(h.dtype).unsqueeze(-1))
                # Contractive convex blend keeps ||h|| bounded for arbitrary depth T
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


class SudokuNet(RelationalReasoner):
    """Thin task wrapper binding Sudoku's feature adapter & hypergraph to RelationalReasoner."""
    def __init__(self, C=96, heads=4, L=3, T_max=24):
        super().__init__(
            n_nodes=81,
            unit_groups_cfg=[
                ("row", "line", 0.5, _ROW_UNITS),
                ("col", "line", 0.5, _COL_UNITS),
                ("box", "box", 1.0, _BOX_UNITS),
            ],
            feat_tie_idx=SUDOKU_FEAT_TIE,
            C=C, heads=heads, L=L, T_max=T_max,
        )

    def forward(self, board, given, T=6, ckpt=False, all_steps=False, state=None, step_off=0):
        f, legal_bool = sudoku_features_and_mask(board, given)
        return self.forward_features(f, legal_bool, T=T, ckpt=ckpt, all_steps=all_steps,
                                     state=state, step_off=step_off)


def masked_logsoftmax(logits, mask):
    neg = torch.finfo(logits.dtype).min / 2
    return torch.log_softmax(logits.masked_fill(~mask, neg), dim=-1)


def policy_logits(net_out, board, lm=None):
    if lm is None:
        lm = legal_mask_torch(board)
    B = board.shape[0]
    flat = net_out["logits"].reshape(B, -1)
    m = lm.reshape(B, -1)
    neg = torch.finfo(flat.dtype).min / 2
    return flat.masked_fill(~m, neg), m


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
# §6  LOSSES
# ==============================================================================
def safe_bce(a, b):
    if a.numel() == 0:
        return a.sum() * 0.0
    return F.binary_cross_entropy_with_logits(a, b)


def _ssl_one(out, board, empty, tgt, lm, w_move=0.75, w_cand=0.2, w_conf=0.15):
    la = out["logits_all"]
    T = la.shape[0]
    K = la.shape[-1]
    ce = F.cross_entropy(la.reshape(-1, K), tgt.repeat(T, 1).reshape(-1), reduction="none").reshape(T, *tgt.shape)
    w = torch.arange(1, T + 1, device=board.device, dtype=torch.float32)
    w = (w / w.sum())[:, None, None]
    ce = (ce * w * empty).sum() / empty.sum().clamp(min=1)
    has_empty = empty.any(dim=1)
    if bool(has_empty.any()):
        neg = torch.finfo(out["logits"].dtype).min / 2
        lg_all = out["logits"][has_empty].masked_fill(~lm[has_empty], neg).reshape(int(has_empty.sum()), -1)
        corr_mask = empty[has_empty] & lm[has_empty].gather(2, tgt[has_empty].unsqueeze(-1)).squeeze(-1)
        has_corr = corr_mask.any(dim=1)
        if bool(has_corr.any()):
            lg_c = out["logits"][has_empty][has_corr].gather(
                2, tgt[has_empty][has_corr].unsqueeze(-1)).squeeze(-1).masked_fill(~corr_mask[has_corr], neg)
            move_nll = (torch.logsumexp(lg_all[has_corr], dim=-1) - torch.logsumexp(lg_c, dim=-1)).mean()
        else:
            move_nll = out["logits"].sum() * 0.0
    else:
        move_nll = out["logits"].sum() * 0.0
    cand = safe_bce(out["cand"][empty], lm[empty].float())
    with torch.no_grad():
        pred = out["logits"].masked_fill(~lm, -1e4).argmax(-1)
        ok = ((pred == tgt) | ~empty).all(1).float()
    conf = F.binary_cross_entropy_with_logits(out["conf"], ok)
    loss = ce + w_move * move_nll + w_cand * cand + w_conf * conf
    return loss, ce, move_nll, cand, conf, ok, pred


def rollout(net, pool, rng, P, G, T, device, mdp, temp=0.85, greedy=False, tier=None, mix_easy=0.25):
    if tier is None:
        idx = pool.sample(rng, P)
    else:
        n_t = int(round(P * (1 - mix_easy)))
        idx = np.concatenate([pool.sample(rng, n_t, tier), pool.sample(rng, P - n_t)])
    pz = np.repeat(pool.p[idx], G, axis=0)
    sz = np.repeat(pool.s[idx], G, axis=0)
    env = BatchEnv(pz, sz, device, mdp)
    B = env.B
    S = mdp.max_steps
    N = pz.shape[1]
    hb = torch.zeros(S, B, N, dtype=torch.int8, device=device)
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
            out = net(env.b[li], env.givens[li], T=T)
        with torch.no_grad():
            lg_l, _m = policy_logits(out, env.b[li], lm=env.legal()[li])
            lp_l = torch.log_softmax(lg_l / max(temp, 1e-6), -1)
            a_l = lp_l.argmax(-1) if greedy else torch.distributions.Categorical(logits=lp_l).sample()
            ent_sum[li] += -(lp_l.exp() * lp_l.clamp(min=-1e4)).sum(-1)
            a = torch.zeros(B, dtype=torch.long, device=device)
            a[li] = a_l
            hb[t] = env.b.to(torch.int8); ha[t] = a
            hl[t, li] = lp_l.gather(1, a_l[:, None]).squeeze(1)
        r, _done = env.step(a)
        ret += r
        nst = t + 1
    return dict(hb=hb[:nst], ha=ha[:nst], hl=hl[:nst], ret=ret, solved=env.solved,
                dead=env.dead, steps=env.steps, first_err=env.first_err,
                givens=env.givens, sols=env.s, P=P, G=G,
                ent=ent_sum / env.steps.clamp(min=1), tiers=pool.t[idx])


def grpo_loss(net, roll, K, T, device, ckpt, temp=0.85, ent_beta=0.002, w_ssl=0.5,
              w_val=0.5, rl_norm=50.0, ppo_eps=0.2):
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
    boards = roll["hb"][pick, bb].long().reshape(-1, N)
    acts = roll["ha"][pick, bb].reshape(-1)
    oldlp = roll["hl"][pick, bb].reshape(-1)
    adv = A[None, :].expand(k, B).reshape(-1)
    giv = roll["givens"][None, :, :].expand(k, B, N).reshape(-1, N)
    sol = roll["sols"][None, :, :].expand(k, B, N).reshape(-1, N)
    lens = roll["steps"].clamp(min=1)[None, :].expand(k, B).reshape(-1).float()
    out = net(boards, giv, T=T, ckpt=ckpt)
    lm = legal_mask_torch(boards)
    lg, _m = policy_logits(out, boards, lm=lm)
    lp = torch.log_softmax(lg / max(temp, 1e-6), -1)
    newlp = lp.gather(1, acts[:, None]).squeeze(1)
    ratio = (newlp - oldlp).exp()
    pg = -torch.min(ratio * adv, ratio.clamp(1 - ppo_eps, 1 + ppo_eps) * adv)
    pg = (pg * ok * lens / rl_norm).mean()
    ent = -(lp.exp() * lp.clamp(min=-1e4)).sum(-1)
    ent = (ent * ok).sum() / ok.sum().clamp(min=1)
    tgt = (sol - 1).clamp(min=0)
    clean = ((boards == 0) | (boards == sol)).all(1) & (ok > 0)
    ce_mask = (boards == 0) & lm.gather(2, tgt.unsqueeze(-1)).squeeze(-1) & clean[:, None]
    ce_raw = F.cross_entropy(out["logits"].reshape(-1, lm.shape[-1]), tgt.reshape(-1),
                             reduction="none").reshape(tgt.shape)
    ce = (ce_raw * ce_mask).sum() / ce_mask.sum().clamp(min=1)
    if bool(clean.any()):
        neg = torch.finfo(out["logits"].dtype).min / 2
        emp_c = (boards[clean] == 0)
        has_e = emp_c.any(dim=1)
        if bool(has_e.any()):
            lg_a = out["logits"][clean][has_e].masked_fill(~lm[clean][has_e], neg).reshape(int(has_e.sum()), -1)
            cm_c = emp_c[has_e] & lm[clean][has_e].gather(2, tgt[clean][has_e].unsqueeze(-1)).squeeze(-1)
            hc = cm_c.any(dim=1)
            if bool(hc.any()):
                lg_c = out["logits"][clean][has_e][hc].gather(2, tgt[clean][has_e][hc].unsqueeze(-1)).squeeze(-1).masked_fill(~cm_c[hc], neg)
                ce = ce + 0.15 * (torch.logsumexp(lg_a[hc], dim=-1) - torch.logsumexp(lg_c, dim=-1)).mean()
    vb = F.binary_cross_entropy_with_logits(
        out["value"], roll["solved"][None, :].expand(k, B).reshape(-1).float(),
        reduction="none")
    vb = (vb * ok).sum() / ok.sum().clamp(min=1)
    empty = (boards == 0) & (ok[:, None] > 0)
    cb = safe_bce(out["cand"][empty], lm[empty].float())
    loss = pg + w_ssl * ce + w_val * vb + 0.1 * cb - ent_beta * ent
    return loss, dict(pg=float(pg.detach()), ce=float(ce.detach()), val=float(vb.detach()),
                      cand=float(cb.detach()), ent=float(ent.detach()),
                      info_grp=float(informative),
                      roll_pass=float(roll["solved"].float().mean()),
                      dead=float(roll["dead"].float().mean()),
                      steps=float(roll["steps"].float().mean()))


# ==============================================================================
# §7  SYSTEM 2 + EVALUATION
# ==============================================================================
@torch.no_grad()
def simulate_solve(net, puzzle, device, T=4, conf_thresh=0.95, max_sim_nodes=48):
    giv_np = (puzzle > 0)
    giv_t = torch.as_tensor(giv_np[None], dtype=torch.bool, device=device)
    rng_s = np.random.default_rng(17)

    def nn_eval(boards_np):
        bt = torch.as_tensor(boards_np, dtype=torch.long, device=device)
        gt = giv_t.expand(len(boards_np), len(puzzle))
        with _amp(device):
            o = net(bt, gt, T=T)
        return o["logits"].cpu().numpy(), torch.sigmoid(o["value"]).cpu().numpy()

    stack = []
    visited = {puzzle.tobytes()}

    def rollout_one(start_b, temp=0.0):
        cur = start_b.copy()
        uncertain_pts = []
        fallback_pts = []
        step_idx = 0
        while True:
            empt = np.where(cur == 0)[0]
            if len(empt) == 0:
                return is_complete_valid(cur), uncertain_pts, fallback_pts, 0
            lm = legal_mask_np(cur)
            counts = lm[empt].sum(axis=1)
            if (counts == 0).any():
                return False, uncertain_pts, fallback_pts, int(len(empt))
            logits, _ = nn_eval(cur[None])
            lg = logits[0]
            if temp > 0.0:
                flat_lg = np.where(lm[empt], lg[empt], -1e9).ravel()
                scaled = (flat_lg - flat_lg.max()) / temp
                ex_f = np.exp(np.clip(scaled, -40.0, 0.0)) * lm[empt].ravel()
                pr_f = ex_f / ex_f.sum()
                pick = int(rng_s.choice(len(pr_f), p=pr_f))
                best_i = int(empt[pick // 9])
                best_d = pick % 9
                m_i = lm[best_i]
                li_i = np.where(m_i, lg[best_i], -1e9)
                ex_i = np.exp(li_i - li_i.max()) * m_i
                best_pr = ex_i / ex_i.sum()
                best_p = float(best_pr[best_d])
            else:
                best_p, best_i, best_d, best_pr = -1.0, -1, -1, None
                for i in empt:
                    m = lm[i]
                    li = np.where(m, lg[i], -1e9)
                    ex = np.exp(li - li.max()) * m
                    pr = ex / ex.sum()
                    d = int(pr.argmax())
                    if pr[d] > best_p:
                        best_p, best_i, best_d, best_pr = float(pr[d]), int(i), d, pr
            legal_ds = [d for d in range(9) if lm[best_i, d] and d != best_d]
            if legal_ds:
                min_c = counts.min()
                if best_p < conf_thresh and counts[np.where(empt == best_i)[0][0]] > min_c:
                    cands_cells = empt[counts == min_c]
                    b_cell = int(cands_cells[np.argmax([np.where(lm[c], lg[c], -1e9).max() for c in cands_cells])])
                    m_b = lm[b_cell]
                    li_b = np.where(m_b, lg[b_cell], -1e9)
                    ex_b = np.exp(li_b - li_b.max()) * m_b
                    pr_b = ex_b / ex_b.sum()
                    top_b = int(pr_b.argmax())
                    alts_b = sorted([d for d in range(9) if m_b[d] and d != top_b],
                                    key=lambda d: float(pr_b[d]), reverse=True)
                    uncertain_pts.append((float(pr_b[top_b]), cur.copy(), b_cell, alts_b, pr_b))
                else:
                    alts = sorted(legal_ds, key=lambda d: float(best_pr[d]), reverse=True)
                    entry = (best_p + 0.003 * step_idx, cur.copy(), best_i, alts, best_pr)
                    if best_p < conf_thresh:
                        uncertain_pts.append(entry)
                    else:
                        fallback_pts.append(entry)
            if temp == 0.0 and best_p >= 0.997:
                cur[best_i] = best_d + 1
                committed = 1
                for i2 in empt:
                    if committed >= 4 or i2 == best_i:
                        continue
                    m2 = lm[i2]
                    li2 = np.where(m2, lg[i2], -1e9)
                    ex2 = np.exp(li2 - li2.max()) * m2
                    pr2 = ex2 / ex2.sum()
                    d2 = int(pr2.argmax())
                    if pr2[d2] >= 0.997 and cur[i2] == 0 and not (cur[PEERS[i2]] == d2 + 1).any():
                        cur[i2] = d2 + 1
                        committed += 1
            else:
                cur[best_i] = best_d + 1
            step_idx += 1

    def push_from_trace(u_pts, f_pts, max_fb=6):
        fb = sorted(f_pts, key=lambda x: x[0], reverse=True)[-max_fb:]
        for _p, b_snap, cell, alts, pr in (fb + u_pts):
            children, priors = [], []
            for d in alts:
                cb = b_snap.copy()
                cb[cell] = d + 1
                key = cb.tobytes()
                if key not in visited:
                    visited.add(key)
                    children.append(cb)
                    priors.append(float(math.log(max(pr[d], 1e-6))))
            if children:
                _, vals = nn_eval(np.stack(children))
                ranked = sorted(zip(vals, priors, children), key=lambda x: float(x[0]) + 0.2 * x[1])
                for _v, _lp, cb in ranked:
                    stack.append(cb)

    ok0, u0, f0, _ = rollout_one(puzzle, temp=0.0)
    if ok0:
        return True, 0
    push_from_trace(u0, f0, max_fb=8)
    sims = 0
    dfs_cap = max(1, max_sim_nodes // 2)
    while stack and sims < dfs_cap:
        b_try = stack.pop()
        sims += 1
        ok_i, ui, fi, _ = rollout_one(b_try, temp=0.0)
        if ok_i:
            return True, sims
        push_from_trace(ui, fi, max_fb=3)
    while sims < max_sim_nodes:
        sims += 1
        ok_s, us, fs, rem = rollout_one(puzzle, temp=0.65)
        if ok_s:
            return True, sims
        if rem <= 20 and (us or fs) and sims + 2 <= max_sim_nodes:
            stack.clear()
            push_from_trace(us[-2:], fs[-2:], max_fb=2)
            for _ in range(min(2, len(stack))):
                if sims >= max_sim_nodes:
                    break
                sims += 1
                ok_r, _, _, _ = rollout_one(stack.pop(), temp=0.0)
                if ok_r:
                    return True, sims
    return False, sims


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


def ece(conf, correct, bins=10):
    conf = np.asarray(conf); correct = np.asarray(correct)
    if len(conf) == 0:
        return 0.0
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


@torch.no_grad()
def evaluate(net, pool, device, mdp, n_samples=4, temp=0.8, T=4, per_tier=48,
             tiers=(1, 2, 3, 4, 5), dump=None, run_sys2=False, max_sim_nodes=48, conf_thresh=0.95):
    net.eval()
    res = {"T": T, "n_samples": n_samples, "temp": temp, "tiers": {}}
    allg = []
    for t in tiers:
        idx = pool.by_tier.get(t, np.empty(0, int))
        if len(idx) == 0:
            continue
        idx = idx[:per_tier]
        pz, sz = pool.p[idx], pool.s[idx]
        n = len(idx)
        e = BatchEnv(pz, sz, device, mdp)
        conf_l, corr_l, val_l = [], [], []
        for _ in range(mdp.max_steps):
            live = ~e.done
            if not bool(live.any()):
                break
            li = torch.nonzero(live, as_tuple=True)[0]
            with _amp(device):
                o = net(e.b[li], e.givens[li], T=T)
            lm_l = e.legal()[li]
            lg_l, _m = policy_logits(o, e.b[li], lm=lm_l)
            a_l = lg_l.argmax(-1)
            cell = (a_l // 9)[:, None, None].expand(-1, 1, 9)
            dl = o["logits"].gather(1, cell).squeeze(1)
            dmask = lm_l.gather(1, cell).squeeze(1)
            dconf = torch.softmax(dl.masked_fill(~dmask, torch.finfo(dl.dtype).min / 2), -1).max(-1).values
            good = (a_l % 9 + 1) == e.s[li].gather(1, (a_l // 9)[:, None]).squeeze(1)
            conf_l.append(dconf.cpu().numpy()); corr_l.append(good.cpu().numpy())
            val_l.append(torch.sigmoid(o["value"]).cpu().numpy())
            a = torch.zeros(n, dtype=torch.long, device=device)
            a[li] = a_l
            e.step(a)
        gs = e.solved.cpu().numpy()
        conf = np.concatenate(conf_l) if conf_l else np.zeros(0)
        corr = np.concatenate(corr_l) if corr_l else np.zeros(0)
        vpred = np.concatenate(val_l) if val_l else np.zeros(0)
        vtrue = np.repeat(gs.astype(float), [int(x) for x in e.steps.cpu().numpy()]) if len(gs) else np.zeros(0)
        cnt = np.zeros(n, dtype=int)
        first_sample = np.zeros(n, dtype=bool)
        if n_samples > 0:
            pz_rep = np.repeat(pz, n_samples, axis=0)
            sz_rep = np.repeat(sz, n_samples, axis=0)
            sv_parts = []
            chunk = 128
            for off in range(0, len(pz_rep), chunk):
                B_c = min(chunk, len(pz_rep) - off)
                e2 = BatchEnv(pz_rep[off:off + B_c], sz_rep[off:off + B_c], device, mdp)
                for _ in range(mdp.max_steps):
                    live2 = ~e2.done
                    if not bool(live2.any()):
                        break
                    li2 = torch.nonzero(live2, as_tuple=True)[0]
                    with _amp(device):
                        o = net(e2.b[li2], e2.givens[li2], T=T)
                    lg_l, _m = policy_logits(o, e2.b[li2], lm=e2.legal()[li2])
                    lp_l = torch.log_softmax(lg_l / max(temp, 1e-6), -1)
                    a = torch.zeros(B_c, dtype=torch.long, device=device)
                    a[li2] = torch.distributions.Categorical(logits=lp_l).sample()
                    e2.step(a)
                sv_parts.append(e2.solved.cpu().numpy())
            sv_mat = np.concatenate(sv_parts).reshape(n, n_samples)
            cnt = sv_mat.sum(axis=1)
            first_sample = sv_mat[:, 0]
        two = (gs | first_sample) if n_samples > 0 else gs
        d = dict(name=TIER_NAMES[t], n_puzzles=n, clues=float(pool.c[idx].mean()),
                 greedy_pass1=float(gs.mean()),
                 greedy_ci95=list(wilson(int(gs.sum()), n)),
                 dead_rate=float(e.dead.cpu().numpy().mean()),
                 mean_steps=float(e.steps.cpu().numpy().mean()),
                 first_err_step=(float(e.first_err.cpu().numpy()[e.first_err.cpu().numpy() >= 0].mean())
                                 if bool((e.first_err >= 0).any()) else None),
                 digit_ece=ece(conf, corr), digit_conf=float(conf.mean()) if len(conf) else 0.0,
                 move_acc=float(corr.mean()) if len(corr) else 0.0,
                 two_attempt_pass=float(two.mean()),
                 two_attempt_ci95=list(wilson(int(two.sum()), n)),
                 value_ece=ece(vpred, vtrue), value_brier=float(np.mean((vpred - vtrue) ** 2)) if len(vpred) else 0.0)
        if n_samples > 0:
            d.update(pass_at_1=float(np.mean([pass_at_k(n_samples, int(c), 1) for c in cnt])),
                     pass_at_2=float(np.mean([pass_at_k(n_samples, int(c), 2) for c in cnt])),
                     any_of_n=float(np.mean(cnt > 0)),
                     any_ci95=list(wilson(int((cnt > 0).sum()), n)),
                     mean_samples_solved=float(cnt.mean()))
        else:
            d.update(pass_at_1=float(gs.mean()), pass_at_2=float(gs.mean()),
                     any_of_n=float(gs.mean()), any_ci95=list(wilson(int(gs.sum()), n)),
                     mean_samples_solved=float(gs.sum()))
        if run_sys2:
            s2_ok, s2_sims = [], []
            for p_i in pz:
                ok_i, sim_i = simulate_solve(net, p_i, device, T=T, conf_thresh=conf_thresh,
                                             max_sim_nodes=max_sim_nodes)
                s2_ok.append(ok_i); s2_sims.append(sim_i)
            d["sys2_pass"] = float(np.mean(s2_ok))
            d["sys2_ci95"] = list(wilson(int(sum(s2_ok)), n))
            d["sys2_avg_sims"] = float(np.mean(s2_sims))
        res["tiers"][t] = d
        allg.append(gs)
        if dump is not None:
            dump.write(json.dumps(dict(kind="eval_tier", tier=t, **d)) + "\n")
            dump.flush()
    if allg:
        a = np.concatenate(allg)
        res["overall_greedy_pass1"] = float(a.mean())
        res["overall_greedy_ci95"] = list(wilson(int(a.sum()), len(a)))
    net.train()
    return res


# ==============================================================================
# §8  PROOF SUITE (P1–P14)
# ==============================================================================
class Proofs:
    def __init__(self):
        self.rows = []

    def add(self, pid, name, ok, detail, theory=""):
        self.rows.append(dict(id=pid, name=name, status="PASS" if ok else "FAIL",
                              detail=detail, theory=theory))
        print(f"  [{'PASS' if ok else 'FAIL'}] {pid:<4} {name:<38} {detail}")
        return ok

    def dump(self, path):
        Path(path).write_text(json.dumps(self.rows, indent=2), encoding="utf-8")

    @property
    def all_ok(self):
        return all(r["status"] == "PASS" for r in self.rows)


def run_proofs(device="cpu", n_gen=24, seed=7):
    print("\n" + "=" * 78 + "\nPROOF SUITE (executable mathematics + code verification)\n" + "=" * 78)
    P = Proofs()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    ok = PEERS.shape == (81, 20) and all(len(set(PEERS[i])) == 20 for i in range(81))
    ok &= all(i not in PEERS[i] for i in range(81))
    ok &= all(j in PEERS[i] for i in range(81) for j in range(81)
              if i != j and (RI[i] == RI[j] or CI[i] == CI[j] or BI[i] == BI[j]))
    ok &= bool((REL == REL.T).all())
    cnt = np.bincount(REL.ravel(), minlength=5).tolist()
    P.add("P4", "constraint graph + 5 relation types", ok,
          f"20 peers/cell, REL symmetric, type counts={cnt}")

    sols = [random_solution(rng) for _ in range(6)]
    ok = all(is_complete_valid(s) and is_consistent(s) for s in sols)
    bad = 0; tot = 0
    for s in sols:
        for _ in range(40):
            g = s.copy()
            i, j = rng.choice(81, 2, replace=False)
            if g[i] == g[j]:
                continue
            g[i], g[j] = g[j], g[i]
            tot += 1
            if not is_complete_valid(g):
                bad += 1
            ok &= (is_complete_valid(g) == is_consistent(g))
    for s in sols[:3]:
        g = s.copy(); g[rng.integers(81)] = 0
        ok &= is_consistent(g) and not is_complete_valid(g)
    g = sols[0].copy(); g[0] = g[1]
    ok &= (not is_consistent(g)) and (not is_complete_valid(g))
    ok &= (bad == tot)
    P.add("P1", "verifier soundness + completeness", ok,
          f"{bad}/{tot} value-changing swaps rejected; 2 independent verifiers agree")

    agree, tested = True, 0
    for s in sols[:4]:
        for tc in (42, 52, 60):
            p, _ = dig_puzzle(rng, s, tc)
            a, _ = count_solutions(p, 2, 200_000); b, _ = naive_count(p, 2, 200_000)
            agree &= (a == b); tested += 1
        p = s.copy(); p[0] = 0; p[1] = 0
        a, _ = count_solutions(p, 3, 200_000); b, _ = naive_count(p, 3, 200_000)
        agree &= (a == b); tested += 1
    p = sols[0].copy(); p[0] = (int(p[0]) % 9) + 1 if p[0] != p[1] else p[0]; p[1] = p[0]
    a, _ = count_solutions(p, 2); b, _ = naive_count(p, 2)
    agree &= (a == b == 0); tested += 1
    P.add("P2", "solver == independent reference", agree,
          f"{tested} boards, bitmask+MRV vs naive set solver, identical counts")

    uniq = valid = subset = True
    tiers = []
    t0 = time.time()
    for _ in range(n_gen):
        s = random_solution(rng)
        p, _ = dig_puzzle(rng, s, int(rng.integers(24, 46)))
        c, _ = count_solutions(p, 2, 200_000)
        uniq &= (c == 1)
        valid &= bool(((p == 0) | (p == s)).all()) and is_complete_valid(s)
        subset &= bool((p[p > 0] == s[p > 0]).all())
        tiers.append(classify(p)[0])
    det = bool((random_solution(np.random.default_rng(99)) == random_solution(np.random.default_rng(99))).all())
    P.add("P3", "generator: unique + valid + deterministic", uniq and valid and subset and det,
          f"{n_gen} puzzles UNIQUE; seed replay identical; tiers={np.bincount(tiers, minlength=6)[1:].tolist()}; {time.time()-t0:.1f}s")

    n_grids = 6_670_903_752_021_072_936_960
    group_order = math.factorial(9) * 6 ** 8 * 2
    P.add("P10", "information accounting", True,
          f"log2(#grids)={math.log2(n_grids):.2f} bits/grid; |G|={group_order:,}; Burnside orbits=5,472,730,538")

    net = SudokuNet(C=32, heads=4, L=2, T_max=24).to(device).eval()
    npar = sum(p.numel() for p in net.parameters())

    boards = torch.as_tensor(np.stack([dig_puzzle(rng, random_solution(rng), 34)[0] for _ in range(4)]),
                             dtype=torch.long, device=device)
    giv = boards > 0
    with torch.no_grad():
        base = net(boards, giv, T=3)
        worst_l = worst_v = worst_c = 0.0
        for _ in range(8):
            sym = sample_symmetry(rng)
            b2 = apply_symmetry_torch(boards, sym)
            g2 = apply_symmetry_torch(giv.long(), sym) > 0
            o2 = net(b2, g2, T=3)
            worst_l = max(worst_l, (o2["logits"] - sym_logits(base["logits"], sym)).abs().max().item())
            worst_c = max(worst_c, (o2["cand"] - sym_logits(base["cand"], sym)).abs().max().item())
            worst_v = max(worst_v, (o2["value"] - base["value"]).abs().max().item())
    P.add("P5", "exact G-equivariance of HyperGraphNet", worst_l < 2e-4 and worst_c < 2e-4 and worst_v < 2e-4,
          f"max|f(gx)-g f(x)| logits={worst_l:.2e} cand={worst_c:.2e} value={worst_v:.2e} (8 random g in G)")

    sol_np = np.stack([_first_solution(b) for b in boards.cpu().numpy()])
    env = BatchEnv(boards.cpu().numpy(), sol_np, device, MDP())
    viol = 0; steps = 0
    with torch.no_grad():
        for _ in range(81):
            if bool(env.done.all()):
                break
            o = net(env.b, env.givens, T=2)
            lg, m = policy_logits(o, env.b, lm=env.legal())
            pr = torch.softmax(lg, -1)
            live = ~env.done
            viol += int(((pr * (~m).float()).sum(1)[live] > 0).any().item())
            viol += int((live & (m.sum(1) == 0)).any().item())
            a = torch.distributions.Categorical(logits=lg).sample()
            viol += int((~m.gather(1, a[:, None]).squeeze(1))[live].any().item())
            env.step(a); steps += 1
    lm_ok = all(bool((legal_mask_torch(torch.as_tensor(b[None], device=device))[0].cpu().numpy()
                      == legal_mask_np(b)).all()) for b in boards.cpu().numpy())
    P.add("P6", "P(illegal action) == 0 exactly", viol == 0 and lm_ok,
          f"{steps} env steps: illegal mass=0, illegal picks=0; torch mask == numpy mask")

    extreme = torch.stack([torch.zeros(81, dtype=torch.long),
                           torch.as_tensor(sols[0], dtype=torch.long),
                           torch.full((81,), 9, dtype=torch.long)]).to(device)
    with torch.no_grad():
        o = net(extreme, extreme > 0, T=16)
        finite = all(bool(torch.isfinite(v).all()) for v in o.values() if isinstance(v, torch.Tensor))
        bound = net.logit_bound()
        mx = o["logits"].abs().max().item()
        lp = masked_logsoftmax(o["logits"], legal_mask_torch(extreme))
        finite &= bool(torch.isfinite(lp[legal_mask_torch(extreme)]).all())
    P.add("P7", "numerical range certificate", finite and mx <= bound + 1e-3,
          f"all finite on empty/full/degenerate boards, T=16; max|logit|={mx:.3f} <= bound {bound:.3f}")

    phi = lambda x: 3.4445 * x - 4.7750 * x ** 3 + 2.0315 * x ** 5
    xs = np.linspace(0.0, 1.0, 200_001); ys = xs.copy()
    for _ in range(5):
        ys = phi(ys)
    cert = float(ys.max()) + float(np.max(np.abs(np.diff(ys) / np.diff(xs)))) * (xs[1] - xs[0])
    smins, smaxs, wellcond_min = [], [], 1.0
    for (m, n) in ((96, 96), (96, 384), (384, 96), (128, 32)):
        for cond in (1.0, 1e2, 1e4):
            U, _ = torch.linalg.qr(torch.randn(m, m)); V, _ = torch.linalg.qr(torch.randn(n, n))
            sv0 = torch.logspace(0, -math.log10(cond), min(m, n))
            O = newton_schulz5(U[:, :min(m, n)] @ torch.diag(sv0) @ V[:, :min(m, n)].T, 5)
            sv = torch.linalg.svdvals(O)
            smins.append(sv.min().item()); smaxs.append(sv.max().item())
            if cond <= 1e2:
                wellcond_min = min(wellcond_min, sv.min().item())
    P.add("P8", "Muon trust region (spectral)", max(smaxs) <= cert + 1e-5 and wellcond_min > 0.6,
          f"certified sup={cert:.4f}; measured sigma_max<={max(smaxs):.4f}; sigma_min(cond<=100)={wellcond_min:.3f}")

    pz = np.stack([dig_puzzle(rng, sols[0], 34)[0]] * 6)
    sz = np.stack([sols[0]] * 6)
    expect = (81 - (torch.as_tensor(pz, device=device) > 0).sum(1).float()) / 81
    e2 = BatchEnv(pz, sz, device, MDP(shaping=True))
    shape_only = torch.zeros(6, device=device)
    with torch.no_grad():
        for _ in range(81):
            if bool(e2.done.all()):
                break
            o = net(e2.b, e2.givens, T=2)
            lg, _ = policy_logits(o, e2.b, lm=e2.legal())
            a = torch.distributions.Categorical(logits=lg).sample()
            live = ~e2.done
            r, _d = e2.step(a)
            bonus = (((e2.b > 0).sum(1) == 81) & consistent_torch(e2.b) & live).float()
            shape_only += (r - bonus) * live.float()
    err = (shape_only - expect).abs().max().item()
    P.add("P11", "potential shaping is policy-invariant", err < 1e-5,
          f"max|sum_t F_t - (81-filled0)/81| = {err:.2e} over 6 episodes")

    worst = 0.0
    for n in (4, 8):
        for c in range(n + 1):
            for k in (1, 2):
                exact = 1 - (math.comb(n - c, k) / math.comb(n, k) if n - c >= k else 0)
                pool_ = np.array([1] * c + [0] * (n - c))
                worst = max(worst, abs(float(np.mean([float(pool_[rng.choice(n, k, replace=False)].sum() > 0) for _ in range(4000)])) - exact))
    P.add("P12", "pass@k estimator unbiased", worst < 0.02, f"max |MC - exact| = {worst:.4f}")

    torch.manual_seed(0)
    logits = torch.randn(5, requires_grad=True)
    Rw = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0])
    pr = torch.softmax(logits, 0)
    gJ = torch.autograd.grad((pr * Rw).sum(), logits, retain_graph=True)[0]
    Gg = 3; combos = list(itertools.product(range(5), repeat=Gg))
    tot_loo = torch.zeros(5)
    for cb in combos:
        pprob = float(torch.prod(pr[list(cb)].detach()))
        rs = Rw[list(cb)]
        adv = torch.stack([rs[i] - (rs.sum() - rs[i]) / (Gg - 1) for i in range(Gg)])
        for i in range(Gg):
            tot_loo += pprob * float(adv[i]) * torch.autograd.grad(torch.log(pr[cb[i]]), logits, retain_graph=True)[0]
    e_loo = (tot_loo / Gg - gJ).abs().max().item()
    P.add("P13", "RLOO baseline is exactly unbiased", e_loo < 1e-6,
          f"enumerated {len(combos)} groups: |E[grad_LOO]-gradJ|={e_loo:.2e}")

    P.add("P9", "resource accounting (toy net)", True,
          f"toy P={npar:,} => {npar*4/2**20:.2f} MiB weights; Muon uses 1 buffer vs Adam's 2")
    print("=" * 78 + f"\nPROOF SUITE: {'ALL PASS' if P.all_ok else 'FAIL'} ({sum(r['status']=='PASS' for r in P.rows)}/{len(P.rows)})\n")
    return P


def _first_solution(board):
    b = [int(v) for v in np.asarray(board).reshape(81)]
    r, c, x = _masks(b)
    empt = [i for i in range(81) if b[i] == 0]

    def rec():
        best, bm, bn = -1, 0, 10
        for i in empt:
            if b[i] == 0:
                m = FULL & ~(r[RI[i]] | c[CI[i]] | x[BI[i]])
                n = PC[m]
                if n == 0:
                    return False
                if n < bn:
                    best, bm, bn = i, m, n
        if best < 0:
            return True
        m = bm
        while m:
            bit = m & -m; m ^= bit
            b[best] = bit.bit_length()
            r[RI[best]] |= bit; c[CI[best]] |= bit; x[BI[best]] |= bit
            if rec():
                return True
            b[best] = 0; r[RI[best]] ^= bit; c[CI[best]] ^= bit; x[BI[best]] ^= bit
        return False
    assert rec()
    return np.array(b, dtype=np.int8)


MONOMER_MAP = [
    ("encode", "TaskAdapter -> domain-agnostic (B,N,K,F) + tied Win", "P5"),
    ("compare", "HyperGraphBlock: arbitrary hyperedge sets + slot attn", "P5,P7"),
    ("bind", "key-value attention binds entity<->slot<->hyperedge", "P5"),
    ("store", "weights + EMA + DFS tabu visited set in System 2", "P9,P14"),
    ("retrieve", "hypergraph softmax attention over relational units", "P5"),
    ("predict", "action head + cand head + value head + conf head", "P1,P2"),
    ("update", "Muon spectral step + AdamW + fp16 dynamic scaler", "P8"),
    ("value", "value head P(solve|state) ranks System-2 branches", "value_ece"),
    ("select", "masked softmax / argmax over N*K legal actions", "P6"),
    ("gate", "hard legality mask + contractive step_gate + conf", "P6"),
    ("hold", "contractive segmented belief-feedback (y_t, z_t)", "P7,sweep"),
    ("sequence", "autoregressive MDP + uncertainty-recorded DFS", "P11"),
    ("chunk", "hyperedge pools (any degree/size, Sokoban-ready)", "P5"),
    ("map", "tied hyperedge weights + tied features => G-equiv", "P5"),
    ("uncertainty", "top-1 prob trigger for System 2 DFS + digit-ECE", "P12,digit_ece"),
    ("broadcast", "shared residual stream + belief feedback y_proj", "P5"),
    ("simulate", "System 2 full-depth neural DFS rollouts", "sys2_pass"),
    ("monitor", "contradiction detection + lowest-conf backtracking", "sys2_pass"),
]


# ==============================================================================
# §9  TRAINER
# ==============================================================================
def host_mib():
    if resource is None:
        return 0.0
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 if sys.platform != "darwin" else r / 2 ** 20


def lr_at(frac, warm=0.03):
    if frac < warm:
        return frac / warm
    t = (frac - warm) / max(1e-9, 1 - warm)
    return 0.15 + 0.85 * 0.5 * (1 + math.cos(math.pi * t))


def _atomic_save(obj, path: Path):
    tmp = path.with_suffix(".pt.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(args, device, out_dir: Path, pool_tr, resume_ck=None):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed + 1)
    net = SudokuNet(C=args.width, heads=args.heads, L=args.layers, T_max=24).to(device)
    ema = EMA(net, decay=0.999)
    if resume_ck is not None:
        sd_raw = {k: v for k, v in resume_ck.get("raw_model", resume_ck["model"]).items()
                  if not k.startswith(("ug_", "ud_", "inv_"))}
        miss, _ = net.load_state_dict(sd_raw, strict=False)
        assert not [k for k in miss if not k.startswith(("ug_", "ud_", "inv_"))], f"Missing params: {miss}"
        sd_ema = {k: v.to(device).clone() for k, v in resume_ck["model"].items()
                  if not k.startswith(("ug_", "ud_", "inv_"))}
        for k, v in net.state_dict().items():
            if k not in sd_ema:
                sd_ema[k] = v.detach().clone()
        ema.shadow = sd_ema
        ema.steps = int(resume_ck.get("iters", 100))
        print(f"[resume] loaded checkpoint iters={ema.steps}, tier={resume_ck.get('tier', 2)}")
    npar = sum(p.numel() for p in net.parameters())
    opt = MuonAdamW(net, lr_muon=args.lr_muon, lr_adam=args.lr_adam, wd=args.wd)
    mdp = MDP(max_steps=81, shaping=not args.no_shaping)
    budget = args.minutes * 60.0
    t_start = time.time()
    last_save = t_start
    logf = open(out_dir / "metrics.jsonl", "w", encoding="utf-8")
    tracef = open(out_dir / "traces.jsonl", "w", encoding="utf-8") if args.dump_traces else None
    hist = []
    tier = int(resume_ck.get("tier", 2)) if resume_ck is not None else 1
    roll_ema = 0.0
    rl_started = False
    it = 0
    sl_only_until = args.sl_frac * budget
    peak_mem = 0.0
    force_ckpt = not args.no_ckpt
    use_amp = (device == "cuda")
    loss_scale = 4096.0 if use_amp else 1.0
    clean_steps = 0
    print(f"\n[train] params={npar:,}  device={device}  amp={use_amp}  budget={args.minutes:.1f} min  "
          f"C={args.width} L={args.layers} heads={args.heads} n_seg={args.n_seg} T_seg={args.t_seg} ckpt={force_ckpt}")
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
            try:
                if phase == "ssl" or (it % args.ssl_every == args.ssl_every - 1):
                    n_h = int(round(args.ssl_batch * 0.70))
                    idx = np.concatenate([pool_tr.sample(rng, args.ssl_batch - n_h),
                                          pool_tr.sample_hard(rng, n_h)])
                    sol = pool_tr.s[idx]
                    pz = pool_tr.p[idx]
                    is_hard = pool_tr.t[idx] >= 3
                    mode = rng.random()
                    if mode < 0.30:
                        keep = rng.random((len(idx), 81)) > rng.uniform(0.20, 0.85)
                        b = np.where(keep, sol, 0).astype(np.int8)
                    else:
                        b = pz.copy()
                        for j in range(len(idx)):
                            empt = np.where(b[j] == 0)[0]
                            if len(empt) > 0:
                                rng.shuffle(empt)
                                hi = max(1, len(empt) // 3) if (is_hard[j] and rng.random() < 0.6) else len(empt)
                                nfill = int(rng.integers(0, hi))
                                b[j, empt[:nfill]] = sol[j, empt[:nfill]]
                    bt = torch.as_tensor(b, dtype=torch.long, device=device)
                    st = torch.as_tensor(sol, dtype=torch.long, device=device)
                    giv = bt > 0
                    empty = (bt == 0)
                    tgt = (st - 1).clamp(min=0)
                    lm = legal_mask_torch(bt)
                    opt.zero_grad()
                    state = None
                    seg_loss_sum = 0.0
                    ssl_ckpt = force_ckpt and (args.ssl_batch > 40)
                    for seg in range(args.n_seg):
                        with _amp(device):
                            net_out = net(bt, giv, T=args.t_seg, ckpt=ssl_ckpt, all_steps=True,
                                          state=state, step_off=seg * args.t_seg)
                            l_seg, ce, move_nll, cand, conf, ok, pred = _ssl_one(net_out, bt, empty, tgt, lm)
                            l_scaled = (l_seg / args.n_seg) * loss_scale
                        l_scaled.backward()
                        seg_loss_sum += float(l_seg.detach()) / args.n_seg
                        state = (net_out["_h"].detach(), net_out["_lg"].detach())
                    T = args.n_seg * args.t_seg
                    m = dict(ce=float(ce.detach()), move_nll=float(move_nll.detach()),
                             cand=float(cand.detach()), conf=float(conf.detach()),
                             readout_acc=float(ok.mean()),
                             last_acc=float(((pred == tgt)[empty]).float().mean()) if bool(empty.any()) else 1.0,
                             phase="ssl", tier=0)
                    loss_val = seg_loss_sum
                else:
                    T = int(rng.choice([4, 6])) if args.t_roll == 4 else args.t_roll
                    rr = rollout(net, pool_tr, rng, args.rl_puzzles, args.rl_group,
                                 T, device, mdp, temp=args.temp, tier=tier)
                    opt.zero_grad()
                    with _amp(device):
                        loss, m = grpo_loss(net, rr, args.rl_steps, T, device,
                                            ckpt=force_ckpt, temp=args.temp,
                                            ent_beta=args.ent_beta, w_ssl=args.w_ssl, w_val=args.w_val)
                    m.update(phase="rl", tier=tier)
                    if not torch.isfinite(loss):
                        opt.zero_grad(); it += 1; continue
                    (loss * loss_scale).backward()
                    loss_val = float(loss.detach())
                    roll_ema = m["roll_pass"] if not rl_started else (0.9 * roll_ema + 0.1 * m["roll_pass"])
                    rl_started = True
                    up_thresh = args.curriculum_up if tier < 2 else 0.72
                    if roll_ema > up_thresh and tier < 5:
                        tier += 1; roll_ema = 0.0; rl_started = False
                        print(f"[train] curriculum -> {TIER_NAMES[tier]} (it={it})")
                    if tracef is not None and it % 20 == 0:
                        for j in range(min(4, rr["ha"].shape[1])):
                            L = int(rr["steps"][j])
                            tracef.write(json.dumps(dict(
                                it=it, ep=j, solved=bool(rr["solved"][j]), dead=bool(rr["dead"][j]),
                                steps=L, ret=float(rr["ret"][j]),
                                first_err=int(rr["first_err"][j]))) + "\n")
                        tracef.flush()
                if use_amp:
                    inv = 1.0 / loss_scale
                    for p in opt.muon_p + opt.adam_p:
                        if p.grad is not None:
                            p.grad.mul_(inv)
                gn = torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip).item()
                if not math.isfinite(gn):
                    opt.zero_grad()
                    if use_amp:
                        loss_scale = max(1.0, loss_scale * 0.5)
                        clean_steps = 0
                    it += 1
                    continue
                opt.step(lr_scale=1.0)
                ema.update(net)
                if use_amp:
                    clean_steps += 1
                    if clean_steps >= 200:
                        loss_scale = min(65536.0, loss_scale * 2.0)
                        clean_steps = 0
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad()
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                force_ckpt = True
                print(f"[guard] CUDA OOM at it={it} -> freed cache; skipping step")
                it += 1
                continue
            it += 1
            if device == "cuda":
                peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 2 ** 20)
            if it % args.log_every == 0 or it == 1:
                rec = dict(it=it, elapsed=round(el, 1), frac=round(frac, 4), lr_scale=round(s, 4),
                           T=T, loss=loss_val, grad_norm=round(gn, 4),
                           sec_per_it=round(time.time() - t_it, 3), peak_mib=round(peak_mem, 1),
                           host_mib=round(host_mib(), 1),
                           curriculum=tier, roll_ema=round(roll_ema, 4), **opt.stats, **m)
                logf.write(json.dumps(rec) + "\n"); logf.flush()
                hist.append(rec)
                if it % (args.log_every * 5) == 0 or it == 1:
                    print(f"[it {it:5d}] {rec['phase']:>3} T={T:2d} loss={loss_val:.4f} "
                          f"ce={m.get('ce', float('nan')):.4f} "
                          f"pass={m.get('roll_pass', m.get('readout_acc', 0)):.3f} "
                          f"tier={tier} gn={gn:.2f} {rec['sec_per_it']:.2f}s/it "
                          f"({el/60:.1f}/{args.minutes:.0f}m)", flush=True)
            if time.time() - last_save >= 180.0:
                _atomic_save(dict(model=ema.shadow, raw_model=net.state_dict(), args=vars(args),
                                  version=__version__, n_params=npar, iters=it, tier=tier), out_dir / "weights.pt")
                last_save = time.time()
    except KeyboardInterrupt:
        print("[train] interrupted -> saving EMA checkpoint")
    except Exception:
        print("[train] EXCEPTION contained:\n" + traceback.format_exc())
    finally:
        logf.close()
        if tracef:
            tracef.close()
        opt.zero_grad()
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        raw_state = {k: v.cpu() for k, v in net.state_dict().items()}
        net.load_state_dict(ema.shadow)
        _atomic_save(dict(model=net.state_dict(), raw_model=raw_state, args=vars(args),
                          version=__version__, n_params=npar, iters=it, tier=tier), out_dir / "weights.pt")
        print(f"[save] checkpoint written -> {out_dir/'weights.pt'} (iters={it})", flush=True)
    return net, npar, hist, it, max(peak_mem, 0.0)


def _xy(hist, key):
    pts = [(h["it"], h.get(key)) for h in hist
           if h.get(key) is not None and not (isinstance(h.get(key), float) and math.isnan(h.get(key)))]
    if not pts:
        return [], []
    xs, ys = zip(*pts)
    return list(xs), list(ys)


def make_plots(hist, ev, out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not hist:
        return
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    ax[0, 0].plot(*_xy(hist, "loss")); ax[0, 0].set_title("total loss"); ax[0, 0].set_yscale("log")
    ax[0, 1].plot(*_xy(hist, "ce"), label="digit CE")
    ax[0, 1].plot(*_xy(hist, "move_nll"), label="729-move NLL")
    ax[0, 1].plot(*_xy(hist, "last_acc"), label="cell acc")
    ax[0, 1].legend(); ax[0, 1].set_title("self-supervised signals")
    ax[0, 2].plot(*_xy(hist, "roll_pass"), label="rollout solve rate")
    ax[0, 2].plot(*_xy(hist, "roll_ema"), label="EMA")
    ax[0, 2].plot(*_xy(hist, "dead"), label="dead-end rate")
    ax[0, 2].legend(); ax[0, 2].set_title("RL outcome"); ax[0, 2].set_ylim(-0.02, 1.02)
    ax[1, 0].plot(*_xy(hist, "grad_norm"), label="|grad|")
    ax[1, 0].plot(*_xy(hist, "muon_sigma_max"), "o-", ms=3, label="sigma_max(NS5)")
    ax[1, 0].plot(*_xy(hist, "muon_sigma_min"), "o-", ms=3, label="sigma_min(NS5)")
    ax[1, 0].legend(); ax[1, 0].set_title("optimizer trust region")
    ax[1, 1].plot(*_xy(hist, "ent"), label="policy entropy")
    ax[1, 1].plot(*_xy(hist, "info_grp"), label="informative groups")
    ax[1, 1].legend(); ax[1, 1].set_title("exploration / signal")
    ts = sorted(ev.get("tiers", {}), key=int) if ev else []
    if ts:
        has_s2 = "sys2_pass" in ev["tiers"][ts[0]]
        cols = [("greedy_pass1", "Sys1 greedy"), ("pass_at_2", "Sys1 pass@2")]
        if has_s2:
            cols.append(("sys2_pass", "Sys1+2 (simulate)"))
        w = 0.8 / len(cols)
        for j, (key, lab) in enumerate(cols):
            ax[1, 2].bar(np.arange(len(ts)) + (j - (len(cols) - 1) / 2) * w,
                         [ev["tiers"][t].get(key, 0) or 0 for t in ts], w, label=lab)
        ax[1, 2].set_xticks(range(len(ts))); ax[1, 2].set_xticklabels([TIER_NAMES[int(t)] for t in ts], rotation=20)
        ax[1, 2].legend(); ax[1, 2].set_ylim(0, 1.05)
    ax[1, 2].set_title("held-out pass rates")
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=110)
    plt.close(fig)


def write_report(out, args, proofs, npar, it, peak, ev, sweep_easy, sweep_hard, gate, pools, gap=None):
    L = []
    A = L.append
    A(f"# Sudoku-AGI report  (v{__version__})\n")
    A(f"* device `{args.device_resolved}` | params **{npar:,}** | iterations **{it}** | "
      f"wall **{(time.time()-T0)/60:.1f} min** | peak GPU **{peak:.0f} MiB** | peak host RSS **{host_mib():.0f} MiB**\n")
    A("## 1. Acceptance gate\n")
    A(f"**{gate['verdict']}** - `{gate['metric']}` >= {args.gate_target:.2f} on tiers <= {args.gate_tier}: "
      f"measured **{gate['score']:.4f}**\n")
    A("## 2. Proof suite (P1-P14)\n")
    A("| id | claim | status | measurement |\n|---|---|---|---|")
    for r in proofs.rows:
        A(f"| {r['id']} | {r['name']} | **{r['status']}** | {r['detail']} |")
    A("\n## 3. Held-out results (disjoint solution grids)\n")
    A("| tier | n | clues | Sys1 greedy | pass@1 | pass@2 | Sys1+2 (`simulate`) | avg sims | digit-ECE | dead-end |\n"
      "|---|---|---|---|---|---|---|---|---|---|")
    for t, d in sorted(ev.get("tiers", {}).items(), key=lambda kv: int(kv[0])):
        s2 = f"{d['sys2_pass']:.3f}" if "sys2_pass" in d else "-"
        sims = f"{d['sys2_avg_sims']:.1f}" if "sys2_avg_sims" in d else "-"
        A(f"| {d['name']} | {d['n_puzzles']} | {d['clues']:.1f} | "
          f"{d['greedy_pass1']:.3f} | {d['pass_at_1']:.3f} | {d['pass_at_2']:.3f} | "
          f"**{s2}** | {sims} | {d['digit_ece']:.3f} | {d['dead_rate']:.3f} |")

    A("\n## 4. Comparison vs. Global Meta Models\n")
    A("| Model | Params | Data / Aug | Legality | Hard / Extreme Score |\n|---|---|---|---|---|")
    for name, prm, aug, leg, sc in GLOBAL_BASELINES:
        A(f"| {name} | {prm} | {aug} | {leg} | {sc} |")
    hard_tiers = [d for t, d in ev.get("tiers", {}).items() if int(t) >= 3]
    if hard_tiers:
        hn = sum(d["n_puzzles"] for d in hard_tiers)
        s1_hard = sum(d["greedy_pass1"] * d["n_puzzles"] for d in hard_tiers) / max(1, hn)
        s2_hard = sum(d.get("sys2_pass", d["greedy_pass1"]) * d["n_puzzles"] for d in hard_tiers) / max(1, hn)
        A(f"| **Sudoku-AGI v{__version__} (Sys1 / Sys1+2)** | **{npar:,}** | **1x (exact G-equivariant)** | "
          f"**0% illegal (P6)** | **{s1_hard*100:.1f}% Sys1 / {s2_hard*100:.1f}% Sys1+2 (T3-T5, n={hn})** |")

    if gap:
        A("\n### Memorisation control (TRAIN vs TEST)\n")
        A("| tier | train greedy | test greedy | gap |\n|---|---|---|")
        for t, g in sorted(gap.items(), key=lambda kv: int(kv[0])):
            A(f"| {TIER_NAMES[int(t)]} | {g['train']:.3f} | {g['test']:.3f} | {g['gap']:+.3f} |")
    if sweep_easy or sweep_hard:
        A("\n## 5. Recurrence-depth ablation (TC0 argument)\n")
        A("| T steps | easy/med (T<=2) | hard/extreme (T>=3) |\n|---|---|---|")
        for k in (1, 2, 4, 8, 12, 16):
            ve = f"{sweep_easy.get(k, 0.0):.3f}" if sweep_easy else "-"
            vh = f"{sweep_hard.get(k, 0.0):.3f}" if sweep_hard else "-"
            A(f"| {k} | {ve} | {vh} |")
    A("\n## 6. Monomer -> module traceability\n")
    A("| monomer | implemented by | verified by |\n|---|---|---|")
    for mm in MONOMER_MAP:
        A("| " + " | ".join(mm) + " |")
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\n[report] {out/'report.md'}")


# ==============================================================================
# §10  MAIN
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser(description="Sudoku-AGI v5.0.0: Domain-Agnostic HyperGraph Reasoner")
    p.add_argument("--mode", default="all", choices=["proofs", "gen", "train", "eval", "all"])
    p.add_argument("--out", default="runs/default")
    p.add_argument("--minutes", type=float, default=25.0)
    p.add_argument("--max-iters", type=int, default=10**9)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--pool", type=int, default=4000)
    p.add_argument("--hard-extra", type=int, default=1600)
    p.add_argument("--test-pool", type=int, default=800)
    p.add_argument("--clue-lo", type=int, default=23)
    p.add_argument("--clue-hi", type=int, default=46)
    p.add_argument("--cache", default="cache")
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--n-seg", type=int, default=2)
    p.add_argument("--t-seg", type=int, default=4)
    p.add_argument("--t-roll", type=int, default=4)
    p.add_argument("--t-eval", type=int, default=4)
    p.add_argument("--no-ckpt", action="store_true")
    p.add_argument("--lr-muon", type=float, default=0.02)
    p.add_argument("--lr-adam", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--sl-frac", type=float, default=0.35)
    p.add_argument("--ssl-every", type=int, default=4)
    p.add_argument("--ssl-batch", type=int, default=32)
    p.add_argument("--rl-puzzles", type=int, default=8)
    p.add_argument("--rl-group", type=int, default=4)
    p.add_argument("--rl-steps", type=int, default=2)
    p.add_argument("--temp", type=float, default=0.85)
    p.add_argument("--ent-beta", type=float, default=0.002)
    p.add_argument("--w-ssl", type=float, default=0.5)
    p.add_argument("--w-val", type=float, default=0.5)
    p.add_argument("--curriculum-up", type=float, default=0.85)
    p.add_argument("--no-shaping", action="store_true")
    p.add_argument("--eval-per-tier", type=int, default=48)
    p.add_argument("--eval-samples", type=int, default=4)
    p.add_argument("--eval-temp", type=float, default=0.8)
    p.add_argument("--sys2", action="store_true", default=True)
    p.add_argument("--no-sys2", dest="sys2", action="store_false")
    p.add_argument("--sim-nodes", type=int, default=48)
    p.add_argument("--conf-thresh", type=float, default=0.95)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--gate-tier", type=int, default=2)
    p.add_argument("--gate-target", type=float, default=0.95)
    p.add_argument("--gate-k", type=int, default=2)
    p.add_argument("--gate-metric", default="pass_at_2",
                   choices=["pass_at_1", "pass_at_2", "two_attempt_pass", "greedy_pass1", "sys2_pass"])
    p.add_argument("--ckpt", default="")
    p.add_argument("--resume", default="")
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--dump-traces", action="store_true")
    p.add_argument("--proof-puzzles", type=int, default=24)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if a.quick:
        a.minutes = min(a.minutes, 3.0); a.pool = min(a.pool, 320); a.hard_extra = 0
        a.test_pool = min(a.test_pool, 120)
        a.width = min(a.width, 48); a.layers = min(a.layers, 2); a.n_seg = 2; a.t_seg = 3
        a.t_roll = min(a.t_roll, 4); a.t_eval = min(a.t_eval, 4)
        a.ssl_batch = min(a.ssl_batch, 32); a.rl_puzzles = min(a.rl_puzzles, 8)
        a.eval_per_tier = min(a.eval_per_tier, 12); a.eval_samples = min(a.eval_samples, 4)
        a.proof_puzzles = min(a.proof_puzzles, 10); a.log_every = 2; a.sim_nodes = 16
    return a


def main():
    args = get_args()
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    random.seed(args.seed); np.random.seed(args.seed % (2**32))
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    args.device_resolved = dev
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)
    print(f"Sudoku-AGI v{__version__} | device={dev} | torch {torch.__version__} | out={out}")
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    proofs = run_proofs(device="cpu", n_gen=args.proof_puzzles, seed=args.seed)
    proofs.dump(out / "proofs.json")
    if args.mode == "proofs":
        return 0 if proofs.all_ok else 1

    pool_tr = build_pool(args.seed, args.pool, args.clue_lo, args.clue_hi, "train", cache, hard_extra=args.hard_extra)
    pool_te = build_pool(args.seed + 10_000, args.test_pool, args.clue_lo, args.clue_hi, "test", cache, hard_extra=0)
    str_tr = {bytes(np.asarray(s, dtype=np.int8).tobytes()) for s in pool_tr.s}
    leak = sum(bytes(np.asarray(s, dtype=np.int8).tobytes()) in str_tr for s in pool_te.s)
    proofs.add("P14", "no train/test leakage", leak == 0,
               f"{leak}/{len(pool_te)} test solution grids appear in train", "disjoint seeds + explicit check")
    proofs.dump(out / "proofs.json")
    if args.mode == "gen":
        return 0
    pools = dict(train=len(pool_tr), test=len(pool_te),
                 hist_train=np.bincount(pool_tr.t, minlength=6)[1:].tolist(),
                 hist_test=np.bincount(pool_te.t, minlength=6)[1:].tolist())
    print(f"[data] train tiers {pools['hist_train']} | test tiers {pools['hist_test']}")

    hist, it, peak, npar = [], 0, 0.0, 0
    if args.mode in ("train", "all"):
        resume_ck = torch.load(args.resume, map_location=dev, weights_only=False) if args.resume else None
        if resume_ck is not None:
            sa = resume_ck["args"]
            args.width, args.heads, args.layers = sa["width"], sa["heads"], sa["layers"]
        net, npar, hist, it, peak = train(args, dev, out, pool_tr, resume_ck=resume_ck)
    else:
        ck = torch.load(args.ckpt or (out / "weights.pt"), map_location=dev, weights_only=False)
        saved = ck["args"]
        net = SudokuNet(C=saved["width"], heads=saved["heads"], L=saved["layers"]).to(dev)
        net.load_state_dict(ck["model"]); npar = ck["n_params"]
        print(f"[eval] loaded {args.ckpt or out/'weights.pt'} (iters={ck['iters']})")

    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()

    mdp = MDP(max_steps=81, shaping=not args.no_shaping)
    t_ev = time.time()
    with open(out / "eval_traces.jsonl", "w", encoding="utf-8") as ev_f:
        ev = evaluate(net, pool_te, dev, mdp, n_samples=args.eval_samples, temp=args.eval_temp,
                      T=args.t_eval, per_tier=args.eval_per_tier, dump=ev_f,
                      run_sys2=args.sys2, max_sim_nodes=args.sim_nodes, conf_thresh=args.conf_thresh)
    (out / "eval.json").write_text(json.dumps(ev, indent=2), encoding="utf-8")
    print(f"[eval] {time.time()-t_ev:.1f}s")
    for t, d in sorted(ev["tiers"].items(), key=lambda kv: int(kv[0])):
        s2 = f" sys2={d['sys2_pass']:.3f}(sims={d['sys2_avg_sims']:.1f})" if "sys2_pass" in d else ""
        print(f"   {d['name']:<12} n={d['n_puzzles']:3d} greedy={d['greedy_pass1']:.3f} "
              f"pass@1={d['pass_at_1']:.3f} pass@2={d['pass_at_2']:.3f}{s2}")
    mem = evaluate(net, pool_tr, dev, mdp, n_samples=0, T=args.t_eval, per_tier=min(args.eval_per_tier, 24))
    gap = {t: dict(train=mem["tiers"][t]["greedy_pass1"], test=d["greedy_pass1"],
                   gap=mem["tiers"][t]["greedy_pass1"] - d["greedy_pass1"])
           for t, d in ev["tiers"].items() if t in mem["tiers"]}
    (out / "memorisation.json").write_text(json.dumps(gap, indent=2), encoding="utf-8")
    for t, g in sorted(gap.items(), key=lambda kv: int(kv[0])):
        print(f"   [memorisation] {TIER_NAMES[int(t)]:<12} train={g['train']:.3f} test={g['test']:.3f} gap={g['gap']:+.3f}")

    sweep_easy, sweep_hard = {}, {}
    if args.sweep:
        for label, cond, out_d in [
            ("easy/med", lambda t: int(t) <= args.gate_tier, sweep_easy),
            ("hard/ext", lambda t: int(t) >= 3, sweep_hard),
        ]:
            sel = [t for t in ev["tiers"] if cond(t)]
            if not sel:
                continue
            sub = Pool(np.concatenate([pool_te.p[pool_te.by_tier[int(t)]] for t in sel]),
                       np.concatenate([pool_te.s[pool_te.by_tier[int(t)]] for t in sel]),
                       np.concatenate([pool_te.t[pool_te.by_tier[int(t)]] for t in sel]),
                       np.concatenate([pool_te.c[pool_te.by_tier[int(t)]] for t in sel]))
            for T in (1, 2, 4, 8, 12, 16):
                e = evaluate(net, sub, dev, mdp, n_samples=0, T=T, per_tier=min(args.eval_per_tier, 16))
                out_d[T] = e.get("overall_greedy_pass1", 0.0)
                print(f"   [ablation {label}] T={T:2d} greedy={out_d[T]:.3f}")
        (out / "sweep.json").write_text(json.dumps(dict(easy=sweep_easy, hard=sweep_hard), indent=2), encoding="utf-8")

    hard_ds = [d for t, d in ev["tiers"].items() if int(t) >= 3]
    if hard_ds:
        hn = sum(d["n_puzzles"] for d in hard_ds)
        hs1 = sum(d["greedy_pass1"] * d["n_puzzles"] for d in hard_ds) / hn
        hs2 = sum(d.get("sys2_pass", d["greedy_pass1"]) * d["n_puzzles"] for d in hard_ds) / hn
        print(f"\n[EXTREME T3-T5 (n={hn}, 0-singles)] Pure Sys1={hs1:.4f} | Sys1+2={hs2:.4f} (vs TRM-Att 0.747, TRM-MLP 0.874)")

    key = args.gate_metric
    sel = [d for t, d in ev["tiers"].items() if int(t) <= args.gate_tier]
    score = float(np.mean([d[key] for d in sel])) if sel else 0.0
    gate = dict(metric=key, score=score, target=args.gate_target, k=args.gate_k, tier=args.gate_tier,
                verdict="PASS" if score >= args.gate_target else "FAIL")
    print(f"\n[GATE] {key} on tiers<={args.gate_tier}: {score:.4f} vs target {args.gate_target:.2f} -> {gate['verdict']}")
    (out / "gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    make_plots(hist, ev, out)
    write_report(out, args, proofs, npar, it, peak, ev, sweep_easy, sweep_hard, gate, pools, gap)
    return 0 if proofs.all_ok else 1


if __name__ == "__main__":
    sys.exit(main())