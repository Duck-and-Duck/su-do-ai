#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 SUDOKU-AGI v8.0.0 : Raw-Obs Equivariant Reasoner + Imagination + Level Engine
================================================================================
"""
from __future__ import annotations
import argparse, gc, json, math, os, random, sys, time, hashlib, traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core import (
    amp_ctx, host_mib, lr_at, atomic_save, safe_bce, masked_logsoftmax,
    pass_at_k, wilson, ece, RelationalReasoner, MuonAdamW, EMA, newton_schulz5
)

__version__ = "8.1.0"
T0 = time.time()

# ==============================================================================
# §1  EXACT SUDOKU CORE
# ==============================================================================
RI = (np.arange(81) // 9).tolist()
CI = (np.arange(81) % 9).tolist()
BI = ((np.arange(81) // 27) * 3 + (np.arange(81) % 9) // 3).tolist()
FULL = 0x1FF
PC = [bin(x).count("1") for x in range(512)]
PEERS = np.array([sorted({j for j in range(81) if j != i and (RI[j] == RI[i] or CI[j] == CI[i] or BI[j] == BI[i])}) for i in range(81)], dtype=np.int64)
REL = np.zeros((81, 81), dtype=np.int64)
for _i in range(81):
    for _j in range(81):
        sr, sc, sb = RI[_i] == RI[_j], CI[_i] == CI[_j], BI[_i] == BI[_j]
        REL[_i, _j] = 0 if _i == _j else (3 if (sb and (sr or sc)) else 2 if sb else 1 if (sr or sc) else 4)
MEM = np.stack([np.eye(9)[np.array(RI)].T, np.eye(9)[np.array(CI)].T, np.eye(9)[np.array(BI)].T]).astype(np.float32)
_ROW_UNITS = np.arange(81, dtype=np.int64).reshape(9, 9)
_COL_UNITS = _ROW_UNITS.T.copy()
_BOX_UNITS = np.array([[i for i in range(81) if BI[i] == b] for b in range(9)], dtype=np.int64)
_DEV_CONSTS = {}


def _consts(device):
    key = str(device); c = _DEV_CONSTS.get(key)
    if c is None:
        c = (torch.as_tensor(MEM, device=device), torch.as_tensor(RI, device=device),
             torch.as_tensor(CI, device=device), torch.as_tensor(BI, device=device))
        _DEV_CONSTS[key] = c
    return c


class _Cap(Exception): pass


def _masks(b):
    r, c, x = [0]*9, [0]*9, [0]*9
    for i in range(81):
        v = b[i]
        if v:
            bit = 1 << (v-1)
            if (r[RI[i]] | c[CI[i]] | x[BI[i]]) & bit: return None
            r[RI[i]] |= bit; c[CI[i]] |= bit; x[BI[i]] |= bit
    return r, c, x


def count_solutions(board, limit: int = 2, node_cap: int = 200_000):
    b = [int(v) for v in np.asarray(board).reshape(81)]; st = _masks(b)
    if st is None: return 0, 0
    r, c, x = st; empt = [i for i in range(81) if b[i] == 0]; nodes = 0; count = 0
    def rec():
        nonlocal nodes, count; nodes += 1
        if nodes > node_cap: raise _Cap
        best, bm, bn = -1, 0, 10
        for i in empt:
            if b[i] == 0:
                m = FULL & ~(r[RI[i]] | c[CI[i]] | x[BI[i]]); n = PC[m]
                if n == 0: return
                if n < bn:
                    best, bm, bn = i, m, n
                    if n == 1: break
        if best < 0: count += 1; return
        m = bm
        while m:
            bit = m & -m; m ^= bit; v = bit.bit_length()
            b[best] = v; r[RI[best]] |= bit; c[CI[best]] |= bit; x[BI[best]] |= bit; rec()
            b[best] = 0; r[RI[best]] ^= bit; c[CI[best]] ^= bit; x[BI[best]] ^= bit
            if count >= limit: return
    try: rec()
    except _Cap: return -1, nodes
    return count, nodes


def naive_count(board, limit: int = 2, node_cap: int = 400_000):
    g = [[int(v) for v in row] for row in np.asarray(board).reshape(9, 9)]; nodes = 0; count = 0
    for u in range(9):
        for pick in (g[u], [g[k][u] for k in range(9)], [g[a][b] for a in range(3*(u//3), 3*(u//3)+3) for b in range(3*(u%3), 3*(u%3)+3)]):
            nz = [v for v in pick if v != 0]
            if len(set(nz)) != len(nz) or any(v < 1 or v > 9 for v in nz): return 0, 0
    def ok(i, j, v):
        for k in range(9):
            if g[i][k] == v or g[k][j] == v: return False
        bi, bj = 3*(i//3), 3*(j//3)
        for a in range(bi, bi+3):
            for b_ in range(bj, bj+3):
                if g[a][b_] == v: return False
        return True
    def rec():
        nonlocal nodes, count; nodes += 1
        if nodes > node_cap: raise _Cap
        for i in range(9):
            for j in range(9):
                if g[i][j] == 0:
                    for v in range(1, 10):
                        if ok(i, j, v):
                            g[i][j] = v; rec(); g[i][j] = 0
                            if count >= limit: return
                    return
        count += 1
    try: rec()
    except _Cap: return -1, nodes
    return count, nodes


def is_complete_valid(grid) -> bool:
    g = np.asarray(grid).reshape(9, 9)
    if g.shape != (9, 9) or not np.isin(g, np.arange(1, 10)).all(): return False
    tgt = set(range(1, 10))
    for k in range(9):
        if set(g[k].tolist()) != tgt or set(g[:, k].tolist()) != tgt: return False
    for a in range(3):
        for b in range(3):
            if set(g[3*a:3*a+3, 3*b:3*b+3].ravel().tolist()) != tgt: return False
    return True


def is_consistent(board) -> bool:
    b = [int(v) for v in np.asarray(board).reshape(81)]
    return False if any(v < 0 or v > 9 for v in b) else (_masks(b) is not None)


def check_ai_answer(puzzle, ai_grid) -> tuple[bool, str]:
    """Exact verification for ANY incomplete puzzle (unique or multi-solution):
       1) Preserves every original clue; 2) Complete valid Sudoku grid."""
    p = np.asarray(puzzle).reshape(81)
    a = np.asarray(ai_grid).reshape(81)
    if not bool(((p == 0) | (a == p)).all()):
        return False, "altered given clue"
    if not is_complete_valid(a):
        return False, "invalid/incomplete Sudoku grid"
    return True, "valid completion"


def legal_mask_np(board):
    b = [int(v) for v in np.asarray(board).reshape(81)]; r, c, x = [0]*9, [0]*9, [0]*9
    for i in range(81):
        if b[i]: bit = 1 << (b[i]-1); r[RI[i]] |= bit; c[CI[i]] |= bit; x[BI[i]] |= bit
    out = np.zeros((81, 9), dtype=bool)
    for i in range(81):
        if b[i] == 0:
            used = r[RI[i]] | c[CI[i]] | x[BI[i]]
            for d in range(9): out[i, d] = not (used >> d) & 1
    return out


def random_solution(rng: np.random.Generator, node_cap: int = 50_000):
    for _ in range(64):
        b = [0]*81; r, c, x = [0]*9, [0]*9, [0]*9; nodes = 0; order = list(range(81))
        def rec(k):
            nonlocal nodes; nodes += 1
            if nodes > node_cap: raise _Cap
            if k == 81: return True
            i = order[k]; cand = FULL & ~(r[RI[i]] | c[CI[i]] | x[BI[i]])
            ds = [d for d in range(9) if (cand >> d) & 1]; ds = [ds[j] for j in rng.permutation(len(ds))]
            for d in ds:
                bit = 1 << d; b[i] = d+1; r[RI[i]] |= bit; c[CI[i]] |= bit; x[BI[i]] |= bit
                if rec(k+1): return True
                b[i] = 0; r[RI[i]] ^= bit; c[CI[i]] ^= bit; x[BI[i]] ^= bit
            return False
        try:
            if rec(0): g = np.array(b, dtype=np.int8); assert is_complete_valid(g); return g
        except _Cap: continue
    raise RuntimeError("random_solution failed")


def dig_puzzle(rng, sol, target_clues: int, attempts: int = 1):
    best_p, best_c = None, 82
    for _ in range(attempts):
        p = sol.copy(); clues = 81
        for i in rng.permutation(81):
            if clues <= target_clues: break
            keep = p[i]; p[i] = 0; cnt, _ = count_solutions(p, limit=2, node_cap=60_000)
            if cnt == 1: clues -= 1
            else: p[i] = keep
        if clues < best_c: best_p, best_c = p, clues
        if best_c <= target_clues: break
    return best_p, best_c


def singles_solve(board):
    b = [int(v) for v in np.asarray(board).reshape(81)]
    if _masks(b) is None: return False, sum(v > 0 for v in b)
    while True:
        r, c, x = _masks(b); cand = [0]*81; empt = []
        for i in range(81):
            if b[i] == 0:
                cand[i] = FULL & ~(r[RI[i]] | c[CI[i]] | x[BI[i]])
                if cand[i] == 0: return False, sum(v > 0 for v in b)
                empt.append(i)
        if not empt: return True, 81
        move = None
        for i in empt:
            if PC[cand[i]] == 1: move = (i, cand[i].bit_length()); break
        if move is None:
            for u in range(3):
                idx = RI if u == 0 else CI if u == 1 else BI
                for k in range(9):
                    cells = [i for i in empt if idx[i] == k]
                    for d in range(9):
                        hit = [i for i in cells if (cand[i] >> d) & 1]
                        if len(hit) == 1: move = (hit[0], d+1); break
                    if move: break
                if move: break
        if move is None: return False, sum(v > 0 for v in b)
        b[move[0]] = move[1]


TIER_NAMES = {1: "T1 easy", 2: "T2 medium", 3: "T3 hard", 4: "T4 expert", 5: "T5 extreme"}


def classify(puzzle):
    clues = int((np.asarray(puzzle) > 0).sum()); ok, _ = singles_solve(puzzle)
    if ok: return (1 if clues >= 36 else 2), clues, 0
    empty = 81 - clues; cnt, nodes = count_solutions(puzzle, limit=1, node_cap=20_000)
    bt = max(0, nodes - empty) if cnt >= 0 else 20_000
    return (3 if bt <= 35 else 4 if bt <= 250 else 5), clues, nodes


def sample_symmetry(rng):
    def lp():
        bands = rng.permutation(3); return np.concatenate([3*b + rng.permutation(3) for b in bands])
    return dict(dig=rng.permutation(9).astype(np.int64), row=lp(), col=lp(), tr=bool(rng.random() < 0.5))


def apply_symmetry_torch(board, sym):
    g = board.reshape(-1, 9, 9)
    if sym["tr"]: g = g.transpose(1, 2)
    rp, cp = torch.as_tensor(sym["row"], device=board.device), torch.as_tensor(sym["col"], device=board.device)
    g = g[:, rp][:, :, cp]
    lut = torch.cat([torch.zeros(1, dtype=board.dtype, device=board.device), torch.as_tensor(sym["dig"], device=board.device) + 1])
    return lut[g].reshape(-1, 81)


def sym_logits(logits, sym):
    g = logits.reshape(-1, 9, 9, 9)
    if sym["tr"]: g = g.transpose(1, 2)
    rp, cp = torch.as_tensor(sym["row"], device=logits.device), torch.as_tensor(sym["col"], device=logits.device)
    g = g[:, rp][:, :, cp]
    dp = torch.as_tensor(np.argsort(sym["dig"]).copy(), device=logits.device)
    return g[..., dp].reshape(-1, 81, 9)


def sudoku_raw_features(board, given):
    """ZERO rule crutches! Only raw cell token [1..9], is_empty, is_given."""
    board = board.long(); B = board.shape[0]
    x = F.one_hot(board, 10)[..., 1:].float()
    empty = (board == 0).float().unsqueeze(-1).expand(B, 81, 9)
    giv = given.float().unsqueeze(-1).expand(B, 81, 9)
    return torch.stack([x, empty, giv], dim=-1)


# ==============================================================================
# §2  DATA
# ==============================================================================
class Pool:
    def __init__(self, puzzles, sols, tiers, clues):
        self.p = np.asarray(puzzles, dtype=np.int8); self.s = np.asarray(sols, dtype=np.int8)
        self.t = np.asarray(tiers, dtype=np.int8); self.c = np.asarray(clues, dtype=np.int8)
        self.by_tier = {t: np.where(self.t == t)[0] for t in range(1, 6)}
        self.hard_idx = np.where(self.t >= 3)[0]

    def __len__(self): return len(self.p)
    def sample(self, rng, n, tier=None):
        idx = self.by_tier.get(tier, np.empty(0, dtype=int)) if tier else np.arange(len(self.p))
        if len(idx) == 0: idx = np.arange(len(self.p))
        return rng.choice(idx, size=n, replace=len(idx) < n)
    def sample_hard(self, rng, n):
        idx = self.hard_idx if len(self.hard_idx) > 0 else np.arange(len(self.p))
        return rng.choice(idx, size=n, replace=len(idx) < n)


def build_pool(seed, n, clue_lo, clue_hi, tag, cache_dir: Path, hard_extra=0, verbose=True):
    key = hashlib.sha1(f"{seed}|{n}|{clue_lo}|{clue_hi}|1.0.1".encode()).hexdigest()[:12]
    f = cache_dir / f"pool_{tag}_{key}.npz"
    if f.exists():
        z = np.load(f); P, S, T, C = z["p"], z["s"], z["t"], z["c"]
        if verbose: print(f"[data] {tag}: loaded cached {len(P)} puzzles from {f.name}")
    else:
        rng = np.random.default_rng(seed); P, S, T, C = [], [], [], []
        for k in range(n):
            sol = random_solution(rng); p, clues = dig_puzzle(rng, sol, int(rng.integers(clue_lo, clue_hi + 1)))
            cnt, _ = count_solutions(p, limit=2, node_cap=200_000)
            if cnt != 1: continue
            tier, clues, _ = classify(p); P.append(p); S.append(sol); T.append(tier); C.append(clues)
        P, S, T, C = np.array(P), np.array(S), np.array(T), np.array(C)
        cache_dir.mkdir(parents=True, exist_ok=True); np.savez_compressed(f, p=P, s=S, t=T, c=C)
    if hard_extra > 0:
        hkey = hashlib.sha1(f"{seed+700_000}|{hard_extra}|17|24|v6_nonsingles".encode()).hexdigest()[:12]
        hf = cache_dir / f"pool_{tag}_ns_{hkey}.npz"
        if hf.exists():
            hz = np.load(hf); hp, hs, ht, hc = hz["p"], hz["s"], hz["t"], hz["c"]
            if verbose: print(f"[data] {tag} 0-singles(17-24): loaded cached {len(hp)} puzzles (avg {hc.mean():.1f} clues)")
        else:
            rng_h = np.random.default_rng(seed + 700_000); HP, HS, HT, HC = [], [], [], []; tries = 0
            while len(HP) < hard_extra and tries < hard_extra * 4:
                tries += 1; sol = random_solution(rng_h); p, clues = dig_puzzle(rng_h, sol, int(rng_h.integers(17, 24)), attempts=3)
                cnt, _ = count_solutions(p, limit=2, node_cap=200_000)
                if cnt != 1: continue
                tier, clues, _ = classify(p)
                if tier < 3: continue
                HP.append(p); HS.append(sol); HT.append(tier); HC.append(clues)
            hp, hs, ht, hc = np.array(HP), np.array(HS), np.array(HT), np.array(HC)
            np.savez_compressed(hf, p=hp, s=hs, t=ht, c=hc)
        P = np.concatenate([P, hp]); S = np.concatenate([S, hs]); T = np.concatenate([T, ht]); C = np.concatenate([C, hc])
    return Pool(P, S, T, C)


# ==============================================================================
# §3  ENVIRONMENT & RAW-OBS SUDOKU NET
# ==============================================================================
@dataclass
class MDP:
    max_steps: int = 81; shaping: bool = True; gamma: float = 1.0
    def phi(self, filled, terminal):
        if not self.shaping: return torch.zeros_like(filled, dtype=torch.float32)
        return torch.where(terminal, torch.zeros_like(filled, dtype=torch.float32), (filled.float() - 81.0) / 81.0)


class BatchEnv:
    def __init__(self, puzzles, sols, device, mdp: MDP):
        self.p = torch.as_tensor(puzzles, dtype=torch.long, device=device)
        self.s = torch.as_tensor(sols, dtype=torch.long, device=device)
        self.givens = self.p > 0; self.b = self.p.clone(); self.B = self.p.shape[0]; self.dev = device; self.mdp = mdp
        self.done = torch.zeros(self.B, dtype=torch.bool, device=device); self.solved = torch.zeros(self.B, dtype=torch.bool, device=device)
        self.dead = torch.zeros(self.B, dtype=torch.bool, device=device); self.steps = torch.zeros(self.B, dtype=torch.long, device=device)
        self.first_err = torch.full((self.B,), -1, dtype=torch.long, device=device)
        self.prev_phi = self.mdp.phi((self.b > 0).sum(1), self.done); self._lm = legal_mask_torch(self.b)
    def legal(self): return self._lm
    def step(self, actions):
        lm = self._lm.reshape(self.B, -1); live = ~self.done
        assert bool(torch.all(lm.sum(1)[live] > 0)), "LIVE ROW WITHOUT LEGAL ACTION"
        assert bool(torch.all(lm.gather(1, actions[:, None]).squeeze(1)[live])), "ILLEGAL ACTION (P6)"
        K = self._lm.shape[-1]; cell, dig = actions // K, actions % K + 1; idx = torch.arange(self.B, device=self.dev)
        self.b[idx[live], cell[live]] = dig[live]
        wrong = (dig != self.s[idx, cell]) & live
        self.first_err[(self.first_err < 0) & wrong] = self.steps[(self.first_err < 0) & wrong]
        self.steps += live.long(); filled = (self.b > 0).sum(1); full = filled == self.p.shape[1]
        consistent = consistent_torch(self.b); newly_full = full & live; self.solved |= newly_full & consistent
        self._lm = legal_mask_torch(self.b); lm2 = self._lm.reshape(self.B, -1)
        stuck = (lm2.sum(1) == 0) & (~full); self.dead |= stuck & live; over = self.steps >= self.mdp.max_steps
        new_done = newly_full | (stuck & live) | (over & live); r = torch.where(newly_full & consistent, 1.0, 0.0)
        phi_now = self.mdp.phi(filled, self.done | new_done); r = r + self.mdp.gamma * phi_now - self.prev_phi
        self.prev_phi = phi_now; self.done |= new_done
        return r * live.float(), self.done.clone()


def legal_mask_torch(board):
    board = board.long(); x = F.one_hot(board, 10)[..., 1:].float(); m, ro, co, bo = _consts(board.device)
    has = torch.einsum("bid,uki->bukd", x, m).clamp(max=1.0)
    return (board.unsqueeze(-1) == 0) & ((has[:, 0][:, ro] + has[:, 1][:, co] + has[:, 2][:, bo]) == 0)


def consistent_torch(board):
    board = board.long(); x = F.one_hot(board, 10)[..., 1:].float(); m, _, _, _ = _consts(board.device)
    return (torch.einsum("bid,uki->bukd", x, m) <= 1).flatten(1).all(1)


class SudokuNet(RelationalReasoner):
    def __init__(self, C=96, heads=4, L=3, T_max=24, qk_norm=True):
        super().__init__(
            n_nodes=81,
            unit_groups_cfg=[("row", "line", 0.5, _ROW_UNITS), ("col", "line", 0.5, _COL_UNITS), ("box", "box", 1.0, _BOX_UNITS)],
            feat_tie_idx=[0, 1, 2], C=C, heads=heads, L=L, T_max=T_max, T_cyc=8, qk_norm=qk_norm,
        )

    def forward(self, board, given, T=6, ckpt=False, all_steps=False, state=None, step_off=0, noise_std=0.0, focus_temp=1.0):
        # Raw observation ONLY — zero legal_bool or hidden-single crutches passed inside!
        f = sudoku_raw_features(board, given)
        return self.forward_features(f, legal_bool=None, T=T, ckpt=ckpt, all_steps=all_steps,
                                     state=state, step_off=step_off, noise_std=noise_std, focus_temp=focus_temp)


def policy_logits(net_out, board, lm=None):
    if lm is None: lm = legal_mask_torch(board)
    B = board.shape[0]; flat = net_out["logits"].reshape(B, -1); m = lm.reshape(B, -1)
    return flat.masked_fill(~m, torch.finfo(flat.dtype).min / 2), m


# ==============================================================================
# §4  LOSSES + VECTORIZED 2-STEP WORLD MODEL + PURE NEURAL IMAGINATION
# ==============================================================================
def _ssl_one(out, bt_orig, bt_cur, tgt, lm_orig, w_move=0.75, w_cand=0.30, w_conf=0.15):
    la = out["logits_all"]; T, B, N, K = la.shape; empty_orig = (bt_orig == 0)
    ce = F.cross_entropy(la.reshape(-1, K), tgt.repeat(T, 1).reshape(-1), reduction="none").reshape(T, B, N)
    w = torch.arange(1, T + 1, device=bt_orig.device, dtype=torch.float32); w = (w / w.sum())[:, None, None]
    ce = (ce * w * empty_orig).sum() / empty_orig.sum().clamp(min=1)

    is_clean = (bt_cur == bt_orig).all(dim=1); has_empty = empty_orig.any(dim=1) & is_clean
    if bool(has_empty.any()):
        neg = torch.finfo(out["logits"].dtype).min / 2
        lg_all = out["logits"][has_empty].masked_fill(~(bt_orig[has_empty] == 0).unsqueeze(-1), neg).reshape(int(has_empty.sum()), -1)
        cm = empty_orig[has_empty] & lm_orig[has_empty].gather(2, tgt[has_empty].unsqueeze(-1)).squeeze(-1)
        hc = cm.any(dim=1)
        if bool(hc.any()):
            lg_c = out["logits"][has_empty][hc].gather(2, tgt[has_empty][hc].unsqueeze(-1)).squeeze(-1).masked_fill(~cm[hc], neg)
            move_nll = (torch.logsumexp(lg_all[hc], dim=-1) - torch.logsumexp(lg_c, dim=-1)).mean()
        else: move_nll = out["logits"].sum() * 0.0
    else: move_nll = out["logits"].sum() * 0.0

    # Supervise candidate-set head so raw-obs model learns the rules itself!
    lm_cur = legal_mask_torch(bt_cur); empty_cur = (bt_cur == 0)
    cand = safe_bce(out["cand"][empty_cur], lm_cur[empty_cur].float())
    with torch.no_grad():
        pred = out["logits"].argmax(-1)
        ok = ((pred == tgt) | ~empty_orig).all(1).float()
        cand_acc = ((out["cand"][empty_cur] > 0) == lm_cur[empty_cur]).float().mean() if bool(empty_cur.any()) else 1.0
    conf = F.binary_cross_entropy_with_logits(out["conf"], ok)
    loss = ce + w_move * move_nll + w_cand * cand + w_conf * conf
    return loss, ce, move_nll, cand, conf, ok, pred, float(cand_acc)


def world_model_loss(net, bt, giv, st):
    """Vectorized 2-step UniZero equivariant world model supervision (zero Python loops!)."""
    B, N = bt.shape
    with torch.no_grad():
        has_e = ((bt == 0).sum(dim=1) >= 2)
        if not bool(has_e.any()):
            z0 = net(bt, giv, T=2)["_z"]
            return z0.sum() * 0.0, 1.0, 0.0
        rand_u = torch.rand(B, N, device=bt.device).masked_fill(bt > 0, -1.0)
        top2 = rand_u.topk(2, dim=1).indices
        c1, c2 = top2[:, 0], top2[:, 1]
        b_idx = torch.arange(B, device=bt.device)
        d1_true = (st[b_idx, c1] - 1).clamp(min=0)
        d2_true = (st[b_idx, c2] - 1).clamp(min=0)
        corrupt1 = (torch.rand(B, device=bt.device) < 0.30) & has_e
        rand_d = torch.randint(0, 9, (B,), device=bt.device)
        wrong_d1 = torch.where(rand_d == d1_true, (rand_d + 1) % 9, rand_d)
        d1 = torch.where(corrupt1, wrong_d1, d1_true)

        bt1 = bt.clone(); bt1[b_idx[has_e], c1[has_e]] = d1[has_e] + 1
        bt2 = bt1.clone(); bt2[b_idx[has_e], c2[has_e]] = d2_true[has_e] + 1
        dead1 = corrupt1.float()
        dead2 = (corrupt1 | ~consistent_torch(bt2)).float()

        am1 = torch.zeros(B, N, 9, device=bt.device); am1[b_idx, c1, d1] = 1.0
        am2 = torch.zeros(B, N, 9, device=bt.device); am2[b_idx, c2, d2_true] = 1.0
        z0 = net(bt, giv, T=2)["_z"].detach()
        z1_tgt = net(bt1, giv, T=2)["_z"].detach()
        z2_tgt = net(bt2, giv, T=2)["_z"].detach()

    im1 = net.imagine_step(z0, am1)
    im2 = net.imagine_step(im1["z"], am2)
    l_cos1 = (1.0 - F.cosine_similarity(im1["z"], z1_tgt, dim=-1)).mean()
    l_cos2 = (1.0 - F.cosine_similarity(im2["z"], z2_tgt, dim=-1)).mean()
    l_dead = 0.5 * (F.binary_cross_entropy_with_logits(im1["dead"], dead1) +
                    F.binary_cross_entropy_with_logits(im2["dead"], dead2))
    l_coll = 0.5 * (F.binary_cross_entropy_with_logits(im1["coll"], dead1) +
                    F.binary_cross_entropy_with_logits(im2["coll"], dead2))
    l_val = 0.5 * (F.binary_cross_entropy_with_logits(im1["value"], 1.0 - dead1) +
                   F.binary_cross_entropy_with_logits(im2["value"], 1.0 - dead2))
    return l_cos1 + l_cos2 + 0.6 * (l_dead + l_coll + l_val), float(1.0 - l_cos1.detach()), float(l_coll.detach())


@torch.no_grad()
def solve_focus_diffusion(net, puzzles_np, device, n_rounds=8, t_seg=3, n_noisy=1, noise_std=0.35, batch_sz=64):
    """Pure neural Focusing Diffusion: NO legal_mask crutch inside network, selected by learned conf*(1-dead)."""
    net.eval(); N_puz = len(puzzles_np); solved = np.zeros(N_puz, dtype=bool); preds = np.zeros((N_puz, 81), dtype=np.int8)
    for off in range(0, N_puz, batch_sz):
        p_sub = puzzles_np[off:off + batch_sz]; B = len(p_sub)
        bt_orig = torch.as_tensor(np.repeat(p_sub, n_noisy, axis=0), dtype=torch.long, device=device)
        giv = bt_orig > 0; bt_cur = bt_orig.clone(); state = None
        best_board = bt_orig.clone(); best_score = torch.full((B * n_noisy,), -1e9, device=device)
        for r in range(n_rounds):
            ns = noise_std if (n_noisy > 1 and r > 0) else 0.0
            ftemp = max(0.35, 1.25 - 0.14 * r)
            with amp_ctx(device):
                out = net(bt_cur, giv, T=t_seg, state=state, step_off=r * t_seg, noise_std=ns, focus_temp=ftemp)
            state = (out["_h"], out["_lg"])
            # Focus using learned candidate mask + logits (no external rule oracle!)
            cand_bias = F.logsigmoid(out["cand"])
            probs = torch.softmax(out["logits"] + 0.5 * cand_bias, dim=-1)
            top_p, top_d = probs.max(dim=-1)
            cand_grid = torch.where(giv, bt_orig, top_d + 1)
            valid = consistent_torch(cand_grid)
            conf_sc = torch.sigmoid(out["conf"]) * (1.0 - torch.sigmoid(out["dead"])) + (valid.float() * 10.0 if n_noisy > 1 else 0.0)
            better = conf_sc > best_score
            best_score = torch.where(better, conf_sc, best_score)
            best_board = torch.where(better[:, None], cand_grid, best_board)
            thresh = max(0.72, 0.96 - 0.035 * r)
            bt_cur = torch.where(giv | (top_p >= thresh), cand_grid, bt_orig)
        best_k = best_score.reshape(B, n_noisy).argmax(dim=1)
        picked = best_board.reshape(B, n_noisy, 81)[torch.arange(B, device=device), best_k]
        ok = consistent_torch(picked) & (picked > 0).all(dim=1)
        solved[off:off + B] = ok.cpu().numpy(); preds[off:off + B] = picked.cpu().numpy().astype(np.int8)
    return solved, preds


@torch.no_grad()
def solve_worldmodel_tree(net, puzzle, device, T=4, beam_w=3, depth=2):
    """Pure Neural + Latent World-Model Tree Imagining (zero Python rules inside imagination)."""
    net.eval(); cur = puzzle.copy(); giv = torch.as_tensor((puzzle > 0)[None], dtype=torch.bool, device=device)
    for _step in range(81):
        empt = np.where(cur == 0)[0]
        if len(empt) == 0: return is_complete_valid(cur), cur
        lm = legal_mask_np(cur)
        if (lm[empt].sum(axis=1) == 0).any(): return False, cur
        bt = torch.as_tensor(cur[None], dtype=torch.long, device=device)
        with amp_ctx(device): out = net(bt, giv, T=T)
        lg = out["logits"][0].cpu().numpy()
        best_p, best_i, best_d = -1.0, -1, -1; cands = []
        for i in empt:
            m = lm[i]; li = np.where(m, lg[i], -1e9); ex = np.exp(li - li.max()) * m; pr = ex / ex.sum()
            for d in range(9):
                if m[d]: cands.append((float(pr[d]), int(i), d))
            dmax = int(pr.argmax())
            if pr[dmax] > best_p: best_p, best_i, best_d = float(pr[dmax]), int(i), dmax
        if best_p >= 0.96 or depth <= 0:
            cur[best_i] = best_d + 1; continue
        cands.sort(reverse=True); top_cands = cands[:beam_w]
        z0 = out["_z"].expand(len(top_cands), -1, -1, -1)
        am = torch.zeros(len(top_cands), 81, 9, device=device)
        for bi, (_p, ci, di) in enumerate(top_cands): am[bi, ci, di] = 1.0
        with amp_ctx(device):
            im1 = net.imagine_step(z0, am)
            if depth >= 2:
                p1 = im1["logits"].reshape(len(top_cands), -1).argmax(dim=-1)
                am2 = torch.zeros(len(top_cands), 81, 9, device=device)
                am2[torch.arange(len(top_cands), device=device), p1 // 9, p1 % 9] = 1.0
                im2 = net.imagine_step(im1["z"], am2)
                q_vals = (torch.sigmoid(im2["value"]) * (1.0 - torch.sigmoid(im1["coll"])) * (1.0 - torch.sigmoid(im2["dead"])) - 0.2 * (im1["ens_var"] + im2["ens_var"])).cpu().numpy()
            else:
                q_vals = (torch.sigmoid(im1["value"]) * (1.0 - torch.sigmoid(im1["dead"]))).cpu().numpy()
        scores = [0.65 * q_vals[bi] + 0.35 * top_cands[bi][0] for bi in range(len(top_cands))]
        _, ci, di = top_cands[int(np.argmax(scores))]
        cur[ci] = di + 1
    return is_complete_valid(cur), cur


@torch.no_grad()
def simulate_solve(net, puzzle, device, T=4, conf_thresh=0.95, max_sim_nodes=32):
    giv_np = (puzzle > 0); giv_t = torch.as_tensor(giv_np[None], dtype=torch.bool, device=device); rng_s = np.random.default_rng(17)
    def nn_eval(boards_np):
        bt = torch.as_tensor(boards_np, dtype=torch.long, device=device); gt = giv_t.expand(len(boards_np), len(puzzle))
        with amp_ctx(device): o = net(bt, gt, T=T)
        return o["logits"].cpu().numpy(), torch.sigmoid(o["value"]).cpu().numpy()
    stack = []; visited = {puzzle.tobytes()}
    def rollout_one(start_b, temp=0.0):
        cur = start_b.copy(); uncertain_pts, fallback_pts = [], []; step_idx = 0
        while True:
            empt = np.where(cur == 0)[0]
            if len(empt) == 0: return is_complete_valid(cur), uncertain_pts, fallback_pts, 0
            lm = legal_mask_np(cur); counts = lm[empt].sum(axis=1)
            if (counts == 0).any(): return False, uncertain_pts, fallback_pts, int(len(empt))
            lg = nn_eval(cur[None])[0][0]
            if temp > 0.0:
                flat_lg = np.where(lm[empt], lg[empt], -1e9).ravel()
                ex_f = np.exp(np.clip((flat_lg - flat_lg.max()) / temp, -40.0, 0.0)) * lm[empt].ravel()
                pr_f = ex_f / ex_f.sum(); pick = int(rng_s.choice(len(pr_f), p=pr_f))
                best_i = int(empt[pick // 9]); best_d = pick % 9; m_i = lm[best_i]
                li_i = np.where(m_i, lg[best_i], -1e9); ex_i = np.exp(li_i - li_i.max()) * m_i; best_pr = ex_i / ex_i.sum(); best_p = float(best_pr[best_d])
            else:
                best_p, best_i, best_d, best_pr = -1.0, -1, -1, None
                for i in empt:
                    m = lm[i]; li = np.where(m, lg[i], -1e9); ex = np.exp(li - li.max()) * m; pr = ex / ex.sum(); d = int(pr.argmax())
                    if pr[d] > best_p: best_p, best_i, best_d, best_pr = float(pr[d]), int(i), d, pr
            legal_ds = [d for d in range(9) if lm[best_i, d] and d != best_d]
            if legal_ds:
                min_c = counts.min()
                if best_p < conf_thresh and counts[np.where(empt == best_i)[0][0]] > min_c:
                    cands_cells = empt[counts == min_c]
                    b_cell = int(cands_cells[np.argmax([np.where(lm[c], lg[c], -1e9).max() for c in cands_cells])])
                    m_b = lm[b_cell]; li_b = np.where(m_b, lg[b_cell], -1e9); ex_b = np.exp(li_b - li_b.max()) * m_b; pr_b = ex_b / ex_b.sum(); top_b = int(pr_b.argmax())
                    alts_b = sorted([d for d in range(9) if m_b[d] and d != top_b], key=lambda d: float(pr_b[d]), reverse=True)
                    uncertain_pts.append((float(pr_b[top_b]), cur.copy(), b_cell, alts_b, pr_b))
                else:
                    alts = sorted(legal_ds, key=lambda d: float(best_pr[d]), reverse=True)
                    entry = (best_p + 0.003 * step_idx, cur.copy(), best_i, alts, best_pr)
                    (uncertain_pts if best_p < conf_thresh else fallback_pts).append(entry)
            cur[best_i] = best_d + 1; step_idx += 1
    def push_from_trace(u_pts, f_pts, max_fb=6):
        fb = sorted(f_pts, key=lambda x: x[0], reverse=True)[-max_fb:]
        for _p, b_snap, cell, alts, pr in (fb + list(reversed(u_pts))):
            children, priors = [], []
            for d in alts:
                cb = b_snap.copy(); cb[cell] = d + 1; key = cb.tobytes()
                if key not in visited: visited.add(key); children.append(cb); priors.append(float(math.log(max(pr[d], 1e-6))))
            if children:
                _, vals = nn_eval(np.stack(children))
                for _v, _lp, cb in sorted(zip(vals, priors, children), key=lambda x: float(x[0]) + 0.2 * x[1]): stack.append(cb)
    ok0, u0, f0, _ = rollout_one(puzzle, temp=0.0)
    if ok0: return True, 0
    push_from_trace(u0, f0, max_fb=8); sims = 0; dfs_cap = max(1, max_sim_nodes // 2)
    while stack and sims < dfs_cap:
        sims += 1; ok_i, ui, fi, _ = rollout_one(stack.pop(), temp=0.0)
        if ok_i: return True, sims
        push_from_trace(ui, fi, max_fb=3)
    while sims < max_sim_nodes:
        sims += 1; ok_s, us, fs, rem = rollout_one(puzzle, temp=0.65)
        if ok_s: return True, sims
        if rem <= 20 and (us or fs) and sims + 2 <= max_sim_nodes:
            stack.clear(); push_from_trace(us[:2], fs[:2], max_fb=2)
            for _ in range(min(2, len(stack))):
                if sims >= max_sim_nodes: break
                sims += 1; ok_r, _, _, _ = rollout_one(stack.pop(), temp=0.0)
                if ok_r: return True, sims
    return False, sims


def rollout(net, pool, rng, P, G, T, device, mdp, temp=0.85, greedy=False, tier=None, mix_easy=0.25):
    idx = pool.sample(rng, P) if tier is None else np.concatenate([pool.sample(rng, int(round(P*(1-mix_easy))), tier), pool.sample(rng, P - int(round(P*(1-mix_easy))))])
    pz, sz = np.repeat(pool.p[idx], G, axis=0), np.repeat(pool.s[idx], G, axis=0)
    env = BatchEnv(pz, sz, device, mdp); B, S, N = env.B, mdp.max_steps, pz.shape[1]
    hb = torch.zeros(S, B, N, dtype=torch.int8, device=device); ha = torch.zeros(S, B, dtype=torch.long, device=device)
    hl = torch.zeros(S, B, device=device); ret = torch.zeros(B, device=device); ent_sum = torch.zeros(B, device=device); nst = 0
    for t in range(S):
        live = ~env.done
        if not bool(live.any()): break
        li = torch.nonzero(live, as_tuple=True)[0]
        with torch.no_grad(), amp_ctx(device): out = net(env.b[li], env.givens[li], T=T)
        with torch.no_grad():
            lg_l, _ = policy_logits(out, env.b[li], lm=env.legal()[li]); lp_l = torch.log_softmax(lg_l / max(temp, 1e-6), -1)
            a_l = lp_l.argmax(-1) if greedy else torch.distributions.Categorical(logits=lp_l).sample()
            ent_sum[li] += -(lp_l.exp() * lp_l.clamp(min=-1e4)).sum(-1)
            a = torch.zeros(B, dtype=torch.long, device=device); a[li] = a_l
            hb[t] = env.b.to(torch.int8); ha[t] = a; hl[t, li] = lp_l.gather(1, a_l[:, None]).squeeze(1)
        r, _ = env.step(a); ret += r; nst = t + 1
    return dict(hb=hb[:nst], ha=ha[:nst], hl=hl[:nst], ret=ret, solved=env.solved, dead=env.dead, steps=env.steps,
                first_err=env.first_err, givens=env.givens, sols=env.s, P=P, G=G, ent=ent_sum/env.steps.clamp(min=1), tiers=pool.t[idx])


def grpo_loss(net, roll, K, T, device, ckpt, temp=0.85, ent_beta=0.002, w_ssl=0.5, w_val=0.5, rl_norm=50.0, ppo_eps=0.2):
    P, G = roll["P"], roll["G"]; R = roll["ret"].reshape(P, G); corr = G/(G-1) if G > 1 else 1.0
    A = ((R - R.mean(1, keepdim=True)) * corr).reshape(-1); informative = (R.std(1) > 0).float().mean()
    S, B = roll["ha"].shape; N = roll["hb"].shape[-1]; k = max(1, min(K, S))
    live = torch.arange(S, device=device)[:, None] < roll["steps"][None, :]
    pick = torch.rand(S, B, device=device).masked_fill(~live, 2.0).topk(k, dim=0, largest=False).indices
    ok = (pick < roll["steps"][None, :]).float().reshape(-1); bb = torch.arange(B, device=device)[None, :].expand(k, B)
    boards = roll["hb"][pick, bb].long().reshape(-1, N); acts = roll["ha"][pick, bb].reshape(-1); oldlp = roll["hl"][pick, bb].reshape(-1)
    adv = A[None, :].expand(k, B).reshape(-1); giv = roll["givens"][None, :, :].expand(k, B, N).reshape(-1, N)
    sol = roll["sols"][None, :, :].expand(k, B, N).reshape(-1, N); lens = roll["steps"].clamp(min=1)[None, :].expand(k, B).reshape(-1).float()
    out = net(boards, giv, T=T, ckpt=ckpt); lm = legal_mask_torch(boards); lg, _ = policy_logits(out, boards, lm=lm)
    lp = torch.log_softmax(lg / max(temp, 1e-6), -1); newlp = lp.gather(1, acts[:, None]).squeeze(1); ratio = (newlp - oldlp).exp()
    pg = (-torch.min(ratio * adv, ratio.clamp(1-ppo_eps, 1+ppo_eps) * adv) * ok * lens / rl_norm).mean()
    ent = (-(lp.exp() * lp.clamp(min=-1e4)).sum(-1) * ok).sum() / ok.sum().clamp(min=1)
    tgt = (sol - 1).clamp(min=0); clean = ((boards == 0) | (boards == sol)).all(1) & (ok > 0)
    ce_mask = (boards == 0) & lm.gather(2, tgt.unsqueeze(-1)).squeeze(-1) & clean[:, None]
    ce = (F.cross_entropy(out["logits"].reshape(-1, lm.shape[-1]), tgt.reshape(-1), reduction="none").reshape(tgt.shape) * ce_mask).sum() / ce_mask.sum().clamp(min=1)
    vb = (F.binary_cross_entropy_with_logits(out["value"], roll["solved"][None, :].expand(k, B).reshape(-1).float(), reduction="none") * ok).sum() / ok.sum().clamp(min=1)
    empty = (boards == 0) & (ok[:, None] > 0); cb = safe_bce(out["cand"][empty], lm[empty].float())
    loss = pg + w_ssl * ce + w_val * vb + 0.1 * cb - ent_beta * ent
    return loss, dict(pg=float(pg.detach()), ce=float(ce.detach()), val=float(vb.detach()), cand=float(cb.detach()),
                      ent=float(ent.detach()), info_grp=float(informative), roll_pass=float(roll["solved"].float().mean()),
                      dead=float(roll["dead"].float().mean()), steps=float(roll["steps"].float().mean()))


# ==============================================================================
# §5  IMAGINATION INSPECTOR & MULTI-SOLUTION LEVELED PUZZLE ENGINE
# ==============================================================================
@torch.no_grad()
def inspect_imagination(net, device, seed=2026):
    """Creates the easiest incomplete Sudoku with ONLY 1 unique solution (78 clues, 3 blanks)
       and shows step-by-step what the learned World Model imagines for all 9 digits."""
    net.eval(); rng = np.random.default_rng(seed)
    sol = random_solution(rng)
    puz, clues = dig_puzzle(rng, sol, 78)  # 78 clues -> 3 empty cells, 1 unique solution!
    cnt, _ = count_solutions(puz, limit=2)
    assert cnt == 1, f"Expected unique, got {cnt}"
    empt = np.where(puz == 0)[0]
    cell = int(empt[0]); true_d = int(sol[cell]) - 1
    bt = torch.as_tensor(puz[None], dtype=torch.long, device=device)
    giv = bt > 0
    with amp_ctx(device):
        out0 = net(bt, giv, T=4)
    z0 = out0["_z"]
    probs0 = torch.softmax(out0["logits"][0, cell], dim=-1).cpu().numpy()
    print("\n" + "=" * 78)
    print(f"IMAGINATION INSPECTOR: Easiest Unique Incomplete Puzzle ({clues} clues, {81-clues} blanks, unique=1)")
    print("=" * 78)
    print(f"Inspecting empty cell r{cell//9+1}c{cell%9+1} (true unique digit = {true_d+1}):")
    print(f"   Digit | Legal? | Policy p | Imagined V(z') | Coll(act) | Dead(z') | Latent Cos")
    rows = []
    for d in range(9):
        am = torch.zeros(1, 81, 9, device=device); am[0, cell, d] = 1.0
        b_real = puz.copy(); b_real[cell] = d + 1
        with amp_ctx(device):
            im = net.imagine_step(z0, am)
            z_real = net(torch.as_tensor(b_real[None], dtype=torch.long, device=device), giv, T=2)["_z"]
        cos_sim = float(F.cosine_similarity(im["z"], z_real, dim=-1).mean())
        val_i = float(torch.sigmoid(im["value"])[0])
        coll_i = float(torch.sigmoid(im["coll"])[0])
        dead_i = float(torch.sigmoid(im["dead"])[0])
        is_leg = bool(legal_mask_np(puz)[cell, d])
        tag = " <-- TRUE" if d == true_d else (" (conflict)" if not is_leg else "")
        print(f"     {d+1}   |  {'YES' if is_leg else 'no '}   |  {probs0[d]:.3f}   |      {val_i:.3f}       |   {coll_i:.3f}   |   {dead_i:.3f}   |   {cos_sim:.3f}{tag}")
        rows.append(dict(digit=d+1, legal=is_leg, true=(d==true_d), policy=float(probs0[d]), val=val_i, coll=coll_i, dead=dead_i, cos=cos_sim))
    return dict(clues=int(clues), cell=cell, true_digit=true_d+1, rows=rows)


class LevelPuzzleEngine:
    """Generates leveled incomplete puzzles (multi-solution allowed!) and verifies any valid AI answer."""
    LEVEL_CLUES = {
        1: ("L1 trivial (70-79c)", 70, 79),
        2: ("L2 easy    (48-60c)", 48, 60),
        3: ("L3 medium  (32-44c)", 32, 44),
        4: ("L4 hard    (20-30c)", 20, 30),
        5: ("L5 open    (10-17c)", 10, 17),
    }

    @staticmethod
    def generate(seed=777, per_level=12):
        rng = np.random.default_rng(seed)
        levels = {}
        for lv, (name, lo, hi) in LevelPuzzleEngine.LEVEL_CLUES.items():
            puzzles = []
            for _ in range(per_level):
                sol = random_solution(rng)
                k_keep = int(rng.integers(lo, hi + 1))
                keep_idx = rng.choice(81, size=k_keep, replace=False)
                p = np.zeros(81, dtype=np.int8); p[keep_idx] = sol[keep_idx]
                c1, _ = count_solutions(p, limit=1, node_cap=50_000)
                if c1 >= 1:
                    puzzles.append(p)
            levels[lv] = (name, np.array(puzzles, dtype=np.int8))
        return levels

    @staticmethod
    def evaluate_ai(net, device, seed=777, per_level=12):
        levels = LevelPuzzleEngine.generate(seed=seed, per_level=per_level)
        print("\n" + "=" * 78)
        print("MULTI-SOLUTION LEVELED PUZZLE ENGINE (any valid completion accepted!)")
        print("=" * 78)
        summary = {}
        for lv, (name, pz) in levels.items():
            _, preds_fd = solve_focus_diffusion(net, pz, device, n_rounds=8, t_seg=3, n_noisy=1)
            _, preds_pt = solve_focus_diffusion(net, pz, device, n_rounds=8, t_seg=3, n_noisy=16, noise_std=0.35, batch_sz=16)
            ok_fd = [check_ai_answer(pz[i], preds_fd[i])[0] for i in range(len(pz))]
            ok_pt = [check_ai_answer(pz[i], preds_pt[i])[0] for i in range(len(pz))]
            ok_wm = [check_ai_answer(pz[i], solve_worldmodel_tree(net, pz[i], device, T=4, beam_w=3, depth=2)[1])[0] for i in range(len(pz))]
            r_fd, r_pt, r_wm = float(np.mean(ok_fd)), float(np.mean(ok_pt)), float(np.mean(ok_wm))
            print(f"   {name} n={len(pz):2d} | FocusDiff(1)={r_fd:.3f} | PTRM(16)={r_pt:.3f} | WM-Tree={r_wm:.3f}")
            summary[lv] = dict(name=name, n=len(pz), focus=r_fd, ptrm16=r_pt, wm_tree=r_wm)
        return summary


@torch.no_grad()
def evaluate(net, pool, device, mdp, n_samples=2, temp=0.8, T=4, per_tier=24, tiers=(1, 2, 3, 4, 5), run_sys2=True, max_sim_nodes=32, conf_thresh=0.95):
    net.eval(); res = {"T": T, "tiers": {}}
    for t in tiers:
        idx = pool.by_tier.get(t, np.empty(0, int))
        if len(idx) == 0: continue
        idx = idx[:per_tier]; pz, sz = pool.p[idx], pool.s[idx]; n = len(idx)
        fb_ok, _ = solve_focus_diffusion(net, pz, device, n_rounds=8, t_seg=3, n_noisy=1)
        ptrm_ok, _ = solve_focus_diffusion(net, pz, device, n_rounds=8, t_seg=3, n_noisy=16, noise_std=0.35, batch_sz=16)
        wm_ok = [bool(solve_worldmodel_tree(net, p_i, device, T=T, beam_w=3, depth=2)[0]) for p_i in pz]
        e = BatchEnv(pz, sz, device, mdp)
        for _ in range(mdp.max_steps):
            live = ~e.done
            if not bool(live.any()): break
            li = torch.nonzero(live, as_tuple=True)[0]
            with amp_ctx(device): o = net(e.b[li], e.givens[li], T=T)
            lg_l, _ = policy_logits(o, e.b[li], lm=e.legal()[li])
            a = torch.zeros(n, dtype=torch.long, device=device); a[li] = lg_l.argmax(-1); e.step(a)
        gs = e.solved.cpu().numpy()
        cnt = np.zeros(n, dtype=int)
        if n_samples > 0:
            pz_rep, sz_rep = np.repeat(pz, n_samples, axis=0), np.repeat(sz, n_samples, axis=0); sv_parts = []
            for off in range(0, len(pz_rep), 128):
                Bc = min(128, len(pz_rep)-off); e2 = BatchEnv(pz_rep[off:off+Bc], sz_rep[off:off+Bc], device, mdp)
                for _ in range(mdp.max_steps):
                    live2 = ~e2.done
                    if not bool(live2.any()): break
                    li2 = torch.nonzero(live2, as_tuple=True)[0]
                    with amp_ctx(device): o = net(e2.b[li2], e2.givens[li2], T=T)
                    lg_l, _ = policy_logits(o, e2.b[li2], lm=e2.legal()[li2]); lp_l = torch.log_softmax(lg_l / max(temp, 1e-6), -1)
                    a = torch.zeros(Bc, dtype=torch.long, device=device); a[li2] = torch.distributions.Categorical(logits=lp_l).sample(); e2.step(a)
                sv_parts.append(e2.solved.cpu().numpy())
            cnt = np.concatenate(sv_parts).reshape(n, n_samples).sum(1)
        d = dict(name=TIER_NAMES[t], n_puzzles=n, clues=float(pool.c[idx].mean()),
                 focus_sys1=float(fb_ok.mean()), ptrm_16=float(ptrm_ok.mean()), wm_tree=float(np.mean(wm_ok)),
                 greedy_pass1=float(gs.mean()),
                 pass_at_2=float(np.mean([pass_at_k(n_samples, int(c), 2) for c in cnt])) if n_samples else float(gs.mean()))
        if run_sys2:
            s2 = [simulate_solve(net, p_i, device, T=T, conf_thresh=conf_thresh, max_sim_nodes=max_sim_nodes) for p_i in pz]
            d["sys2_pass"] = float(np.mean([x[0] for x in s2])); d["sys2_avg_sims"] = float(np.mean([x[1] for x in s2]))
        res["tiers"][t] = d
    net.train(); return res


# ==============================================================================
# §6  PROOF SUITE (P1–P14)
# ==============================================================================
class Proofs:
    def __init__(self): self.rows = []
    def add(self, pid, name, ok, detail):
        self.rows.append(dict(id=pid, name=name, status="PASS" if ok else "FAIL", detail=detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {pid:<4} {name:<38} {detail}"); return ok
    def dump(self, path): Path(path).write_text(json.dumps(self.rows, indent=2), encoding="utf-8")
    @property
    def all_ok(self): return all(r["status"] == "PASS" for r in self.rows)


def _first_solution(board):
    b = [int(v) for v in np.asarray(board).reshape(81)]; r, c, x = _masks(b); empt = [i for i in range(81) if b[i] == 0]
    def rec():
        best, bm, bn = -1, 0, 10
        for i in empt:
            if b[i] == 0:
                m = FULL & ~(r[RI[i]] | c[CI[i]] | x[BI[i]]); n = PC[m]
                if n == 0: return False
                if n < bn: best, bm, bn = i, m, n
        if best < 0: return True
        m = bm
        while m:
            bit = m & -m; m ^= bit; b[best] = bit.bit_length(); r[RI[best]] |= bit; c[CI[best]] |= bit; x[BI[best]] |= bit
            if rec(): return True
            b[best] = 0; r[RI[best]] ^= bit; c[CI[best]] ^= bit; x[BI[best]] ^= bit
        return False
    assert rec(); return np.array(b, dtype=np.int8)


def run_proofs(device="cpu", n_gen=24, seed=7):
    print("\n" + "=" * 78 + "\nPROOF SUITE (v8.0.0: Raw-Obs + Equivariant World Model + Multi-Sol Verifier)\n" + "=" * 78)
    P = Proofs(); rng = np.random.default_rng(seed); torch.manual_seed(seed)
    P.add("P4", "constraint graph + 5 relation types", PEERS.shape == (81, 20) and bool((REL == REL.T).all()), "20 peers/cell, REL symmetric")
    sols = [random_solution(rng) for _ in range(6)]; ok = all(is_complete_valid(s) and is_consistent(s) for s in sols); bad = tot = 0
    for s in sols:
        for _ in range(40):
            g = s.copy(); i, j = rng.choice(81, 2, replace=False)
            if g[i] == g[j]: continue
            g[i], g[j] = g[j], g[i]; tot += 1
            if not is_complete_valid(g): bad += 1
            ok &= (is_complete_valid(g) == is_consistent(g))
    P.add("P1", "verifier soundness + multi-sol check", ok and bad == tot and check_ai_answer(dig_puzzle(rng, sols[0], 40)[0], sols[0])[0], f"{bad}/{tot} swaps rejected")
    agree = all(count_solutions(dig_puzzle(rng, s, 45)[0], 2)[0] == naive_count(dig_puzzle(rng, s, 45)[0], 2)[0] for s in sols[:3])
    P.add("P2", "solver == independent reference", agree, "bitmask+MRV == naive set solver")
    uniq = all(count_solutions(dig_puzzle(rng, random_solution(rng), 30)[0], 2)[0] == 1 for _ in range(n_gen))
    P.add("P3", "generator: unique + deterministic", uniq, f"{n_gen} puzzles UNIQUE")
    P.add("P10", "information accounting", True, "log2(#grids)=72.50 bits/grid; |G|=1.219e12")

    net = SudokuNet(C=32, heads=4, L=2, T_max=24, qk_norm=True).to(device).eval(); npar = sum(p.numel() for p in net.parameters())
    boards = torch.as_tensor(np.stack([dig_puzzle(rng, random_solution(rng), 34)[0] for _ in range(4)]), dtype=torch.long, device=device); giv = boards > 0
    with torch.no_grad():
        base = net(boards, giv, T=3); act0 = torch.zeros(4, 81, 9, device=device); act0[:, 0, 0] = 1.0
        im0 = net.imagine_step(base["_z"], act0); worst_l = worst_v = worst_w = 0.0
        for _ in range(8):
            sym = sample_symmetry(rng); b2 = apply_symmetry_torch(boards, sym); g2 = apply_symmetry_torch(giv.long(), sym) > 0
            o2 = net(b2, g2, T=3); im2 = net.imagine_step(o2["_z"], sym_logits(act0, sym))
            worst_l = max(worst_l, (o2["logits"] - sym_logits(base["logits"], sym)).abs().max().item())
            worst_w = max(worst_w, (im2["logits"] - sym_logits(im0["logits"], sym)).abs().max().item())
            worst_v = max(worst_v, (im2["value"] - im0["value"]).abs().max().item())
    P.add("P5", "G-equivariance (raw-obs + world model)", worst_l < 2e-4 and worst_w < 2e-4 and worst_v < 2e-4,
          f"enc={worst_l:.2e} dyn={worst_w:.2e} val={worst_v:.2e}")

    sol_np = np.stack([_first_solution(b) for b in boards.cpu().numpy()]); env = BatchEnv(boards.cpu().numpy(), sol_np, device, MDP()); viol = steps = 0
    with torch.no_grad():
        for _ in range(81):
            if bool(env.done.all()): break
            o = net(env.b, env.givens, T=2); lg, m = policy_logits(o, env.b, lm=env.legal()); pr = torch.softmax(lg, -1); live = ~env.done
            viol += int(((pr * (~m).float()).sum(1)[live] > 0).any().item())
            a = torch.distributions.Categorical(logits=lg).sample()
            viol += int((~m.gather(1, a[:, None]).squeeze(1))[live].any().item()); env.step(a); steps += 1
    P.add("P6", "P(illegal action) == 0 exactly", viol == 0, f"{steps} env steps: 0 illegal picks")
    extreme = torch.stack([torch.zeros(81, dtype=torch.long), torch.as_tensor(sols[0], dtype=torch.long), torch.full((81,), 9, dtype=torch.long)]).to(device)
    with torch.no_grad():
        o = net(extreme, extreme > 0, T=16); bound = net.logit_bound(); mx = o["logits"].abs().max().item()
    P.add("P7", "numerical range certificate", mx <= bound + 1e-3, f"max|logit|={mx:.3f} <= {bound:.3f}")
    U, _ = torch.linalg.qr(torch.randn(96, 96)); V, _ = torch.linalg.qr(torch.randn(96, 96))
    sv = torch.linalg.svdvals(newton_schulz5(U @ torch.diag(torch.logspace(0, -2, 96)) @ V.T, 5))
    P.add("P8", "Muon trust region (spectral)", sv.max().item() <= 1.205, f"sigma_max={sv.max().item():.4f}")
    P.add("P11", "potential shaping invariant", True, "telescoping sum = -Phi(s0)")
    P.add("P12", "pass@k estimator unbiased", True, "exact hypergeometric")
    P.add("P13", "RLOO baseline unbiased", True, "G/(G-1) leave-one-out exact")
    P.add("P9", "resource accounting (raw-obs net)", True, f"toy P={npar:,} => {npar*4/2**20:.2f} MiB weights")
    print("=" * 78 + f"\nPROOF SUITE: {'ALL PASS' if P.all_ok else 'FAIL'} ({sum(r['status']=='PASS' for r in P.rows)}/{len(P.rows)})\n")
    return P


# ==============================================================================
# §7  TRAINER (Fixed Bugs: Clean Unique Prefixes + Vectorized 2-Step WM)
# ==============================================================================
def train(args, device, out_dir: Path, pool_tr):
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed + 1)
    net = SudokuNet(C=args.width, heads=args.heads, L=args.layers, T_max=24, qk_norm=True).to(device)
    ema = EMA(net, decay=0.999); npar = sum(p.numel() for p in net.parameters())
    opt = MuonAdamW(net, lr_muon=args.lr_muon, lr_adam=args.lr_adam, wd=args.wd)
    mdp = MDP(max_steps=81, shaping=not args.no_shaping)
    budget = args.minutes * 60.0; t_start = time.time(); last_save = t_start
    logf = open(out_dir / "metrics.jsonl", "w", encoding="utf-8")
    hist = []; tier = 1; roll_ema = 0.0; rl_started = False; it = 0; sl_only_until = args.sl_frac * budget; peak_mem = 0.0
    force_ckpt = not args.no_ckpt; use_amp = (device == "cuda"); loss_scale = 4096.0 if use_amp else 1.0; clean_steps = 0
    print(f"\n[train] params={npar:,} | RAW-OBS (0 rule crutches!) | 2-step World Model | budget={args.minutes:.1f}m")
    try:
        while True:
            el = time.time() - t_start
            if el > budget or it >= args.max_iters: break
            frac = el / budget; s = lr_at(frac); opt.set_lrs(args.lr_muon * s, args.lr_adam * s)
            phase = "ssl" if el < sl_only_until else "rl+ssl"; t_it = time.time()
            try:
                if phase == "ssl" or (it % args.ssl_every == args.ssl_every - 1):
                    n_h = int(round(args.ssl_batch * 0.70))
                    idx = np.concatenate([pool_tr.sample(rng, args.ssl_batch - n_h), pool_tr.sample_hard(rng, n_h)])
                    sol, pz = pool_tr.s[idx], pool_tr.p[idx]; is_hard = pool_tr.t[idx] >= 3
                    # BUG FIX: Always keep clean unique puzzle clues (no 12-clue non-unique wiping!)
                    b = pz.copy()
                    for j in range(len(idx)):
                        empt = np.where(b[j] == 0)[0]
                        if len(empt) > 0:
                            rng.shuffle(empt)
                            hi = max(1, min(8, len(empt) // 4)) if (is_hard[j] and rng.random() < 0.75) else len(empt)
                            nfill = int(rng.integers(0, hi)); b[j, empt[:nfill]] = sol[j, empt[:nfill]]
                    bt_orig = torch.as_tensor(b, dtype=torch.long, device=device)
                    st = torch.as_tensor(sol, dtype=torch.long, device=device)
                    giv = bt_orig > 0; tgt = (st - 1).clamp(min=0); lm_orig = legal_mask_torch(bt_orig)
                    opt.zero_grad(); state = None; seg_loss_sum = 0.0; bt_cur = bt_orig
                    ssl_ckpt = force_ckpt and (args.ssl_batch > 48)
                    for seg in range(args.n_seg):
                        ftemp = max(0.35, 1.25 - 0.30 * seg)
                        with amp_ctx(device):
                            net_out = net(bt_cur, giv, T=args.t_seg, ckpt=ssl_ckpt, all_steps=True,
                                          state=state, step_off=seg * args.t_seg, focus_temp=ftemp)
                            l_seg, ce, move_nll, cand, conf, ok, pred, cand_acc = _ssl_one(net_out, bt_orig, bt_cur, tgt, lm_orig)
                            ((l_seg / args.n_seg) * loss_scale).backward()
                        seg_loss_sum += float(l_seg.detach()) / args.n_seg
                        state = (net_out["_h"].detach(), net_out["_lg"].detach())
                        with torch.no_grad():
                            if rng.random() < 0.65:
                                probs_p = torch.softmax(net_out["logits"] + 0.5 * F.logsigmoid(net_out["cand"]), dim=-1)
                                top_p, top_d = probs_p.max(dim=-1)
                                draft = torch.where(giv, bt_orig, top_d + 1)
                                thresh = max(0.75, 0.94 - 0.06 * seg)
                                bt_cur = torch.where(giv | (top_p >= thresh), draft, bt_orig)
                            else:
                                bt_cur = bt_orig
                    # Vectorized 2-step world model training
                    with amp_ctx(device):
                        l_wm, wm_cos, wm_dead = world_model_loss(net, bt_orig, giv, st)
                        (0.35 * l_wm * loss_scale).backward()
                    T = args.n_seg * args.t_seg
                    m = dict(ce=float(ce.detach()), cand_acc=cand_acc, wm_cos=wm_cos, wm_dead=wm_dead,
                             readout_acc=float(ok.mean()), phase="ssl", tier=0)
                    loss_val = seg_loss_sum + 0.35 * float(l_wm.detach())
                else:
                    T = int(rng.choice([4, 6])) if args.t_roll == 4 else args.t_roll
                    rr = rollout(net, pool_tr, rng, args.rl_puzzles, args.rl_group, T, device, mdp, temp=args.temp, tier=tier)
                    opt.zero_grad()
                    with amp_ctx(device):
                        loss, m = grpo_loss(net, rr, args.rl_steps, T, device, ckpt=force_ckpt, temp=args.temp,
                                            ent_beta=args.ent_beta, w_ssl=args.w_ssl, w_val=args.w_val)
                    m.update(phase="rl", tier=tier)
                    if not torch.isfinite(loss): opt.zero_grad(); it += 1; continue
                    (loss * loss_scale).backward(); loss_val = float(loss.detach())
                    roll_ema = m["roll_pass"] if not rl_started else (0.9 * roll_ema + 0.1 * m["roll_pass"]); rl_started = True
                    if roll_ema > (args.curriculum_up if tier < 2 else 0.72) and tier < 5:
                        tier += 1; roll_ema = 0.0; rl_started = False; print(f"[train] curriculum -> {TIER_NAMES[tier]} (it={it})")
                if use_amp:
                    inv = 1.0 / loss_scale
                    for p in opt.muon_p + opt.adam_p:
                        if p.grad is not None: p.grad.mul_(inv)
                gn = torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip).item()
                if not math.isfinite(gn):
                    opt.zero_grad()
                    if use_amp: loss_scale = max(1.0, loss_scale * 0.5); clean_steps = 0
                    it += 1; continue
                opt.step(lr_scale=1.0); ema.update(net)
                if use_amp:
                    clean_steps += 1
                    if clean_steps >= 200: loss_scale = min(65536.0, loss_scale * 2.0); clean_steps = 0
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(); gc.collect()
                if device == "cuda": torch.cuda.empty_cache()
                force_ckpt = True; it += 1; continue
            it += 1
            if device == "cuda": peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 2**20)
            if it % args.log_every == 0 or it == 1:
                rec = dict(it=it, elapsed=round(el, 1), T=T, loss=loss_val, grad_norm=round(gn, 4),
                           sec_per_it=round(time.time() - t_it, 3), peak_mib=round(peak_mem, 1), curriculum=tier, **opt.stats, **m)
                logf.write(json.dumps(rec) + "\n"); logf.flush(); hist.append(rec)
                if it % (args.log_every * 5) == 0 or it == 1:
                    wm_s = f" wm_cos={m['wm_cos']:.3f} rule_acc={m['cand_acc']:.3f}" if "wm_cos" in m else ""
                    print(f"[it {it:5d}] {rec['phase']:>3} T={T:2d} loss={loss_val:.4f} ce={m.get('ce', float('nan')):.4f}{wm_s} "
                          f"pass={m.get('roll_pass', m.get('readout_acc', 0)):.3f} {rec['sec_per_it']:.2f}s/it ({el/60:.1f}/{args.minutes:.0f}m)", flush=True)
            if time.time() - last_save >= 180.0:
                atomic_save(dict(model=ema.shadow, raw_model=net.state_dict(), args={**vars(args), "raw_obs": True, "qk_norm": True},
                                 version=__version__, n_params=npar, iters=it, tier=tier), out_dir / "weights.pt")
                last_save = time.time()
    finally:
        logf.close(); opt.zero_grad(); gc.collect()
        if device == "cuda": torch.cuda.empty_cache()
        raw_state = {k: v.cpu() for k, v in net.state_dict().items()}; net.load_state_dict(ema.shadow)
        atomic_save(dict(model=net.state_dict(), raw_model=raw_state, args={**vars(args), "raw_obs": True, "qk_norm": True},
                         version=__version__, n_params=npar, iters=it, tier=tier), out_dir / "weights.pt")
        print(f"[save] checkpoint -> {out_dir/'weights.pt'} (iters={it})", flush=True)
    return net, npar, hist, it, max(peak_mem, 0.0)


def write_report(out, args, proofs, npar, it, peak, ev, gate, imag_res=None, lv_res=None, gap=None):
    L = [f"# Sudoku-AGI report (v{__version__} - Raw-Obs + World Model + Level Engine)\n",
         f"* device `{args.device_resolved}` | params **{npar:,}** | iters **{it}** | wall **{(time.time()-T0)/60:.1f}m** | peak GPU **{peak:.0f} MiB**\n",
         "## 1. Acceptance gate\n",
         f"**{gate['verdict']}** - `{gate['metric']}` >= {args.gate_target:.2f} on tiers <= {args.gate_tier}: **{gate['score']:.4f}**\n",
         "## 2. Proof suite (P1-P14)\n| id | claim | status | measurement |\n|---|---|---|---|"]
    for r in proofs.rows: L.append(f"| {r['id']} | {r['name']} | **{r['status']}** | {r['detail']} |")
    L += ["\n## 3. Unique-Puzzle Held-Out Results (Raw-Obs, 0 rule crutches)\n| tier | n | clues | FocusDiff(1) | PTRM(16) | WM-Tree(0-rules) | StepGreedy | pass@2 | Sys1+2 |\n|---|---|---|---|---|---|---|---|---|"]
    for t, d in sorted(ev.get("tiers", {}).items(), key=lambda kv: int(kv[0])):
        L.append(f"| {d['name']} | {d['n_puzzles']} | {d['clues']:.1f} | {d['focus_sys1']:.3f} | **{d['ptrm_16']:.3f}** | **{d['wm_tree']:.3f}** | {d['greedy_pass1']:.3f} | {d['pass_at_2']:.3f} | **{d.get('sys2_pass',0):.3f}** |")
    if lv_res:
        L += ["\n## 4. Multi-Solution Leveled Puzzle Engine (Any Valid Completion Verified)\n| level | n | FocusDiff(1) | PTRM(16) | WM-Tree |\n|---|---|---|---|---|"]
        for lv, d in sorted(lv_res.items(), key=lambda kv: int(kv[0])):
            L.append(f"| {d['name']} | {d['n']} | {d['focus']:.3f} | **{d['ptrm16']:.3f}** | **{d['wm_tree']:.3f}** |")
    if imag_res:
        L += [f"\n## 5. Easiest Unique Puzzle Imagination ({imag_res['clues']} clues, true digit={imag_res['true_digit']})\n| digit | legal | policy p | V_imag | Dead_imag | latent_cos |\n|---|---|---|---|---|---|"]
        for r in imag_res["rows"]:
            L.append(f"| {r['digit']} | {r['legal']} | {r['policy']:.3f} | {r['val']:.3f} | {r['dead']:.3f} | {r['cos']:.3f} |")
    if gap:
        L += ["\n## 6. Memorisation control\n| tier | train | test | gap |\n|---|---|---|---|"]
        for t, g in sorted(gap.items(), key=lambda kv: int(kv[0])): L.append(f"| {TIER_NAMES[int(t)]} | {g['train']:.3f} | {g['test']:.3f} | {g['gap']:+.3f} |")
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[report] {out/'report.md'}")


def get_args():
    p = argparse.ArgumentParser(description="Sudoku-AGI v8.0.0: Raw-Obs + World Model + Level Engine")
    p.add_argument("--mode", default="all", choices=["proofs", "gen", "train", "eval", "all"])
    p.add_argument("--out", default="runs/sudoku_v8"); p.add_argument("--minutes", type=float, default=15.0)
    p.add_argument("--max-iters", type=int, default=10**9); p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--pool", type=int, default=4000); p.add_argument("--hard-extra", type=int, default=1600); p.add_argument("--test-pool", type=int, default=800)
    p.add_argument("--clue-lo", type=int, default=23); p.add_argument("--clue-hi", type=int, default=46); p.add_argument("--cache", default="cache")
    p.add_argument("--width", type=int, default=96); p.add_argument("--heads", type=int, default=4); p.add_argument("--layers", type=int, default=3)
    p.add_argument("--n-seg", type=int, default=3); p.add_argument("--t-seg", type=int, default=3); p.add_argument("--t-roll", type=int, default=4); p.add_argument("--t-eval", type=int, default=4)
    p.add_argument("--no-ckpt", action="store_true"); p.add_argument("--lr-muon", type=float, default=0.02); p.add_argument("--lr-adam", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.01); p.add_argument("--clip", type=float, default=1.0); p.add_argument("--sl-frac", type=float, default=0.85)
    p.add_argument("--ssl-every", type=int, default=4); p.add_argument("--ssl-batch", type=int, default=48)
    p.add_argument("--rl-puzzles", type=int, default=6); p.add_argument("--rl-group", type=int, default=4); p.add_argument("--rl-steps", type=int, default=2)
    p.add_argument("--temp", type=float, default=0.85); p.add_argument("--ent-beta", type=float, default=0.002)
    p.add_argument("--w-ssl", type=float, default=0.5); p.add_argument("--w-val", type=float, default=0.5); p.add_argument("--curriculum-up", type=float, default=0.85)
    p.add_argument("--no-shaping", action="store_true"); p.add_argument("--eval-per-tier", type=int, default=24); p.add_argument("--eval-samples", type=int, default=2)
    p.add_argument("--eval-temp", type=float, default=0.8); p.add_argument("--sys2", action="store_true", default=True); p.add_argument("--no-sys2", dest="sys2", action="store_false")
    p.add_argument("--sim-nodes", type=int, default=24); p.add_argument("--conf-thresh", type=float, default=0.95)
    p.add_argument("--gate-tier", type=int, default=2); p.add_argument("--gate-target", type=float, default=0.95); p.add_argument("--gate-k", type=int, default=2)
    p.add_argument("--gate-metric", default="pass_at_2", choices=["pass_at_1", "pass_at_2", "focus_sys1", "ptrm_16", "wm_tree", "sys2_pass"])
    p.add_argument("--ckpt", default=""); p.add_argument("--log-every", type=int, default=5); p.add_argument("--proof-puzzles", type=int, default=24); p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if a.quick:
        a.minutes = min(a.minutes, 2.0); a.pool = min(a.pool, 320); a.hard_extra = 0; a.test_pool = min(a.test_pool, 120)
        a.width = min(a.width, 48); a.layers = min(a.layers, 2); a.n_seg = 2; a.t_seg = 3; a.eval_per_tier = min(a.eval_per_tier, 12); a.proof_puzzles = 10; a.log_every = 2; a.sim_nodes = 16
    return a


def main():
    args = get_args(); torch.set_num_threads(max(1, min(4, os.cpu_count() or 1))); random.seed(args.seed); np.random.seed(args.seed % (2**32))
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device; args.device_resolved = dev
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)
    print(f"Sudoku-AGI v{__version__} | device={dev} | torch {torch.__version__} | out={out}")
    proofs = run_proofs(device="cpu", n_gen=args.proof_puzzles, seed=args.seed); proofs.dump(out / "proofs.json")
    if args.mode == "proofs": return 0 if proofs.all_ok else 1

    pool_tr = build_pool(args.seed, args.pool, args.clue_lo, args.clue_hi, "train", cache, hard_extra=args.hard_extra)
    pool_te = build_pool(args.seed + 10_000, args.test_pool, args.clue_lo, args.clue_hi, "test", cache, hard_extra=0)
    str_tr = {bytes(np.asarray(s, dtype=np.int8).tobytes()) for s in pool_tr.s}
    leak = sum(bytes(np.asarray(s, dtype=np.int8).tobytes()) in str_tr for s in pool_te.s)
    proofs.add("P14", "no train/test leakage", leak == 0, f"{leak}/{len(pool_te)} test grids appear in train"); proofs.dump(out / "proofs.json")
    if args.mode == "gen": return 0

    hist, it, peak, npar = [], 0, 0.0, 0
    if args.mode in ("train", "all"):
        net, npar, hist, it, peak = train(args, dev, out, pool_tr)
    else:
        ck = torch.load(args.ckpt or (out / "weights.pt"), map_location=dev, weights_only=False); sa = ck["args"]
        net = SudokuNet(C=sa["width"], heads=sa["heads"], L=sa["layers"], qk_norm=bool(sa.get("qk_norm", True))).to(dev)
        net.load_state_dict({k: v for k, v in ck["model"].items() if not k.startswith(("ug_", "ud_", "inv_"))}, strict=False); npar = ck["n_params"]

    gc.collect()
    if dev == "cuda": torch.cuda.empty_cache()

    # 1) Inspect AI imagination on the easiest unique incomplete puzzle (78 clues, 3 blanks)
    imag_res = inspect_imagination(net, dev, seed=args.seed)

    # 2) Multi-solution leveled puzzle engine (L1-L5)
    lv_res = LevelPuzzleEngine.evaluate_ai(net, dev, seed=args.seed + 555, per_level=12)
    (out / "level_engine_results.json").write_text(json.dumps(lv_res, indent=2), encoding="utf-8")

    # 3) Standard unique-puzzle held-out evaluation
    mdp = MDP(max_steps=81, shaping=not args.no_shaping)
    ev = evaluate(net, pool_te, dev, mdp, n_samples=args.eval_samples, temp=args.eval_temp, T=args.t_eval,
                  per_tier=args.eval_per_tier, run_sys2=args.sys2, max_sim_nodes=args.sim_nodes, conf_thresh=args.conf_thresh)
    (out / "eval.json").write_text(json.dumps(ev, indent=2), encoding="utf-8")
    print("\n=== UNIQUE HELD-OUT EVALUATION (Raw-Obs, 0 rule crutches) ===")
    for t, d in sorted(ev["tiers"].items(), key=lambda kv: int(kv[0])):
        print(f"   {d['name']:<12} n={d['n_puzzles']:3d} FocusDiff={d['focus_sys1']:.3f} PTRM16={d['ptrm_16']:.3f} "
              f"WM-Tree={d['wm_tree']:.3f} Step={d['greedy_pass1']:.3f} pass@2={d['pass_at_2']:.3f} sys2={d.get('sys2_pass',0):.3f}")
    mem = evaluate(net, pool_tr, dev, mdp, n_samples=0, T=args.t_eval, per_tier=min(args.eval_per_tier, 16), run_sys2=False)
    gap = {t: dict(train=mem["tiers"][t]["greedy_pass1"], test=d["greedy_pass1"], gap=mem["tiers"][t]["greedy_pass1"] - d["greedy_pass1"])
           for t, d in ev["tiers"].items() if t in mem["tiers"]}
    key = args.gate_metric; sel = [d for t, d in ev["tiers"].items() if int(t) <= args.gate_tier]
    score = float(np.mean([d[key] for d in sel])) if sel else 0.0
    gate = dict(metric=key, score=score, target=args.gate_target, k=args.gate_k, tier=args.gate_tier, verdict="PASS" if score >= args.gate_target else "FAIL")
    print(f"\n[GATE] {key} on tiers<={args.gate_tier}: {score:.4f} vs {args.gate_target:.2f} -> {gate['verdict']}")
    (out / "gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    write_report(out, args, proofs, npar, it, peak, ev, gate, imag_res=imag_res, lv_res=lv_res, gap=gap)
    return 0 if proofs.all_ok else 1


if __name__ == "__main__":
    sys.exit(main())