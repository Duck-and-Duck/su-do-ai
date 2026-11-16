#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 ARC-AGI-1 v2.0.0 : Zero-Padding Active-Grid Focusing Diffusion + Cross-Demo
================================================================================
Uses core.py (HyperGraphBlock + SelfAttnSet + MuonAdamW + EMA) on exact active
(H, W) grids (ZERO PAD_TOK!). Evaluates Untrained, Zero-Shot, and Demo-Only TTT.
"""
from __future__ import annotations
import argparse, json, math, time, urllib.request, zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core import amp_ctx, lr_at, HyperGraphBlock, MuonAdamW, EMA

__version__ = "2.0.0"
ARC_ZIP_URL = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"
MASK_COL = 10  # 0..9 = real ARC colors, 10 = MASK (only used inside draft_out!)


def download_arc1(cache_dir: Path):
    zf = cache_dir / "arc_agi_1_master.zip"
    if not zf.exists():
        print(f"[download] {ARC_ZIP_URL} ...", flush=True)
        urllib.request.urlretrieve(ARC_ZIP_URL, zf)
    train_tasks, eval_tasks = {}, {}
    with zipfile.ZipFile(zf, "r") as z:
        for name in z.namelist():
            if name.endswith(".json"):
                if "/data/training/" in name:
                    train_tasks[Path(name).stem] = json.loads(z.read(name).decode("utf-8"))
                elif "/data/evaluation/" in name:
                    eval_tasks[Path(name).stem] = json.loads(z.read(name).decode("utf-8"))
    assert len(set(train_tasks) & set(eval_tasks)) == 0
    print(f"[data] ARC-AGI-1: {len(train_tasks)} train / {len(eval_tasks)} eval tasks (0 overlap)", flush=True)
    return train_tasks, eval_tasks


@lru_cache(maxsize=512)
def _unit_specs_hw(H: int, W: int, dev_str: str):
    dev = torch.device(dev_str)
    N = H * W
    row_u = torch.arange(N, dtype=torch.long, device=dev).reshape(H, W)
    col_u = row_u.t().contiguous()
    patches = [[max(0, min(H - 1, r + dr)) * W + max(0, min(W - 1, c + dc))
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
               for r in range(H) for c in range(W)]
    pat_u = torch.tensor(patches, dtype=torch.long, device=dev)

    specs = []
    for ak, w, ti in [("line", 0.5, row_u), ("line", 0.5, col_u), ("patch", 1.0, pat_u)]:
        flat = ti.reshape(-1)
        deg = torch.bincount(flat, minlength=N).float().clamp(min=1.0).view(1, N, 1, 1)
        isp = (flat.numel() == N) and bool((torch.bincount(flat, minlength=N) == 1).all())
        inv = torch.argsort(flat) if isp else None
        specs.append((ak, w, ti, inv, deg))

    # Normalized 2D coordinates in [0, 1] for scale-aware relational reasoning
    rr = torch.linspace(0.0, 1.0, H, device=dev)[:, None].expand(H, W).reshape(N)
    cc = torch.linspace(0.0, 1.0, W, device=dev)[None, :].expand(H, W).reshape(N)
    coords = torch.stack([rr, cc, 1.0 - rr, 1.0 - cc], dim=-1)  # (N, 4)
    return specs, coords


def apply_arc_sym(g, k_rot: int, flip: bool, cperm):
    arr = np.asarray(g, dtype=np.int64)
    if k_rot: arr = np.rot90(arr, k_rot)
    if flip: arr = np.fliplr(arr)
    return cperm[arr].copy()


def infer_output_shape(demos, test_in):
    """Exact structural size rule across demos (same-shape, constant, or integer ratio)."""
    th, tw = len(test_in), len(test_in[0])
    in_s = [(len(d["input"]), len(d["input"][0])) for d in demos]
    out_s = [(len(d["output"]), len(d["output"][0])) for d in demos]
    if all(i == o for i, o in zip(in_s, out_s)):
        return th, tw
    if len(set(out_s)) == 1:
        return out_s[0]
    rh = [o[0] / i[0] for i, o in zip(in_s, out_s)]
    rw = [o[1] / i[1] for i, o in zip(in_s, out_s)]
    if max(rh) - min(rh) < 1e-6 and max(rw) - min(rw) < 1e-6:
        return max(1, min(30, int(round(th * rh[0])))), max(1, min(30, int(round(tw * rw[0]))))
    # Fallback: median aspect or same
    return out_s[0]


class CrossAttn(nn.Module):
    def __init__(self, C, heads):
        super().__init__()
        self.h = heads; self.dk = C // heads
        self.ln_q = nn.LayerNorm(C, eps=1e-5)
        self.ln_kv = nn.LayerNorm(C, eps=1e-5)
        self.q = nn.Linear(C, C, bias=False)
        self.kv = nn.Linear(C, 2 * C, bias=False)
        self.proj = nn.Linear(C, C, bias=False)
        self.qk_scale = nn.Parameter(torch.full((1, heads, 1, 1), 3.0))

    def forward(self, x_q, x_kv):
        # x_q: (1, Nq, C), x_kv: (1, Nkv, C)
        B, Nq, C = x_q.shape; Nkv = x_kv.shape[1]
        q = self.q(self.ln_q(x_q)).reshape(B, Nq, self.h, self.dk).transpose(1, 2)
        kv = self.kv(self.ln_kv(x_kv)).reshape(B, Nkv, 2, self.h, self.dk).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = F.normalize(q, dim=-1) * self.qk_scale.clamp(0.5, 6.0)
        k = F.normalize(k, dim=-1)
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, Nq, C)
        return x_q + self.proj(o)


class ARCNet(nn.Module):
    """Zero-padding variable-(H, W) HyperGraph + Cross-Demo Spatial Attention + Focusing Diffusion."""
    def __init__(self, C=96, heads=4, L=2):
        super().__init__()
        self.C = C
        self.col_emb = nn.Embedding(11, C)   # 0..9 colors + 10=MASK
        self.coord_proj = nn.Linear(4, C)
        self.role_emb = nn.Embedding(4, C)   # 0=demo_in, 1=demo_out, 2=test_in, 3=test_out
        self.pair_emb = nn.Embedding(12, C)  # demo pair index binding (in_i <-> out_i!)
        self.enc_blocks = nn.ModuleList([HyperGraphBlock(C, heads, ["line", "patch"], qk_norm=True) for _ in range(L)])
        self.cross_demo = CrossAttn(C, heads)
        self.dec_blocks = nn.ModuleList([HyperGraphBlock(C, heads, ["line", "patch"], qk_norm=True) for _ in range(L)])
        self.step_gate = nn.Parameter(torch.full((C,), 0.5))
        self.lnf = nn.LayerNorm(C, eps=1e-5)
        self.pix_head = nn.Linear(C, 10)     # ONLY 0..9 real colors (NO PAD!)

    def _encode_grid(self, grid_2d, role_id: int, pair_id: int, device):
        H, W = grid_2d.shape
        specs, coords = _unit_specs_hw(H, W, str(device))
        gt = torch.as_tensor(grid_2d.reshape(-1), dtype=torch.long, device=device)
        h = (self.col_emb(gt) + self.coord_proj(coords)
             + self.role_emb.weight[role_id] + self.pair_emb.weight[min(pair_id, 11)])
        h = h.unsqueeze(0).unsqueeze(2)  # (1, N, 1, C)
        for blk in self.enc_blocks:
            h = blk(h, specs)
        return h.squeeze(2)  # (1, N, C)

    def encode_context(self, demos, test_in, device):
        ctx_parts = []
        for pid, d in enumerate(demos):
            ctx_parts.append(self._encode_grid(np.asarray(d["input"], dtype=np.int64), 0, pid, device))
            ctx_parts.append(self._encode_grid(np.asarray(d["output"], dtype=np.int64), 1, pid, device))
        ctx_parts.append(self._encode_grid(np.asarray(test_in, dtype=np.int64), 2, 11, device))
        return torch.cat(ctx_parts, dim=1)  # (1, N_ctx, C)

    def forward_draft(self, ctx, draft_flat, H_out: int, W_out: int, h_state=None, same_hw_in=None):
        device = ctx.device
        specs, coords = _unit_specs_hw(H_out, W_out, str(device))
        h0 = (self.col_emb(draft_flat) + self.coord_proj(coords)
              + self.role_emb.weight[3] + self.pair_emb.weight[11]).unsqueeze(0)  # (1, N, C)
        if same_hw_in is not None:
            h0 = h0 + self.col_emb(same_hw_in).unsqueeze(0)
        alpha = torch.sigmoid(self.step_gate)
        h = h0 if h_state is None else ((1.0 - alpha) * h_state + alpha * h0)
        h = self.cross_demo(h, ctx)
        h4 = h.unsqueeze(2)
        for blk in self.dec_blocks:
            h4 = blk(h4, specs)
        h_out = h4.squeeze(2)
        logits = self.pix_head(self.lnf(h_out)).float()  # (1, H_out*W_out, 10)
        return logits[0], h_out


def task_loss(net, task, device, rng, aug=True, n_rounds=3):
    pairs = task["train"] + task["test"]
    if len(pairs) < 2: return None
    ti = int(rng.integers(len(pairs)))
    demos_raw = [pairs[i] for i in range(len(pairs)) if i != ti]
    q_raw = pairs[ti]
    k_rot = int(rng.integers(4)) if aug else 0
    flip = bool(rng.random() < 0.5) if aug else False
    cperm = np.arange(10, dtype=np.int64)
    if aug: cperm[1:] = rng.permutation(np.arange(1, 10))

    demos = [{"input": apply_arc_sym(d["input"], k_rot, flip, cperm),
              "output": apply_arc_sym(d["output"], k_rot, flip, cperm)} for d in demos_raw]
    t_in = apply_arc_sym(q_raw["input"], k_rot, flip, cperm)
    t_out = apply_arc_sym(q_raw["output"], k_rot, flip, cperm)
    Ho, Wo = t_out.shape
    tgt = torch.as_tensor(t_out.reshape(-1), dtype=torch.long, device=device)
    same_in = torch.as_tensor(t_in.reshape(-1), dtype=torch.long, device=device) if t_in.shape == t_out.shape else None

    ctx = net.encode_context(demos, t_in, device)
    draft = torch.full((Ho * Wo,), MASK_COL, dtype=torch.long, device=device)
    h_st = None
    tot_loss = 0.0
    for r in range(n_rounds):
        lg, h_st = net.forward_draft(ctx, draft, Ho, Wo, h_state=h_st, same_hw_in=same_in)
        tot_loss = tot_loss + (F.cross_entropy(lg, tgt) / n_rounds)
        with torch.no_grad():
            probs = torch.softmax(lg, dim=-1)
            tp, td = probs.max(dim=-1)
            # Progressive focusing + occasional noise correction
            draft = torch.where(tp >= (0.85 - 0.15 * r), td, torch.full_like(td, MASK_COL))
    return tot_loss


@torch.no_grad()
def predict_arc_task(net, task, device, k_augs=4):
    net.eval()
    demos_raw = task["train"]
    test_in_raw = np.asarray(task["test"][0]["input"], dtype=np.int64)
    cands = {}
    for a_i in range(k_augs):
        k_rot = a_i % 4 if a_i > 0 else 0
        flip = False
        cperm = np.arange(10, dtype=np.int64)
        demos = [{"input": apply_arc_sym(d["input"], k_rot, flip, cperm),
                  "output": apply_arc_sym(d["output"], k_rot, flip, cperm)} for d in demos_raw]
        t_in = apply_arc_sym(test_in_raw, k_rot, flip, cperm)
        Ho, Wo = infer_output_shape(demos, t_in)
        same_in = torch.as_tensor(t_in.reshape(-1), dtype=torch.long, device=device) if t_in.shape == (Ho, Wo) else None
        with amp_ctx(device):
            ctx = net.encode_context(demos, t_in, device)
            draft = torch.full((Ho * Wo,), MASK_COL, dtype=torch.long, device=device)
            h_st = None
            for r in range(4):
                lg, h_st = net.forward_draft(ctx, draft, Ho, Wo, h_state=h_st, same_hw_in=same_in)
                probs = torch.softmax(lg / max(0.5, 1.1 - 0.2 * r), dim=-1)
                tp, td = probs.max(dim=-1)
                draft = torch.where(tp >= (0.88 - 0.12 * r), td, torch.full_like(td, MASK_COL))
        pred = lg.argmax(dim=-1).cpu().numpy().reshape(Ho, Wo)
        if k_rot: pred = np.rot90(pred, 4 - k_rot)
        key = (pred.shape, pred.tobytes())
        cands[key] = (cands.get(key, (0, pred))[0] + 1, pred)
    ranked = sorted(cands.values(), key=lambda x: x[0], reverse=True)
    top2 = [x[1] for x in ranked[:2]]
    if len(top2) == 1: top2.append(top2[0])
    return top2


def eval_arc(net, tasks_dict, device, n_ttt=0, max_tasks=400, label=""):
    keys = sorted(tasks_dict.keys())[:max_tasks]
    p1 = p2 = 0; t0 = time.time()
    base_state = {k: v.detach().clone() for k, v in net.state_dict().items()} if n_ttt > 0 else None
    for idx, tid in enumerate(keys):
        task = tasks_dict[tid]
        if n_ttt > 0 and len(task["train"]) >= 2:
            net.load_state_dict(base_state); net.train()
            opt_t = torch.optim.AdamW(net.parameters(), lr=4e-4, weight_decay=1e-4)
            rng_t = np.random.default_rng(idx + 1)
            for _ in range(n_ttt):
                pairs = task["train"]
                ti = int(rng_t.integers(len(pairs)))
                sub = {"train": [pairs[j] for j in range(len(pairs)) if j != ti], "test": [pairs[ti]]}
                opt_t.zero_grad()
                with amp_ctx(device): l = task_loss(net, sub, device, rng_t, aug=True, n_rounds=2)
                if l is not None: l.backward(); opt_t.step()
        top2 = predict_arc_task(net, task, device, k_augs=4)
        true_out = np.asarray(task["test"][0]["output"], dtype=np.int64)
        ok1 = (top2[0].shape == true_out.shape and np.array_equal(top2[0], true_out))
        ok2 = ok1 or (top2[1].shape == true_out.shape and np.array_equal(top2[1], true_out))
        p1 += int(ok1); p2 += int(ok2)
        if n_ttt > 0 and (idx + 1) % 25 == 0:
            print(f"   [ARC TTT] {idx+1}/{len(keys)}: pass@1={p1/(idx+1)*100:.1f}% pass@2={p2/(idx+1)*100:.1f}% ({time.time()-t0:.1f}s)", flush=True)
    if base_state is not None: net.load_state_dict(base_state)
    print(f"   [{label}] n={len(keys)} | pass@1={p1/len(keys)*100:.2f}% ({p1}/{len(keys)}) | pass@2={p2/len(keys)*100:.2f}% ({p2}/{len(keys)}) ({time.time()-t0:.1f}s)", flush=True)
    return dict(n=len(keys), pass1=p1/len(keys), pass2=p2/len(keys), c1=p1, c2=p2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=12.0)
    ap.add_argument("--ttt-tasks", type=int, default=100)
    ap.add_argument("--ttt-steps", type=int, default=20)
    ap.add_argument("--out", default="runs/arc_v2")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    train_tasks, eval_tasks = download_arc1(Path("cache/thirdparty"))

    net = ARCNet(C=96, heads=4, L=2).to(dev)
    npar = sum(p.numel() for p in net.parameters())
    print(f"\n=== ARC-AGI-1 v2.0.0 (Zero-Padding Focusing Diffusion + Cross-Demo, {npar:,} params, {dev}) ===", flush=True)

    r_untrained = eval_arc(net, eval_tasks, dev, n_ttt=0, max_tasks=400, label="1. WITHOUT ARC Training (Untrained)")

    opt = MuonAdamW(net, lr_muon=0.015, lr_adam=3e-4, wd=1e-4)
    ema = EMA(net, decay=0.998)
    rng = np.random.default_rng(1234); keys_tr = sorted(train_tasks.keys())
    t_start = time.time(); budget = args.minutes * 60.0; it = 0
    print(f"\n[train] Training on 400 official ARC-AGI-1 training tasks for {args.minutes:.1f}m...", flush=True)
    net.train()
    while time.time() - t_start < budget:
        frac = (time.time() - t_start) / budget; s = lr_at(frac); opt.set_lrs(0.015 * s, 3e-4 * s)
        opt.zero_grad(); loss_acc = 0.0; cnt_b = 0
        for _ in range(6):
            tk = train_tasks[keys_tr[int(rng.integers(len(keys_tr)))]]
            with amp_ctx(dev): l = task_loss(net, tk, dev, rng, aug=True, n_rounds=3)
            if l is not None:
                (l / 6.0).backward(); loss_acc += float(l.detach()) / 6.0; cnt_b += 1
        if cnt_b > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); ema.update(net); it += 1
            if it % 50 == 0 or it == 1:
                print(f"   [it {it:4d}] loss={loss_acc:.4f} ({(time.time()-t_start)/60:.1f}/{args.minutes:.0f}m)", flush=True)
    net.load_state_dict(ema.shadow)
    torch.save(dict(model=net.state_dict(), n_params=npar, iters=it), out / "weights.pt")

    print("\n=== ARC-AGI-1 RESULTS AFTER TRAINING ===", flush=True)
    r_tr = eval_arc(net, train_tasks, dev, n_ttt=0, max_tasks=100, label="Train Split Check (100 tasks)")
    r_zs = eval_arc(net, eval_tasks, dev, n_ttt=0, max_tasks=400, label="2. WITH Training -> Zero-Shot Eval (400 tasks, 0 TTT)")
    r_ttt = eval_arc(net, eval_tasks, dev, n_ttt=args.ttt_steps, max_tasks=args.ttt_tasks, label=f"3. WITH Training -> Demo-Only TTT ({args.ttt_steps} steps)")
    print("   Global ARC-AGI-1 refs: DeepSeek-R1(671B) 15.8% | Claude 3.7 28.6% | TRM(7M+430M IDs) 40.0% (29.3% 1-aug)", flush=True)
    (out / "arc_results.json").write_text(json.dumps(dict(params=npar, iters=it, untrained=r_untrained, train=r_tr, zero_shot=r_zs, ttt=r_ttt), indent=2))


if __name__ == "__main__":
    main()