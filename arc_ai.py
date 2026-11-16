#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 ARC-AGI-1 Benchmark Runner (v1.0.0) : Shared core.py HyperGraph + FROG +
 Coarse-to-Fine Focusing Diffusion (Untrained vs Trained + Demo-Only TTT)
================================================================================
"""
from __future__ import annotations
import argparse, copy, io, json, random, time, urllib.request, zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core import amp_ctx, lr_at, HyperGraphBlock, MuonAdamW, EMA

__version__ = "1.0.0"
ARC_ZIP_URL = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"
MAX_SZ = 30
N_CELLS = MAX_SZ * MAX_SZ
PAD_TOK = 10   # 0..9 colors, 10=outside grid


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
    overlap = len(set(train_tasks) & set(eval_tasks))
    assert overlap == 0, f"ARC train/eval overlap: {overlap}"
    print(f"[data] ARC-AGI-1 loaded: {len(train_tasks)} train / {len(eval_tasks)} eval tasks (0 overlap)", flush=True)
    return train_tasks, eval_tasks


def pad_grid(g):
    arr = np.asarray(g, dtype=np.int64)
    h, w = arr.shape
    out = np.full((MAX_SZ, MAX_SZ), PAD_TOK, dtype=np.int64)
    out[:h, :w] = arr
    return out.reshape(-1), h, w


def unpad_grid(flat, h, w):
    return flat.reshape(MAX_SZ, MAX_SZ)[:h, :w]


def apply_arc_sym(g, k_rot: int, flip: bool, cperm):
    arr = np.asarray(g, dtype=np.int64)
    if k_rot: arr = np.rot90(arr, k_rot)
    if flip: arr = np.fliplr(arr)
    return cperm[arr].copy()


class ARCNet(nn.Module):
    """Unified HyperGraph + FROG [mean, max] + Focusing Diffusion for variable <=30x30 ARC grids."""
    def __init__(self, C=96, heads=4, L=2, qk_norm=True):
        super().__init__()
        self.C = C
        tp_row = np.arange(N_CELLS, dtype=np.int64).reshape(MAX_SZ, MAX_SZ)
        tp_col = tp_row.T.copy()
        patches = [[max(0, min(MAX_SZ-1, r+dr))*MAX_SZ + max(0, min(MAX_SZ-1, c+dc))
                    for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
                   for r in range(MAX_SZ) for c in range(MAX_SZ)]
        tp_pat = np.array(patches, dtype=np.int64)
        self.unit_meta = []
        for nm, ak, w, arr in [("row", "line", 0.5, tp_row), ("col", "line", 0.5, tp_col), ("patch", "patch", 1.0, tp_pat)]:
            ti = torch.as_tensor(arr, dtype=torch.long); flat = ti.reshape(-1)
            deg = torch.bincount(flat, minlength=N_CELLS).float().clamp(min=1.0)
            isp = (flat.numel() == N_CELLS) and bool((torch.bincount(flat, minlength=N_CELLS) == 1).all())
            self.register_buffer(f"ug_{nm}", ti); self.register_buffer(f"ud_{nm}", deg.view(1, N_CELLS, 1, 1))
            if isp: self.register_buffer(f"inv_{nm}", torch.argsort(flat))
            self.unit_meta.append((nm, ak, w, isp))

        # Input: in_color(11) + draft_color(12, 11=MASK) + valid_mask(1) -> 24
        self.in_emb = nn.Embedding(11, C)
        self.out_emb = nn.Embedding(12, C)
        self.demo_blocks = nn.ModuleList([HyperGraphBlock(C, heads, ["line", "patch"], qk_norm=qk_norm) for _ in range(L)])
        self.rule_pool = nn.Sequential(nn.Linear(2 * C, C), nn.LayerNorm(C, eps=1e-5))
        self.test_blocks = nn.ModuleList([HyperGraphBlock(C, heads, ["line", "patch"], qk_norm=qk_norm) for _ in range(L)])
        self.lnf = nn.LayerNorm(C, eps=1e-5)
        self.pix_head = nn.Linear(C, 11)   # 0..9 colors + 10=PAD
        self.h_head = nn.Linear(2 * C, MAX_SZ)
        self.w_head = nn.Linear(2 * C, MAX_SZ)

    def _specs(self):
        return [(ak, w, getattr(self, f"ug_{nm}"), (getattr(self, f"inv_{nm}") if isp else None), getattr(self, f"ud_{nm}"))
                for nm, ak, w, isp in self.unit_meta]

    def encode_rule(self, demo_in, demo_out):
        # demo_in, demo_out: (D, 900)
        specs = self._specs()
        h = (self.in_emb(demo_in) + self.out_emb(demo_out)).unsqueeze(2)  # (D, 900, 1, C)
        for blk in self.demo_blocks:
            h = blk(h, specs)
        z = h.squeeze(2)  # (D, 900, C)
        pooled = self.rule_pool(torch.cat([z.mean(1), z.amax(1)], dim=-1))  # (D, C)
        return self.rule_pool(torch.cat([pooled.mean(0), pooled.amax(0)], dim=-1))  # (C,)

    def forward_test(self, test_in, draft_out, rule_vec, T=2):
        # test_in: (B, 900), draft_out: (B, 900), rule_vec: (B, C)
        specs = self._specs()
        h0 = (self.in_emb(test_in) + self.out_emb(draft_out) + rule_vec.unsqueeze(1)).unsqueeze(2)
        h = h0
        for _ in range(T):
            h = 0.5 * h + 0.5 * h0
            for blk in self.test_blocks:
                h = blk(h, specs)
        z = self.lnf(h.squeeze(2))
        g = torch.cat([z.mean(1), z.amax(1)], dim=-1)
        return self.pix_head(z).float(), self.h_head(g).float(), self.w_head(g).float()


def task_loss(net, task, device, rng, aug=True):
    pairs = task["train"] + task["test"]
    if len(pairs) < 2: return None
    # Hold out 1 pair from task as target, remaining pairs as demos
    tgt_idx = int(rng.integers(len(pairs)))
    demos = [pairs[i] for i in range(len(pairs)) if i != tgt_idx]
    q_pair = pairs[tgt_idx]
    k_rot = int(rng.integers(4)) if aug else 0
    flip = bool(rng.random() < 0.5) if aug else False
    cperm = np.arange(10, dtype=np.int64)
    if aug:
        cperm[1:] = rng.permutation(np.arange(1, 10))

    di_l, do_l = [], []
    for d in demos:
        g_i, _, _ = pad_grid(apply_arc_sym(d["input"], k_rot, flip, cperm))
        g_o, _, _ = pad_grid(apply_arc_sym(d["output"], k_rot, flip, cperm))
        di_l.append(g_i); do_l.append(g_o)
    qi, _, _ = pad_grid(apply_arc_sym(q_pair["input"], k_rot, flip, cperm))
    qo, qh, qw = pad_grid(apply_arc_sym(q_pair["output"], k_rot, flip, cperm))

    di_t = torch.as_tensor(np.stack(di_l), dtype=torch.long, device=device)
    do_t = torch.as_tensor(np.stack(do_l), dtype=torch.long, device=device)
    qi_t = torch.as_tensor(qi[None], dtype=torch.long, device=device)
    qo_t = torch.as_tensor(qo[None], dtype=torch.long, device=device)
    ht = torch.tensor([qh - 1], dtype=torch.long, device=device)
    wt = torch.tensor([qw - 1], dtype=torch.long, device=device)

    rule = net.encode_rule(di_t, do_t).unsqueeze(0)
    # 2-round Focusing Diffusion on output grid (coarse draft -> refined grid)
    draft = torch.full_like(qi_t, 11)
    tot_l = 0.0
    for seg in range(2):
        lg, hl, wl = net.forward_test(qi_t, draft, rule, T=2)
        l_pix = F.cross_entropy(lg.reshape(-1, 11), qo_t.reshape(-1))
        l_sz = 0.25 * (F.cross_entropy(hl, ht) + F.cross_entropy(wl, wt))
        tot_l = tot_l + 0.5 * (l_pix + l_sz)
        with torch.no_grad():
            probs = torch.softmax(lg, dim=-1)
            tp, td = probs.max(dim=-1)
            draft = torch.where(tp >= 0.75, td, torch.full_like(td, 11))
    return tot_l


@torch.no_grad()
def predict_arc_task(net, task, device, k_augs=8):
    """Focusing Diffusion prediction with D4 + color test-time voting -> top-2 candidates (pass@2)."""
    net.eval()
    demos = task["train"]
    test_in = task["test"][0]["input"]
    cands = {}
    rng = np.random.default_rng(7)
    for a_i in range(k_augs):
        k_rot = a_i % 4 if a_i > 0 else 0
        flip = (a_i >= 4) if a_i > 0 else False
        cperm = np.arange(10, dtype=np.int64)
        inv_c = np.arange(10, dtype=np.int64)
        di_l, do_l = [], []
        for d in demos:
            g_i, _, _ = pad_grid(apply_arc_sym(d["input"], k_rot, flip, cperm))
            g_o, _, _ = pad_grid(apply_arc_sym(d["output"], k_rot, flip, cperm))
            di_l.append(g_i); do_l.append(g_o)
        qi, _, _ = pad_grid(apply_arc_sym(test_in, k_rot, flip, cperm))
        di_t = torch.as_tensor(np.stack(di_l), dtype=torch.long, device=device)
        do_t = torch.as_tensor(np.stack(do_l), dtype=torch.long, device=device)
        qi_t = torch.as_tensor(qi[None], dtype=torch.long, device=device)
        with amp_ctx(device):
            rule = net.encode_rule(di_t, do_t).unsqueeze(0)
            draft = torch.full_like(qi_t, 11)
            for r in range(3):
                lg, hl, wl = net.forward_test(qi_t, draft, rule, T=2)
                probs = torch.softmax(lg, dim=-1)
                tp, td = probs.max(dim=-1)
                draft = torch.where(tp >= (0.85 - 0.1 * r), td, torch.full_like(td, 11))
        ph = int(hl.argmax(-1)[0].item()) + 1
        pw = int(wl.argmax(-1)[0].item()) + 1
        pred_full = lg[0].argmax(-1).clamp(max=9).cpu().numpy()
        crop = unpad_grid(pred_full, ph, pw)
        if flip: crop = np.fliplr(crop)
        if k_rot: crop = np.rot90(crop, 4 - k_rot)
        key = (crop.shape, crop.tobytes())
        cands[key] = (cands.get(key, (0, crop))[0] + 1, crop)
    ranked = sorted(cands.values(), key=lambda x: x[0], reverse=True)
    top2 = [x[1] for x in ranked[:2]]
    if len(top2) == 1: top2.append(top2[0])
    return top2


def eval_arc(net, eval_tasks, device, n_ttt=0, max_tasks=400, label=""):
    keys = sorted(eval_tasks.keys())[:max_tasks]
    p1 = p2 = 0
    base_state = {k: v.detach().clone() for k, v in net.state_dict().items()} if n_ttt > 0 else None
    t0 = time.time()
    for idx, tid in enumerate(keys):
        task = eval_tasks[tid]
        if n_ttt > 0 and len(task["train"]) >= 2:
            net.load_state_dict(base_state)
            net.train()
            opt_t = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4)
            rng_t = np.random.default_rng(idx + 1)
            for _ in range(n_ttt):
                # Train ONLY on this eval task's demonstration pairs (never test output!)
                sub_task = {"train": task["train"], "test": []}
                pairs = sub_task["train"]
                ti = int(rng_t.integers(len(pairs)))
                sub = {"train": [pairs[j] for j in range(len(pairs)) if j != ti], "test": [pairs[ti]]}
                opt_t.zero_grad()
                with amp_ctx(device):
                    l = task_loss(net, sub, device, rng_t, aug=True)
                if l is not None:
                    l.backward(); opt_t.step()
        top2 = predict_arc_task(net, task, device, k_augs=8)
        true_out = np.asarray(task["test"][0]["output"], dtype=np.int64)
        ok1 = (top2[0].shape == true_out.shape and np.array_equal(top2[0], true_out))
        ok2 = ok1 or (top2[1].shape == true_out.shape and np.array_equal(top2[1], true_out))
        p1 += int(ok1); p2 += int(ok2)
        if n_ttt > 0 and (idx + 1) % 25 == 0:
            print(f"   [ARC TTT progress] {idx+1}/{len(keys)}: pass@1={p1/(idx+1)*100:.1f}% pass@2={p2/(idx+1)*100:.1f}% ({time.time()-t0:.1f}s)", flush=True)
    if base_state is not None:
        net.load_state_dict(base_state)
    print(f"   [{label}] n={len(keys)} | pass@1={p1/len(keys)*100:.2f}% ({p1}/{len(keys)}) | pass@2={p2/len(keys)*100:.2f}% ({p2}/{len(keys)}) ({time.time()-t0:.1f}s)", flush=True)
    return dict(n=len(keys), pass1=p1/len(keys), pass2=p2/len(keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=12.0)
    ap.add_argument("--ttt-tasks", type=int, default=100)
    ap.add_argument("--ttt-steps", type=int, default=16)
    ap.add_argument("--out", default="runs/arc_v1")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    train_tasks, eval_tasks = download_arc1(Path("cache/thirdparty"))

    net = ARCNet(C=96, heads=4, L=2, qk_norm=True).to(dev)
    npar = sum(p.numel() for p in net.parameters())
    print(f"\n=== ARC-AGI-1 BENCHMARK (core.py HyperGraph + FROG + Focusing Diffusion, {npar:,} params, {dev}) ===", flush=True)

    # 1) WITHOUT training
    r_untrained = eval_arc(net, eval_tasks, dev, n_ttt=0, max_tasks=400, label="1. WITHOUT ARC Training (Untrained Zero-Shot)")

    # 2) Train on 400 public ARC-AGI-1 training tasks (0 eval tasks touched!)
    opt = MuonAdamW(net, lr_muon=0.015, lr_adam=3e-4, wd=1e-4)
    ema = EMA(net, decay=0.998)
    rng = np.random.default_rng(1234)
    keys_tr = sorted(train_tasks.keys())
    t_start = time.time(); budget = args.minutes * 60.0; it = 0
    print(f"\n[train] Training on 400 official ARC-AGI-1 training tasks for {args.minutes:.1f}m...", flush=True)
    net.train()
    while time.time() - t_start < budget:
        frac = (time.time() - t_start) / budget
        s = lr_at(frac); opt.set_lrs(0.015 * s, 3e-4 * s)
        opt.zero_grad(); loss_acc = 0.0; cnt_b = 0
        for _ in range(4):
            tk = train_tasks[keys_tr[int(rng.integers(len(keys_tr)))]]
            with amp_ctx(dev):
                l = task_loss(net, tk, dev, rng, aug=True)
            if l is not None:
                (l / 4.0).backward(); loss_acc += float(l.detach()) / 4.0; cnt_b += 1
        if cnt_b > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); ema.update(net); it += 1
            if it % 100 == 0 or it == 1:
                print(f"   [it {it:4d}] loss={loss_acc:.4f} ({(time.time()-t_start)/60:.1f}/{args.minutes:.0f}m)", flush=True)
    net.load_state_dict(ema.shadow)
    torch.save(dict(model=net.state_dict(), n_params=npar, iters=it), out / "weights.pt")

    # 3) WITH training: Zero-Shot on all 400 eval + TTT on eval demos
    print("\n=== ARC-AGI-1 RESULTS AFTER TRAINING ===", flush=True)
    r_tr_check = eval_arc(net, train_tasks, dev, n_ttt=0, max_tasks=100, label="Train Split Check (100 tasks)")
    r_zs = eval_arc(net, eval_tasks, dev, n_ttt=0, max_tasks=400, label="2. WITH Training -> Zero-Shot Eval (400 tasks, 0 TTT)")
    r_ttt = eval_arc(net, eval_tasks, dev, n_ttt=args.ttt_steps, max_tasks=args.ttt_tasks, label=f"3. WITH Training -> Demo-Only TTT ({args.ttt_steps} steps)")
    print("   Global ARC-AGI-1 refs: DeepSeek-R1(671B) 15.8% | Claude 3.7(16K) 28.6% | TRM(7M+430M IDs, 1000-aug) 40.0% (29.3% 1-aug) | HRM 32-40.3%", flush=True)
    (out / "arc_results.json").write_text(json.dumps(dict(params=npar, iters=it, untrained=r_untrained, train_check=r_tr_check, zero_shot=r_zs, ttt=r_ttt), indent=2))


if __name__ == "__main__":
    main()