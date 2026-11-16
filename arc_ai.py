#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 ARC-AGI-1 v3.0.0 : S9-Color & D4-Equivariant Active-Grid Focusing Diffusion
================================================================================
- ZERO PAD_TOK: every grid runs on its exact (H, W) HyperGraphBlock (core.py)
- K=10 color slots (background 0 distinct, foreground 1..9 weight-shared S9)
- Benchmarks WITHOUT pretraining (per-task MDL), WITH pretraining Zero-Shot, & TTT
"""
from __future__ import annotations
import argparse, json, time, urllib.request, zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core import amp_ctx, lr_at, HyperGraphBlock, MuonAdamW, EMA

__version__ = "3.0.0"
ARC_ZIP_URL = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"


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
    dev = torch.device(dev_str); N = H * W
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
    rr = torch.linspace(0.0, 1.0, H, device=dev)[:, None].expand(H, W).reshape(N)
    cc = torch.linspace(0.0, 1.0, W, device=dev)[None, :].expand(H, W).reshape(N)
    coords = torch.stack([rr, cc, 1.0 - rr, 1.0 - cc], dim=-1)  # (N, 4)
    return specs, coords


def apply_arc_rot(g, k_rot: int, flip: bool):
    arr = np.asarray(g, dtype=np.int64)
    if k_rot: arr = np.rot90(arr, k_rot)
    if flip: arr = np.fliplr(arr)
    return arr.copy()


def infer_output_shape(demos, test_in):
    th, tw = len(test_in), len(test_in[0])
    in_s = [(len(d["input"]), len(d["input"][0])) for d in demos]
    out_s = [(len(d["output"]), len(d["output"][0])) for d in demos]
    if all(i == o for i, o in zip(in_s, out_s)): return th, tw
    if len(set(out_s)) == 1: return out_s[0]
    rh = [o[0] / i[0] for i, o in zip(in_s, out_s)]
    rw = [o[1] / i[1] for i, o in zip(in_s, out_s)]
    if max(rh) - min(rh) < 1e-6 and max(rw) - min(rw) < 1e-6:
        return max(1, min(30, int(round(th * rh[0])))), max(1, min(30, int(round(tw * rw[0]))))
    return out_s[0]


def align_in_to_out(in_2d, Ho: int, Wo: int, device):
    """Nearest-neighbor spatial alignment of input grid onto target (Ho, Wo) grid."""
    Hi, Wi = in_2d.shape
    t = torch.as_tensor(in_2d, dtype=torch.float32, device=device)[None, None]
    return F.interpolate(t, size=(Ho, Wo), mode="nearest")[0, 0].long().reshape(-1)


class EquivCrossAttn(nn.Module):
    def __init__(self, C, heads):
        super().__init__()
        self.h = heads; self.dk = C // heads
        self.ln_q = nn.LayerNorm(C, eps=1e-5); self.ln_kv = nn.LayerNorm(C, eps=1e-5)
        self.q = nn.Linear(C, C, bias=False); self.kv = nn.Linear(C, 2 * C, bias=False); self.proj = nn.Linear(C, C, bias=False)
        self.qk_scale = nn.Parameter(torch.full((1, heads, 1, 1), 3.0))

    def forward(self, x_q, x_kv):
        # x_q: (1, Nq*10, C), x_kv: (1, Nkv*10, C)
        B, Nq, C = x_q.shape; Nkv = x_kv.shape[1]
        q = self.q(self.ln_q(x_q)).reshape(B, Nq, self.h, self.dk).transpose(1, 2)
        kv = self.kv(self.ln_kv(x_kv)).reshape(B, Nkv, 2, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q = F.normalize(q, dim=-1) * self.qk_scale.clamp(0.5, 6.0)
        k = F.normalize(kv[0], dim=-1)
        o = F.scaled_dot_product_attention(q, k, kv[1]).transpose(1, 2).reshape(B, Nq, C)
        return x_q + self.proj(o)


class ARCNet(nn.Module):
    """Color-equivariant (K=10 slots) & D4-equivariant HyperGraph Focusing Diffusion."""
    def __init__(self, C=48, heads=4, L=2):
        super().__init__()
        self.C = C
        # 9 features per (cell, color_slot): in_oh, draft_oh, is_zero_slot, in_hist_c, same_hw_flag, 4 coords
        self.Win = nn.Linear(9, C)
        self.color_task_bias = nn.Parameter(torch.zeros(10, C))  # used only during per-task TTT
        self.enc_blocks = nn.ModuleList([HyperGraphBlock(C, heads, ["line", "patch"], qk_norm=True) for _ in range(L)])
        self.role_emb = nn.Embedding(3, C)  # 0=demo_in, 1=demo_out, 2=test_in
        self.cross_demo = EquivCrossAttn(C, heads)
        self.dec_blocks = nn.ModuleList([HyperGraphBlock(C, heads, ["line", "patch"], qk_norm=True) for _ in range(L)])
        self.step_gate = nn.Parameter(torch.full((C,), 0.5))
        self.lnf = nn.LayerNorm(C, eps=1e-5)
        self.head = nn.Linear(C, 1)  # color logits across K=10 slots

    def _grid_feats(self, in_flat, draft_flat, H: int, W: int, same_hw: bool, device):
        N = H * W
        _, coords = _unit_specs_hw(H, W, str(device))
        in_oh = F.one_hot(in_flat, 10).float()                           # (N, 10)
        dr_oh = torch.zeros(N, 10, device=device)
        if draft_flat is not None:
            valid_d = draft_flat < 10
            if bool(valid_d.any()):
                dr_oh[valid_d] = F.one_hot(draft_flat[valid_d], 10).float()
        is_z = torch.zeros(N, 10, device=device); is_z[:, 0] = 1.0
        hist = in_oh.mean(dim=0, keepdim=True).expand(N, 10)
        shw = torch.full((N, 10), 1.0 if same_hw else 0.0, device=device)
        c4 = coords.unsqueeze(1).expand(N, 10, 4)
        f = torch.cat([in_oh.unsqueeze(-1), dr_oh.unsqueeze(-1), is_z.unsqueeze(-1),
                       hist.unsqueeze(-1), shw.unsqueeze(-1), c4], dim=-1)  # (N, 10, 9)
        return f.unsqueeze(0)  # (1, N, 10, 9)

    def _encode_pair(self, in_2d, out_2d, device):
        Hi, Wi = in_2d.shape; Ho, Wo = out_2d.shape
        specs_i, _ = _unit_specs_hw(Hi, Wi, str(device))
        specs_o, _ = _unit_specs_hw(Ho, Wo, str(device))
        fi = self._grid_feats(torch.as_tensor(in_2d.reshape(-1), dtype=torch.long, device=device), None, Hi, Wi, True, device)
        fo = self._grid_feats(torch.as_tensor(out_2d.reshape(-1), dtype=torch.long, device=device), None, Ho, Wo, True, device)
        hi = F.gelu(self.Win(fi)) + self.color_task_bias.view(1, 1, 10, -1) + self.role_emb.weight[0].view(1, 1, 1, -1)
        ho = F.gelu(self.Win(fo)) + self.color_task_bias.view(1, 1, 10, -1) + self.role_emb.weight[1].view(1, 1, 1, -1)
        for blk in self.enc_blocks:
            hi = blk(hi, specs_i); ho = blk(ho, specs_o)
        # Pool per demo to keep cross-attention fast: (1, (Hi*Wi+Ho*Wo) if small else pooled) -> keep full if <=225 else 10-color summary
        zi = hi.reshape(1, -1, self.C) if Hi * Wi <= 225 else torch.cat([hi.mean(1), hi.amax(1)], dim=1)
        zo = ho.reshape(1, -1, self.C) if Ho * Wo <= 225 else torch.cat([ho.mean(1), ho.amax(1)], dim=1)
        return torch.cat([zi, zo], dim=1)

    def encode_context(self, demos, test_in, device):
        parts = [self._encode_pair(np.asarray(d["input"], dtype=np.int64),
                                   np.asarray(d["output"], dtype=np.int64), device) for d in demos]
        Hi, Wi = test_in.shape; specs_i, _ = _unit_specs_hw(Hi, Wi, str(device))
        fi = self._grid_feats(torch.as_tensor(test_in.reshape(-1), dtype=torch.long, device=device), None, Hi, Wi, True, device)
        hi = F.gelu(self.Win(fi)) + self.color_task_bias.view(1, 1, 10, -1) + self.role_emb.weight[2].view(1, 1, 1, -1)
        for blk in self.enc_blocks: hi = blk(hi, specs_i)
        parts.append(hi.reshape(1, -1, self.C) if Hi * Wi <= 225 else torch.cat([hi.mean(1), hi.amax(1)], dim=1))
        return torch.cat(parts, dim=1)

    def forward_focus(self, ctx, in_2d, Ho: int, Wo: int, n_rounds=3, device="cuda"):
        specs_o, _ = _unit_specs_hw(Ho, Wo, str(device))
        in_aligned = align_in_to_out(in_2d, Ho, Wo, device)
        same_hw = (in_2d.shape == (Ho, Wo))
        draft = torch.full((Ho * Wo,), 10, dtype=torch.long, device=device)
        h_st = None; logits_list = []; alpha = torch.sigmoid(self.step_gate)
        for r in range(n_rounds):
            fo = self._grid_feats(in_aligned, draft, Ho, Wo, same_hw, device)
            h0 = F.gelu(self.Win(fo)) + self.color_task_bias.view(1, 1, 10, -1)
            h = h0 if h_st is None else ((1.0 - alpha) * h_st + alpha * h0)
            h = self.cross_demo(h.reshape(1, Ho * Wo * 10, self.C), ctx).reshape(1, Ho * Wo, 10, self.C)
            for blk in self.dec_blocks:
                h = blk(h, specs_o)
            h_st = h
            lg = self.head(self.lnf(h))[0, :, :, 0].float()  # (Ho*Wo, 10)
            logits_list.append(lg)
            if r + 1 < n_rounds:
                with torch.no_grad():
                    probs = torch.softmax(lg / max(0.5, 1.1 - 0.25 * r), dim=-1)
                    tp, td = probs.max(dim=-1)
                    draft = torch.where(tp >= (0.85 - 0.12 * r), td, torch.full_like(td, 10))
        return logits_list


def pair_loss(net, demos, t_in, t_out, device, n_rounds=2):
    Ho, Wo = t_out.shape
    tgt = torch.as_tensor(t_out.reshape(-1), dtype=torch.long, device=device)
    ctx = net.encode_context(demos, t_in, device)
    lgs = net.forward_focus(ctx, t_in, Ho, Wo, n_rounds=n_rounds, device=device)
    loss = sum(F.cross_entropy(lg, tgt) for lg in lgs) / len(lgs)
    acc = float((lgs[-1].argmax(-1) == tgt).float().mean())
    exact = bool((lgs[-1].argmax(-1) == tgt).all())
    return loss, acc, exact


@torch.no_grad()
def predict_arc_task(net, task, device, k_augs=4):
    net.eval(); demos_raw = task["train"]; test_in_raw = np.asarray(task["test"][0]["input"], dtype=np.int64); cands = {}
    for a_i in range(k_augs):
        k_rot = a_i % 4 if a_i > 0 else 0
        demos = [{"input": apply_arc_rot(d["input"], k_rot, False),
                  "output": apply_arc_rot(d["output"], k_rot, False)} for d in demos_raw]
        t_in = apply_arc_rot(test_in_raw, k_rot, False)
        Ho, Wo = infer_output_shape(demos, t_in)
        with amp_ctx(device):
            ctx = net.encode_context(demos, t_in, device)
            lgs = net.forward_focus(ctx, t_in, Ho, Wo, n_rounds=3, device=device)
        pred = lgs[-1].argmax(dim=-1).cpu().numpy().reshape(Ho, Wo)
        if k_rot: pred = np.rot90(pred, 4 - k_rot)
        key = (pred.shape, pred.tobytes()); cands[key] = (cands.get(key, (0, pred))[0] + 1, pred)
    ranked = sorted(cands.values(), key=lambda x: x[0], reverse=True)
    top2 = [x[1] for x in ranked[:2]]
    if len(top2) == 1: top2.append(top2[0])
    return top2


def adapt_and_eval(net, tasks_dict, device, n_ttt=0, max_tasks=400, label=""):
    keys = sorted(tasks_dict.keys())[:max_tasks]; p1 = p2 = 0; t0 = time.time()
    base_state = {k: v.detach().clone() for k, v in net.state_dict().items()} if n_ttt > 0 else None
    for idx, tid in enumerate(keys):
        task = tasks_dict[tid]
        if n_ttt > 0 and len(task["train"]) >= 2:
            net.load_state_dict(base_state); net.train()
            net.color_task_bias.requires_grad_(True)
            opt_t = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=0.02)
            rng_t = np.random.default_rng(idx + 1)
            for _ in range(n_ttt):
                pairs = task["train"]
                ti = int(rng_t.integers(len(pairs)))
                k_rot = int(rng_t.integers(4)); flip = bool(rng_t.random() < 0.5)
                demos = [{"input": apply_arc_rot(pairs[j]["input"], k_rot, flip),
                          "output": apply_arc_rot(pairs[j]["output"], k_rot, flip)} for j in range(len(pairs)) if j != ti]
                t_in = apply_arc_rot(pairs[ti]["input"], k_rot, flip)
                t_out = apply_arc_rot(pairs[ti]["output"], k_rot, flip)
                opt_t.zero_grad()
                with amp_ctx(device):
                    l, _, _ = pair_loss(net, demos, t_in, t_out, device, n_rounds=2)
                l.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt_t.step()
        top2 = predict_arc_task(net, task, device, k_augs=4)
        true_out = np.asarray(task["test"][0]["output"], dtype=np.int64)
        ok1 = (top2[0].shape == true_out.shape and np.array_equal(top2[0], true_out))
        ok2 = ok1 or (top2[1].shape == true_out.shape and np.array_equal(top2[1], true_out))
        p1 += int(ok1); p2 += int(ok2)
        if n_ttt > 0 and (idx + 1) % 20 == 0:
            print(f"   [TTT progress] {idx+1}/{len(keys)}: pass@1={p1/(idx+1)*100:.1f}% pass@2={p2/(idx+1)*100:.1f}% ({time.time()-t0:.1f}s)", flush=True)
    if base_state is not None: net.load_state_dict(base_state)
    print(f"   [{label}] n={len(keys)} | pass@1={p1/len(keys)*100:.2f}% ({p1}/{len(keys)}) | pass@2={p2/len(keys)*100:.2f}% ({p2}/{len(keys)}) ({time.time()-t0:.1f}s)", flush=True)
    return dict(n=len(keys), pass1=p1/len(keys), pass2=p2/len(keys), c1=p1, c2=p2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--mdl-tasks", type=int, default=60)
    ap.add_argument("--mdl-steps", type=int, default=60)
    ap.add_argument("--ttt-tasks", type=int, default=60)
    ap.add_argument("--ttt-steps", type=int, default=60)
    ap.add_argument("--out", default="runs/arc_v3")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    train_tasks, eval_tasks = download_arc1(Path("cache/thirdparty"))

    net = ARCNet(C=48, heads=4, L=2).to(dev)
    npar = sum(p.numel() for p in net.parameters())
    print(f"\n=== ARC-AGI-1 v3.0.0 (S9-Color & D4-Equiv Active-Grid Focusing, {npar:,} params, {dev}) ===", flush=True)

    # 1A) WITHOUT pretraining - raw zero-shot
    r_raw0 = adapt_and_eval(net, eval_tasks, dev, n_ttt=0, max_tasks=400, label="1A. WITHOUT Pretraining (0 steps)")
    # 1B) WITHOUT pretraining - pure per-puzzle MDL adaptation from scratch (CompressARC-style!)
    r_mdl0 = adapt_and_eval(net, eval_tasks, dev, n_ttt=args.mdl_steps, max_tasks=args.mdl_tasks, label=f"1B. WITHOUT Pretraining ({args.mdl_steps}-step MDL on demos)")

    # 2) Pretrain on 400 official ARC training tasks (color_task_bias frozen at 0 so S9-color-equivariant!)
    net.color_task_bias.requires_grad_(False)
    opt = MuonAdamW(net, lr_muon=0.015, lr_adam=4e-4, wd=1e-4)
    ema = EMA(net, decay=0.998)
    rng = np.random.default_rng(1234); keys_tr = sorted(train_tasks.keys())
    t_start = time.time(); budget = args.minutes * 60.0; it = 0
    print(f"\n[train] Pretraining S9-equivariant ARCNet on 400 train tasks for {args.minutes:.1f}m...", flush=True)
    net.train()
    while time.time() - t_start < budget:
        frac = (time.time() - t_start) / budget; s = lr_at(frac); opt.set_lrs(0.015 * s, 4e-4 * s)
        opt.zero_grad(); loss_acc = acc_acc = ex_acc = 0.0; cnt_b = 0
        for _ in range(4):
            tk = train_tasks[keys_tr[int(rng.integers(len(keys_tr)))]]
            pairs = tk["train"] + tk["test"]
            if len(pairs) < 2: continue
            ti = int(rng.integers(len(pairs))); k_rot = int(rng.integers(4)); flip = bool(rng.random() < 0.5)
            demos = [{"input": apply_arc_rot(pairs[j]["input"], k_rot, flip),
                      "output": apply_arc_rot(pairs[j]["output"], k_rot, flip)} for j in range(len(pairs)) if j != ti]
            t_in = apply_arc_rot(pairs[ti]["input"], k_rot, flip)
            t_out = apply_arc_rot(pairs[ti]["output"], k_rot, flip)
            with amp_ctx(dev):
                l, pa, ex = pair_loss(net, demos, t_in, t_out, dev, n_rounds=2)
            (l / 4.0).backward(); loss_acc += float(l.detach()) / 4.0; acc_acc += pa / 4.0; ex_acc += float(ex) / 4.0; cnt_b += 1
        if cnt_b > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); ema.update(net); it += 1
            if it % 50 == 0 or it == 1:
                print(f"   [it {it:4d}] loss={loss_acc:.4f} pix_acc={acc_acc:.3f} exact={ex_acc:.2f} ({(time.time()-t_start)/60:.1f}/{args.minutes:.0f}m)", flush=True)
    net.load_state_dict(ema.shadow)
    torch.save(dict(model=net.state_dict(), n_params=npar, iters=it), out / "weights.pt")

    print("\n=== ARC-AGI-1 RESULTS AFTER PRETRAINING ===", flush=True)
    r_tr = adapt_and_eval(net, train_tasks, dev, n_ttt=0, max_tasks=100, label="Train Split Check (100 tasks)")
    r_zs = adapt_and_eval(net, eval_tasks, dev, n_ttt=0, max_tasks=400, label="2. WITH Pretraining -> Zero-Shot Eval (400 tasks)")
    r_ttt = adapt_and_eval(net, eval_tasks, dev, n_ttt=args.ttt_steps, max_tasks=args.ttt_tasks, label=f"3. WITH Pretraining -> Demo-Only TTT ({args.ttt_steps} steps)")
    (out / "arc_results.json").write_text(json.dumps(dict(params=npar, iters=it, raw0=r_raw0, mdl0=r_mdl0, train=r_tr, zero_shot=r_zs, ttt=r_ttt), indent=2))


if __name__ == "__main__":
    main()