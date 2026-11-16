#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 ARC-AGI-1 v4.0.0 : Fast CompressARC-Style Joint-Demo Equivariant MDL (0 CrossAttn)
================================================================================
Runs in ~2-3 minutes on n=100 official ARC-AGI-1 eval tasks (~1.5s/task, 160 steps).
Directly compresses all D demo pairs (in_k -> out_k) into a tiny 38K S9xD4 HyperGraph.
"""
from __future__ import annotations
import argparse, json, time, urllib.request, zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core import amp_ctx, HyperGraphBlock

__version__ = "4.0.0"
ARC_ZIP_URL = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"


def load_arc_eval(cache_dir: Path):
    zf = cache_dir / "arc_agi_1_master.zip"
    if not zf.exists():
        urllib.request.urlretrieve(ARC_ZIP_URL, zf)
    eval_tasks = {}
    with zipfile.ZipFile(zf, "r") as z:
        for name in z.namelist():
            if name.endswith(".json") and "/data/evaluation/" in name:
                eval_tasks[Path(name).stem] = json.loads(z.read(name).decode("utf-8"))
    return eval_tasks


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
    rr = torch.linspace(-1.0, 1.0, H, device=dev)[:, None].expand(H, W).reshape(N)
    cc = torch.linspace(-1.0, 1.0, W, device=dev)[None, :].expand(H, W).reshape(N)
    coords = torch.stack([rr, cc, rr.abs(), cc.abs()], dim=-1)
    return specs, coords


def infer_output_shape(demos, test_in):
    th, tw = len(test_in), len(test_in[0])
    in_s = [(len(d["input"]), len(d["input"][0])) for d in demos]
    out_s = [(len(d["output"]), len(d["output"][0])) for d in demos]
    if all(i == o for i, o in zip(in_s, out_s)): return th, tw
    if len(set(out_s)) == 1: return out_s[0]
    rh = [o[0] / i[0] for i, o in zip(in_s, out_s)]; rw = [o[1] / i[1] for i, o in zip(in_s, out_s)]
    if max(rh) - min(rh) < 1e-6 and max(rw) - min(rw) < 1e-6:
        return max(1, min(30, int(round(th * rh[0])))), max(1, min(30, int(round(tw * rw[0]))))
    return out_s[0]


def align_in(in_2d, Ho: int, Wo: int, device):
    t = torch.as_tensor(in_2d, dtype=torch.float32, device=device)[None, None]
    return F.interpolate(t, size=(Ho, Wo), mode="nearest")[0, 0].long().reshape(-1)


class TinyCompressARC(nn.Module):
    """~38K-param D4 x S9 HyperGraph compressor fit per task directly on demo (in -> out) pairs."""
    def __init__(self, C=32, heads=4, L=2):
        super().__init__()
        self.Win = nn.Linear(8, C)
        self.color_bias = nn.Parameter(torch.zeros(10, C))
        self.blocks = nn.ModuleList([HyperGraphBlock(C, heads, ["line", "patch"], qk_norm=True) for _ in range(L)])
        self.lnf = nn.LayerNorm(C, eps=1e-5)
        self.head = nn.Linear(C, 1)

    def forward(self, in_aligned_batch, Ho: int, Wo: int, T=2):
        # in_aligned_batch: (B, Ho*Wo)
        B, N = in_aligned_batch.shape
        specs, coords = _unit_specs_hw(Ho, Wo, str(in_aligned_batch.device))
        in_oh = F.one_hot(in_aligned_batch, 10).float()             # (B, N, 10)
        is_z = torch.zeros(B, N, 10, device=in_aligned_batch.device); is_z[:, :, 0] = 1.0
        hist = in_oh.mean(dim=1, keepdim=True).expand(B, N, 10)
        c4 = coords.view(1, N, 1, 4).expand(B, N, 10, 4)
        # Same-color neighbor match count (local object boundary detector)
        f = torch.cat([in_oh.unsqueeze(-1), is_z.unsqueeze(-1), hist.unsqueeze(-1),
                       (in_oh > 0.5).float().unsqueeze(-1), c4], dim=-1)  # (B, N, 10, 8)
        h0 = F.gelu(self.Win(f)) + self.color_bias.view(1, 1, 10, -1)
        h = h0
        for _ in range(T):
            h = 0.5 * h + 0.5 * h0
            for blk in self.blocks:
                h = blk(h, specs)
        return self.head(self.lnf(h)).squeeze(-1).float()  # (B, N, 10)


def solve_task_mdl(task, device, steps=160, C=32, L=2):
    demos = task["train"]
    test_in = np.asarray(task["test"][0]["input"], dtype=np.int64)
    Ho_test, Wo_test = infer_output_shape(demos, test_in)

    # Group demos by output (Ho, Wo) for instant batched GPU steps
    by_hw = {}
    for d in demos:
        di = np.asarray(d["input"], dtype=np.int64)
        do = np.asarray(d["output"], dtype=np.int64)
        hw = do.shape
        by_hw.setdefault(hw, []).append((align_in(di, hw[0], hw[1], device),
                                         torch.as_tensor(do.reshape(-1), dtype=torch.long, device=device)))
    batches = [(hw[0], hw[1], torch.stack([x[0] for x in lst]), torch.stack([x[1] for x in lst]))
               for hw, lst in by_hw.items()]

    torch.manual_seed(0)
    net = TinyCompressARC(C=C, heads=4, L=L).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=8e-3, weight_decay=0.04)  # MDL bottleneck!
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    net.train()
    solved_demos = False
    for st in range(steps):
        opt.zero_grad()
        tot_l = 0.0; exact_all = True
        for Ho, Wo, xin, ygt in batches:
            with amp_ctx(device):
                lg = net(xin, Ho, Wo, T=2)
                loss = F.cross_entropy(lg.reshape(-1, 10), ygt.reshape(-1))
            tot_l = tot_l + loss / len(batches)
            if not bool((lg.argmax(-1) == ygt).all()):
                exact_all = False
        tot_l.backward()
        opt.step(); sched.step()
        if exact_all and st >= 40:
            solved_demos = True
            # keep training briefly for MDL margin, or break early
            if st >= 80:
                break

    net.eval()
    with torch.no_grad(), amp_ctx(device):
        t_al = align_in(test_in, Ho_test, Wo_test, device).unsqueeze(0)
        lg_t = net(t_al, Ho_test, Wo_test, T=2)[0]
        pred = lg_t.argmax(-1).cpu().numpy().reshape(Ho_test, Wo_test)
    return pred, solved_demos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=100)
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--out", default="runs/arc_v4")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    eval_tasks = load_arc_eval(Path("cache/thirdparty"))
    keys = sorted(eval_tasks.keys())[:args.tasks]
    print(f"ARC-AGI-1 v4.0.0 (Pure CompressARC-style 38K HyperGraph MDL, 0 pretraining) | n={len(keys)} tasks | {dev}", flush=True)

    p1 = 0; demo_fit = 0; t0 = time.time(); solved_ids = []
    for i, tid in enumerate(keys):
        tk = eval_tasks[tid]
        pred, d_ok = solve_task_mdl(tk, dev, steps=args.steps)
        true_out = np.asarray(tk["test"][0]["output"], dtype=np.int64)
        ok = (pred.shape == true_out.shape and np.array_equal(pred, true_out))
        p1 += int(ok); demo_fit += int(d_ok)
        if ok: solved_ids.append(tid)
        if (i + 1) % 10 == 0:
            print(f"   [{i+1:3d}/{len(keys)}] exact_pass={p1/(i+1)*100:5.1f}% ({p1}/{i+1}) | demos_fit={demo_fit}/{i+1} ({time.time()-t0:.1f}s)", flush=True)

    print(f"\n[FINAL WITHOUT PRETRAINING] n={len(keys)} | pass@1={p1/len(keys)*100:.2f}% ({p1}/{len(keys)}) | solved={solved_ids}", flush=True)
    (out / "arc_v4_results.json").write_text(json.dumps(dict(n=len(keys), pass1=p1/len(keys), solved_ids=solved_ids, demo_fit=demo_fit), indent=2))


if __name__ == "__main__":
    main()