#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 ARC-AGI-1 v5.0.3 : Fixed-Noise Object-Aware D4xS9 Equivariant ELBO + Shape Router
================================================================================
"""
from __future__ import annotations
import argparse, json, math, time, urllib.request, zipfile
from collections import Counter, deque
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core import SelfAttnSet

__version__ = "5.0.3"
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
def _hw_cache(H: int, W: int, dev_str: str):
    dev = torch.device(dev_str); N = H * W
    rr = torch.linspace(-1.0, 1.0, H, device=dev)[:, None].expand(H, W).reshape(N)
    cc = torch.linspace(-1.0, 1.0, W, device=dev)[None, :].expand(H, W).reshape(N)
    return torch.stack([rr, cc, rr.abs(), cc.abs()], dim=-1)


def extract_object_features(grid_2d: np.ndarray) -> np.ndarray:
    H, W = grid_2d.shape; N = H * W
    feats = np.zeros((H, W, 9), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            col = grid_2d[r, c]
            n4 = sum(1 for dr, dc in ((-1,0),(1,0),(0,-1),(0,1))
                     if 0 <= r+dr < H and 0 <= c+dc < W and grid_2d[r+dr, c+dc] == col)
            n8 = sum(1 for dr in (-1,0,1) for dc in (-1,0,1)
                     if (dr or dc) and 0 <= r+dr < H and 0 <= c+dc < W and grid_2d[r+dr, c+dc] == col)
            feats[r, c, 0] = n4 / 4.0
            feats[r, c, 1] = n8 / 8.0
    vis = np.zeros((H, W), dtype=bool); comps = []
    for r in range(H):
        for c in range(W):
            if grid_2d[r, c] != 0 and not vis[r, c]:
                col = grid_2d[r, c]; q = deque([(r, c)]); vis[r, c] = True; cells = []
                r0 = r1 = r; c0 = c1 = c
                while q:
                    cr, cc = q.popleft(); cells.append((cr, cc))
                    r0 = min(r0, cr); r1 = max(r1, cr); c0 = min(c0, cc); c1 = max(c1, cc)
                    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and not vis[nr, nc] and grid_2d[nr, nc] == col:
                            vis[nr, nc] = True; q.append((nr, nc))
                comps.append((cells, len(cells), r0, r1, c0, c1))
    if comps:
        areas = [x[1] for x in comps]; mx_a, mn_a = max(areas), min(areas); acnt = Counter(areas)
        for cells, area, r0, r1, c0, c1 in comps:
            bh = (r1 - r0 + 1) / H; bw = (c1 - c0 + 1) / W
            solid = float(area == (r1 - r0 + 1) * (c1 - c0 + 1))
            is_max, is_min, is_uniq = float(area == mx_a), float(area == mn_a), float(acnt[area] == 1)
            for cr, cc in cells:
                feats[cr, cc, 2] = area / N; feats[cr, cc, 3] = is_max; feats[cr, cc, 4] = is_min
                feats[cr, cc, 5] = is_uniq; feats[cr, cc, 6] = solid; feats[cr, cc, 7] = bh; feats[cr, cc, 8] = bw
    return feats.reshape(N, 9)


def find_subgrid_mask(in_g: np.ndarray, out_g: np.ndarray):
    hi, wi = in_g.shape; ho, wo = out_g.shape
    if ho > hi or wo > wi: return None
    for r0 in range(hi - ho + 1):
        for c0 in range(wi - wo + 1):
            if np.array_equal(in_g[r0:r0+ho, c0:c0+wo], out_g):
                m = np.zeros((hi, wi), dtype=np.float32); m[r0:r0+ho, c0:c0+wo] = 1.0
                return m.reshape(-1)
    return None


def detect_task_mode(demos):
    in_s = [np.asarray(d["input"]).shape for d in demos]
    out_s = [np.asarray(d["output"]).shape for d in demos]
    if all(i == o for i, o in zip(in_s, out_s)):
        return "SAME_HW", (1, 1), None
    if all(o[0] % i[0] == 0 and o[1] % i[1] == 0 for i, o in zip(in_s, out_s)):
        rh = [o[0] // i[0] for i, o in zip(in_s, out_s)]; rw = [o[1] // i[1] for i, o in zip(in_s, out_s)]
        if len(set(rh)) == 1 and len(set(rw)) == 1:
            return "INT_SCALE", (rh[0], rw[0]), None
    crop_masks = [find_subgrid_mask(np.asarray(d["input"]), np.asarray(d["output"])) for d in demos]
    if all(m is not None for m in crop_masks):
        return "CROP", (1, 1), crop_masks
    return "CONST_GLOBAL", out_s[0], None


class D4SymConv2d(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.wc = nn.Parameter(torch.randn(C, C) * (0.5 / math.sqrt(C)))
        self.wo = nn.Parameter(torch.randn(C, C) * (0.25 / math.sqrt(C)))
        self.wd = nn.Parameter(torch.randn(C, C) * (0.25 / math.sqrt(C)))
        mc = torch.zeros(3, 3); mc[1, 1] = 1.0
        mo = torch.tensor([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]])
        md = torch.tensor([[1.,0.,1.],[0.,0.,0.],[1.,0.,1.]])
        self.register_buffer("mc", mc); self.register_buffer("mo", mo); self.register_buffer("md", md)

    def forward(self, x_bkhw):
        w = (self.wc[:, :, None, None] * self.mc +
             self.wo[:, :, None, None] * self.mo +
             self.wd[:, :, None, None] * self.md)
        return F.conv2d(x_bkhw, w, padding=1)


class FastARCBlock(nn.Module):
    def __init__(self, C, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(C, eps=1e-5)
        self.conv = D4SymConv2d(C)
        self.attn_line = SelfAttnSet(C, heads, qk_norm=True)
        self.ln2 = nn.LayerNorm(C, eps=1e-5)
        self.attn_slot = SelfAttnSet(C, heads, qk_norm=True)
        self.ln3 = nn.LayerNorm(C, eps=1e-5)
        self.mlp = nn.Sequential(nn.Linear(C, 2 * C), nn.GELU(), nn.Linear(2 * C, C))

    def forward(self, h, H: int, W: int):
        B, N, K, C = h.shape; u = self.ln1(h)
        oc = self.conv(u.permute(0, 2, 3, 1).reshape(B * K, C, H, W)).reshape(B, K, C, N).permute(0, 3, 1, 2)
        g = u.reshape(B, H, W, K, C)
        orow = self.attn_line(g.permute(0, 1, 3, 2, 4).reshape(B * H * K, W, C)).reshape(B, H, K, W, C).permute(0, 1, 3, 2, 4).reshape(B, N, K, C)
        ocol = self.attn_line(g.permute(0, 2, 3, 1, 4).reshape(B * W * K, H, C)).reshape(B, W, K, H, C).permute(0, 3, 1, 2, 4).reshape(B, N, K, C)
        h = h + oc + 0.5 * (orow + ocol)
        h = h + self.attn_slot(self.ln2(h).reshape(B * N, K, C)).reshape(B, N, K, C)
        h = h + self.mlp(self.ln3(h))
        return h


class ARCCompNet(nn.Module):
    def __init__(self, C=32, heads=4, L=2):
        super().__init__()
        self.Win = nn.Linear(16, C)
        self.color_bias = nn.Parameter(torch.zeros(10, C))
        self.mu_proj = nn.Linear(C, C)
        self.lv_proj = nn.Linear(C, C)
        nn.init.constant_(self.lv_proj.bias, -5.0)  # small initial std ~0.08 (not 1.0!)
        self.blocks = nn.ModuleList([FastARCBlock(C, heads) for _ in range(L)])
        self.lnf = nn.LayerNorm(C, eps=1e-5)
        self.pix_head = nn.Linear(C, 1)
        self.crop_head = nn.Linear(C, 1)
        self.global_dec = nn.Sequential(nn.Linear(2 * C + 4, C), nn.GELU(), nn.Linear(C, 10))

    def build_features(self, grids_np_list, obj_feats_list, device):
        B = len(grids_np_list); H, W = grids_np_list[0].shape; N = H * W
        coords = _hw_cache(H, W, str(device))
        gt = torch.as_tensor(np.stack([g.reshape(-1) for g in grids_np_list]), dtype=torch.long, device=device)
        ob = torch.as_tensor(np.stack(obj_feats_list), dtype=torch.float32, device=device)
        in_oh = F.one_hot(gt, 10).float()
        is_z = torch.zeros(B, N, 10, device=device); is_z[:, :, 0] = 1.0
        hist = in_oh.mean(dim=1, keepdim=True).expand(B, N, 10)
        return torch.cat([in_oh.unsqueeze(-1), is_z.unsqueeze(-1), hist.unsqueeze(-1),
                          ob.unsqueeze(2).expand(B, N, 10, 9), coords.view(1, N, 1, 4).expand(B, N, 10, 4)], dim=-1)

    def forward(self, f, H: int, W: int, T=2, sample_z=True):
        h0 = F.gelu(self.Win(f)) + self.color_bias.view(1, 1, 10, -1)
        mu = self.mu_proj(h0); lv = self.lv_proj(h0).clamp(-8.0, -3.0)
        kl = -0.5 * (1.0 + lv - mu.pow(2) - lv.exp()).mean()
        h = (mu + torch.randn_like(mu) * torch.exp(0.5 * lv)) if (sample_z and self.training) else mu
        for _ in range(T):
            h = 0.5 * h + 0.5 * h0
            for blk in self.blocks:
                h = blk(h, H, W)
        z = self.lnf(h)
        return self.pix_head(z).squeeze(-1), self.crop_head(z.mean(2)).squeeze(-1), z, kl

    def decode_global(self, z, Ho: int, Wo: int):
        B = z.shape[0]; zm = z.mean(2); g = torch.cat([zm.mean(1), zm.amax(1)], dim=-1)
        coords_o = _hw_cache(Ho, Wo, str(z.device))
        return self.global_dec(torch.cat([g.unsqueeze(1).expand(B, Ho * Wo, -1), coords_o.unsqueeze(0).expand(B, Ho * Wo, 4)], dim=-1))


def apply_d4(g, k_rot: int, flip: bool):
    a = np.asarray(g, dtype=np.int64)
    if k_rot: a = np.rot90(a, k_rot)
    if flip: a = np.fliplr(a)
    return a.copy()


def inv_d4(g, k_rot: int, flip: bool):
    a = np.asarray(g, dtype=np.int64)
    if flip: a = np.fliplr(a)
    if k_rot: a = np.rot90(a, 4 - k_rot)
    return a.copy()


def run_single_view(task, k_rot: int, flip: bool, device, steps=140, C=32, L=2):
    demos = [{"input": apply_d4(d["input"], k_rot, flip), "output": apply_d4(d["output"], k_rot, flip)} for d in task["train"]]
    test_in = apply_d4(task["test"][0]["input"], k_rot, flip)
    mode, param, crop_masks = detect_task_mode(demos)

    by_in_hw = {}
    for idx_d, d in enumerate(demos):
        di, do = d["input"], d["output"]
        di_use = np.repeat(np.repeat(di, param[0], axis=0), param[1], axis=1) if mode == "INT_SCALE" else di
        hw = di_use.shape; ob = extract_object_features(di_use)
        tgt_out = torch.as_tensor(do.reshape(-1), dtype=torch.long, device=device)
        tgt_crop = torch.as_tensor(crop_masks[idx_d], dtype=torch.float32, device=device) if mode == "CROP" else None
        by_in_hw.setdefault(hw, []).append((di_use, ob, tgt_out, tgt_crop, do.shape))

    torch.manual_seed(k_rot)
    net = ARCCompNet(C=C, heads=4, L=L).to(device)
    groups = []
    for hw, lst in by_in_hw.items():
        f_b = net.build_features([x[0] for x in lst], [x[1] for x in lst], device).detach()
        groups.append((hw[0], hw[1], f_b, lst))

    opt = torch.optim.AdamW(net.parameters(), lr=1.5e-2, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    net.train(); demos_fit = False
    for st in range(steps):
        opt.zero_grad(); tot_l = 0.0; all_ok = True
        for Hi, Wi, f_b, lst in groups:
            pix_lg, crop_lg, z, kl = net(f_b, Hi, Wi, T=2, sample_z=(st < steps - 40))
            if mode in ("SAME_HW", "INT_SCALE"):
                ygt = torch.stack([x[2] for x in lst])
                loss_g = F.cross_entropy(pix_lg.reshape(-1, 10), ygt.reshape(-1))
                all_ok &= bool((pix_lg.argmax(-1) == ygt).all())
            elif mode == "CROP":
                cm = torch.stack([x[3] for x in lst])
                loss_g = F.binary_cross_entropy_with_logits(crop_lg, cm)
                all_ok &= bool(((crop_lg > 0) == (cm > 0.5)).all())
            else:
                loss_g = 0.0
                for k_i, item in enumerate(lst):
                    ho_i, wo_i = item[4]
                    glg = net.decode_global(z[k_i:k_i+1], ho_i, wo_i)
                    ygt_i = item[2].unsqueeze(0)
                    loss_g = loss_g + F.cross_entropy(glg.reshape(-1, 10), ygt_i.reshape(-1)) / len(lst)
                    all_ok &= bool((glg.argmax(-1) == ygt_i).all())
            tot_l = tot_l + (loss_g + 0.002 * kl) / len(groups)
        tot_l.backward(); opt.step(); sched.step()
        if all_ok:
            demos_fit = True
            if st >= 80: break

    net.eval()
    with torch.no_grad():
        ti_use = np.repeat(np.repeat(test_in, param[0], axis=0), param[1], axis=1) if mode == "INT_SCALE" else test_in
        f_t = net.build_features([ti_use], [extract_object_features(ti_use)], device)
        pix_t, crop_t, z_t, _ = net(f_t, ti_use.shape[0], ti_use.shape[1], T=2, sample_z=False)
        if mode in ("SAME_HW", "INT_SCALE"):
            pred = pix_t[0].argmax(-1).cpu().numpy().reshape(ti_use.shape)
        elif mode == "CROP":
            cm_p = (crop_t[0].cpu().numpy().reshape(ti_use.shape) > 0)
            if cm_p.any():
                rs, cs = np.where(cm_p); pred = ti_use[rs.min():rs.max()+1, cs.min():cs.max()+1].copy()
            else:
                pred = ti_use
        else:
            Ho, Wo = param; pred = net.decode_global(z_t, Ho, Wo)[0].argmax(-1).cpu().numpy().reshape(Ho, Wo)
    return inv_d4(pred, k_rot, flip), demos_fit


def solve_task(task, device, steps=140):
    fit_cands, all_cands = {}, {}; any_fit = False
    for k_rot in range(2):
        pred, d_fit = run_single_view(task, k_rot, False, device, steps=steps)
        key = (pred.shape, pred.tobytes())
        all_cands[key] = (all_cands.get(key, (0, pred))[0] + 1, pred)
        if d_fit:
            any_fit = True
            fit_cands[key] = (fit_cands.get(key, (0, pred))[0] + 1, pred)
    pool = fit_cands if fit_cands else all_cands
    ranked = sorted(pool.values(), key=lambda x: x[0], reverse=True)
    top2 = [x[1] for x in ranked[:2]]
    if len(top2) == 1: top2.append(top2[0])
    return top2, any_fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=25)
    ap.add_argument("--steps", type=int, default=140)
    ap.add_argument("--out", default="runs/arc_v53")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    eval_tasks = load_arc_eval(Path("cache/thirdparty"))
    keys = sorted(eval_tasks.keys())[:args.tasks]

    t_b = time.time()
    for tid in keys[:2]: solve_task(eval_tasks[tid], dev, steps=args.steps)
    sec_per_task = (time.time() - t_b) / 2.0
    print(f"ARC-AGI-1 v{__version__} | Measured speed: {sec_per_task:.2f}s/task (est ~{sec_per_task*len(keys)/60:.1f} min for {len(keys)} tasks) | {dev}", flush=True)

    p1 = p2 = fit_cnt = 0; t0 = time.time(); solved_ids = []
    for i, tid in enumerate(keys):
        tk = eval_tasks[tid]
        top2, d_fit = solve_task(tk, dev, steps=args.steps)
        true_out = np.asarray(tk["test"][0]["output"], dtype=np.int64)
        ok1 = (top2[0].shape == true_out.shape and np.array_equal(top2[0], true_out))
        ok2 = ok1 or (top2[1].shape == true_out.shape and np.array_equal(top2[1], true_out))
        p1 += int(ok1); p2 += int(ok2); fit_cnt += int(d_fit)
        if ok2: solved_ids.append(tid)
        if (i + 1) % 5 == 0:
            print(f"   [{i+1:3d}/{len(keys)}] pass@1={p1/(i+1)*100:5.1f}% pass@2={p2/(i+1)*100:5.1f}% ({p2}/{i+1}) | demos_fit={fit_cnt}/{i+1} ({time.time()-t0:.1f}s)", flush=True)

    print(f"\n[FINAL WITHOUT PRETRAINING] n={len(keys)} | pass@1={p1/len(keys)*100:.2f}% | pass@2={p2/len(keys)*100:.2f}% ({p2}/{len(keys)}) | demos_fit={fit_cnt}/{len(keys)}", flush=True)
    print(f"   Solved IDs: {solved_ids}", flush=True)
    (out / "arc_v53_results.json").write_text(json.dumps(dict(n=len(keys), pass1=p1/len(keys), pass2=p2/len(keys), demos_fit=fit_cnt, solved_ids=solved_ids), indent=2))


if __name__ == "__main__":
    main()