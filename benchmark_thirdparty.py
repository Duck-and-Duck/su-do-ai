#!/usr/bin/env python3
"""Zero-Shot Third-Party Benchmark: sapientinc/sudoku-extreme (FocusDiff, PTRM, WM-Tree, Sys2)"""
import csv, io, json, time, urllib.request
from pathlib import Path
import numpy as np
import torch
import sudoku_ai as su

OUT = Path("runs/thirdparty"); CACHE = Path("cache/thirdparty")
OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_sudoku_extreme(n=200):
    f = CACHE / f"sudoku_extreme_test_{n}.npz"
    if f.exists():
        z = np.load(f); return z["p"], z["s"], z["clues"]
    url = "https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/main/test.csv"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    P, S, C = [], [], []
    with urllib.request.urlopen(req, timeout=30) as resp:
        reader = csv.reader(io.TextIOWrapper(resp, encoding="utf-8")); next(reader)
        for row in reader:
            q_str = a_str = None
            for col in row:
                c = col.strip()
                if len(c) == 81 and all(ch in "0123456789." for ch in c):
                    if "." in c or "0" in c: q_str = c
                    elif all(ch in "123456789" for ch in c): a_str = c
            if q_str and a_str:
                p = np.array([0 if ch in ".0" else int(ch) for ch in q_str], dtype=np.int8)
                s = np.array([int(ch) for ch in a_str], dtype=np.int8)
                if su.is_complete_valid(s) and ((p == 0) | (p == s)).all():
                    P.append(p); S.append(s); C.append(int((p > 0).sum()))
                    if len(P) >= n: break
    P, S, C = np.array(P), np.array(S), np.array(C); np.savez_compressed(f, p=P, s=S, clues=C)
    return P, S, C


def run_sudoku(ckpt_path="runs/sudoku_v7/weights.pt", n=200, n_sub=100, sim_nodes=32):
    print("\n" + "=" * 78 + f"\nZERO-SHOT BENCHMARK: sapientinc/sudoku-extreme ({ckpt_path})\n" + "=" * 78, flush=True)
    P, S, C = load_sudoku_extreme(n=n)
    tr_sols = set()
    for npz in Path("cache").glob("pool_train*.npz"):
        z = np.load(npz)
        for s in z["s"]: tr_sols.add(np.asarray(s, dtype=np.int8).tobytes())
    leak = sum(np.asarray(s, dtype=np.int8).tobytes() in tr_sols for s in S)
    is_singles = np.array([su.singles_solve(p)[0] for p in P])
    print(f"[data] n={len(P)} | clues avg={C.mean():.1f} ({C.min()}..{C.max()}) | 0-singles={(~is_singles).sum()}/{len(P)} | leakage={leak}/{len(P)}", flush=True)

    ck = torch.load(ckpt_path, map_location=DEV, weights_only=False); sa = ck["args"]
    net = su.SudokuNet(C=sa["width"], heads=sa["heads"], L=sa["layers"], qk_norm=bool(sa.get("qk_norm", False))).to(DEV).eval()
    net.load_state_dict({k: v for k, v in ck["model"].items() if not k.startswith(("ug_", "ud_", "inv_"))}, strict=False)
    npar = ck["n_params"]

    t0 = time.time()
    fb_ok, _ = su.solve_focus_diffusion(net, P, DEV, n_rounds=8, t_seg=3, n_noisy=1, batch_sz=64)
    print(f"   [1. Focusing Diffusion (0 search, {time.time()-t0:.1f}s)]    All={fb_ok.mean()*100:.2f}% | 0-Singles={fb_ok[~is_singles].mean()*100:.2f}%", flush=True)

    t1 = time.time()
    ptrm_ok, _ = su.solve_focus_diffusion(net, P, DEV, n_rounds=8, t_seg=3, n_noisy=32, noise_std=0.35, batch_sz=8)
    print(f"   [2. Relative PTRM K=32 (0 search, {time.time()-t1:.1f}s)]   All={ptrm_ok.mean()*100:.2f}% | 0-Singles={ptrm_ok[~is_singles].mean()*100:.2f}%", flush=True)

    # Pure Latent World-Model Tree Imagining (0 Python rules inside imagination!)
    t2 = time.time()
    wm_ok = []
    for i, p in enumerate(P[:n_sub]):
        if ptrm_ok[i]:
            wm_ok.append(True)
        else:
            ok_w, _ = su.solve_worldmodel_tree(net, p, DEV, T=4, beam_w=3, depth=2)
            wm_ok.append(ok_w)
    wm_arr = np.array(wm_ok); ns_sub = ~is_singles[:n_sub]
    print(f"   [3. Latent WorldModel Tree (0 rules, {time.time()-t2:.1f}s)] All={wm_arr.mean()*100:.2f}% | 0-Singles={wm_arr[ns_sub].mean()*100:.2f}%", flush=True)

    # Full Hybrid Sys1+2
    t3 = time.time()
    s2_ok, s2_sims = [], []
    for i, p in enumerate(P[:n_sub]):
        if wm_arr[i]:
            s2_ok.append(True); s2_sims.append(0)
        else:
            ok, sims = su.simulate_solve(net, p, DEV, T=4, conf_thresh=0.95, max_sim_nodes=sim_nodes)
            s2_ok.append(ok); s2_sims.append(sims)
        if (i + 1) % 25 == 0:
            print(f"   [Sys1+2 progress] {i+1}/{n_sub}: {np.mean(s2_ok)*100:.1f}% ({time.time()-t3:.1f}s)", flush=True)
    s2_arr = np.array(s2_ok)
    print(f"   [4. Hybrid + Sys2 ({sim_nodes}n, avg {np.mean(s2_sims):.1f} sims)] All={s2_arr.mean()*100:.2f}% | 0-Singles={s2_arr[ns_sub].mean()*100:.2f}%", flush=True)
    print(f"   Params: {npar:,} | Global refs: LLMs 0.0% | HRM(27M) 55.0% | TRM-Att(7M) 74.7% | TRM-MLP(5M) 87.4% | PTRM(7M) 98.75%", flush=True)
    res = dict(n=len(P), leakage=leak, focus_all=float(fb_ok.mean()), focus_0s=float(fb_ok[~is_singles].mean()),
               ptrm32_all=float(ptrm_ok.mean()), ptrm32_0s=float(ptrm_ok[~is_singles].mean()),
               wm_tree_all=float(wm_arr.mean()), wm_tree_0s=float(wm_arr[ns_sub].mean()),
               sys2_all=float(s2_arr.mean()), sys2_0s=float(s2_arr[ns_sub].mean()), avg_sims=float(np.mean(s2_sims)))
    (OUT / "thirdparty_results.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    run_sudoku()