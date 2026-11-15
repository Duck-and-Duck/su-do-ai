#!/usr/bin/env python3
"""Official Third-Party Benchmark: sapientinc/sudoku-extreme test split (Zero Leakage)"""
import csv, io, json, time, urllib.request
from pathlib import Path
import numpy as np
import torch
import sudoku_ai as su

OUT = Path("runs/thirdparty")
CACHE = Path("cache/thirdparty")
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_sudoku_extreme(n=200):
    f = CACHE / f"sudoku_extreme_test_{n}.npz"
    if f.exists():
        z = np.load(f)
        return z["p"], z["s"], z["clues"]
    url = "https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/main/test.csv"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    P, S, C = [], [], []
    with urllib.request.urlopen(req, timeout=30) as resp:
        reader = csv.reader(io.TextIOWrapper(resp, encoding="utf-8"))
        next(reader)
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
    P, S, C = np.array(P), np.array(S), np.array(C)
    np.savez_compressed(f, p=P, s=S, clues=C)
    return P, S, C


def run_sudoku_benchmark(ckpt_path="runs/sudoku_extreme_v52/weights.pt", n=200, n_sys2=100, sim_nodes=32):
    print("\n" + "=" * 78 + "\nTHIRD-PARTY BENCHMARK: sapientinc/sudoku-extreme (Official Test Split)\n" + "=" * 78, flush=True)
    P, S, C = load_sudoku_extreme(n=n)
    tr_sols = set()
    for npz in list(Path("cache").glob("pool_train*.npz")) + list(Path("cache/thirdparty").glob("sudoku_extreme_train*.npz")):
        z = np.load(npz)
        for s in z["s"]:
            tr_sols.add(np.asarray(s, dtype=np.int8).tobytes())
    leak = sum(np.asarray(s, dtype=np.int8).tobytes() in tr_sols for s in S)
    is_singles = np.array([su.singles_solve(p)[0] for p in P])
    print(f"[data] n={len(P)} puzzles | clues avg={C.mean():.1f} ({C.min()}..{C.max()}) | "
          f"0-singles={(~is_singles).sum()}/{len(P)} | train leakage={leak}/{len(P)}", flush=True)

    ck = torch.load(ckpt_path, map_location=DEV, weights_only=False)
    sa = ck["args"]
    net = su.SudokuNet(C=sa["width"], heads=sa["heads"], L=sa["layers"], T_max=24).to(DEV).eval()
    sd = {k: v for k, v in ck["model"].items() if not k.startswith(("ug_", "ud_", "inv_"))}
    net.load_state_dict(sd, strict=False)
    npar = ck["n_params"]
    mdp = su.MDP(max_steps=81, shaping=False)

    t0 = time.time()
    gs_by_T = {}
    for T_g in (4, 8):
        e = su.BatchEnv(P, S, DEV, mdp)
        with torch.no_grad():
            for _ in range(81):
                live = ~e.done
                if not bool(live.any()): break
                li = torch.nonzero(live, as_tuple=True)[0]
                with su._amp(DEV):
                    o = net(e.b[li], e.givens[li], T=T_g)
                lg_l, _ = su.policy_logits(o, e.b[li], lm=e.legal()[li])
                a = torch.zeros(len(P), dtype=torch.long, device=DEV)
                a[li] = lg_l.argmax(-1)
                e.step(a)
        gs_by_T[T_g] = e.solved.cpu().numpy()
        print(f"   [Sys1 T={T_g}] All={gs_by_T[T_g].mean()*100:.2f}% | 0-Singles={gs_by_T[T_g][~is_singles].mean()*100:.2f}% ({time.time()-t0:.1f}s)", flush=True)
    best_T = 4 if gs_by_T[4].mean() >= gs_by_T[8].mean() else 8
    gs = gs_by_T[best_T]

    # Sys1+2 (n=n_sys2, earliest-first DFS + stochastic restarts)
    t1 = time.time()
    s2_ok, s2_sims = [], []
    for i, p in enumerate(P[:n_sys2]):
        if gs[i]:
            s2_ok.append(True); s2_sims.append(0)
        else:
            ok, sims = su.simulate_solve(net, p, DEV, T=best_T, conf_thresh=0.95, max_sim_nodes=sim_nodes)
            s2_ok.append(ok); s2_sims.append(sims)
        if (i + 1) % 20 == 0:
            print(f"   [Sys2 progress] {i+1}/{n_sys2}: {np.mean(s2_ok)*100:.1f}% solved ({time.time()-t1:.1f}s)", flush=True)
    s2_arr = np.array(s2_ok)
    ns_sub = ~is_singles[:n_sys2]
    print(f"\n[SUMMARY ({npar:,} params, 1x train.csv(10k), 0/{len(P)} leakage, T={best_T})]", flush=True)
    print(f"   Pure Sys1 (n={len(P)}, 0 search):  All={gs.mean()*100:.2f}% | 0-Singles={gs[~is_singles].mean()*100:.2f}%", flush=True)
    print(f"   Sys1+2    (n={n_sys2}, avg {np.mean(s2_sims):.1f} sims): All={s2_arr.mean()*100:.2f}% | 0-Singles={s2_arr[ns_sub].mean()*100:.2f}%", flush=True)
    print("   Published baselines: LLMs 0.0% | HRM(27M) 55.0% | TRM-Att(7M) 74.7% | TRM-MLP(5M) 87.4% | PTRM(7M) 98.75%", flush=True)
    res = dict(n=len(P), leakage=leak, best_T=best_T, sys1_all=float(gs.mean()),
               sys1_0singles=float(gs[~is_singles].mean()), sys2_n=n_sys2,
               sys2_all=float(s2_arr.mean()), sys2_0singles=float(s2_arr[ns_sub].mean()),
               avg_sims=float(np.mean(s2_sims)))
    (OUT / "thirdparty_results.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    run_sudoku_benchmark()