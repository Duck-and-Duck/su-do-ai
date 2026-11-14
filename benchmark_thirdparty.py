#!/usr/bin/env python3
"""
Third-Party Official Benchmark Evaluator (Zero Cherry-Picking, Zero Leakage):
  A) HuggingFace sapientinc/sudoku-extreme (official test split used by HRM & TRM)
  B) DeepMind google-deepmind/boxoban-levels (official 10x10 test & hard splits)
"""
import csv, io, json, time, urllib.request
from pathlib import Path
import numpy as np
import torch

import sudoku_ai as su
import sokoban_ai as sk

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
    print(f"[download] streaming first {n} puzzles from sapientinc/sudoku-extreme/test.csv ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    P, S, C = [], [], []
    with urllib.request.urlopen(req, timeout=30) as resp:
        text_stream = io.TextIOWrapper(resp, encoding="utf-8")
        reader = csv.reader(text_stream)
        header = next(reader)
        print("   CSV header:", header)
        for row in reader:
            # Find the 81-char question (with . or 0) and 81-char answer (all 1..9)
            q_str = a_str = None
            for col in row:
                c = col.strip()
                if len(c) == 81 and all(ch in "0123456789." for ch in c):
                    if "." in c or "0" in c:
                        q_str = c
                    elif all(ch in "123456789" for ch in c):
                        a_str = c
            if q_str and a_str:
                p = np.array([0 if ch in ".0" else int(ch) for ch in q_str], dtype=np.int8)
                s = np.array([int(ch) for ch in a_str], dtype=np.int8)
                if su.is_complete_valid(s) and ((p == 0) | (p == s)).all():
                    P.append(p); S.append(s); C.append(int((p > 0).sum()))
                    if len(P) >= n:
                        break
    P, S, C = np.array(P), np.array(S), np.array(C)
    np.savez_compressed(f, p=P, s=S, clues=C)
    return P, S, C


def parse_boxoban_file(url, cache_name, limit=100):
    cf = CACHE / cache_name
    if not cf.exists():
        print(f"[download] {url} ...")
        urllib.request.urlretrieve(url, cf)
    lines = cf.read_text(encoding="utf-8").splitlines()
    levels = []
    cur = []
    for ln in lines:
        if ln.startswith(";"):
            if len(cur) == 10:
                levels.append(cur)
                if len(levels) >= limit:
                    break
            cur = []
        elif len(ln) >= 10:
            cur.append(ln[:10])
    if len(cur) == 10 and len(levels) < limit:
        levels.append(cur)
    WL, GL, BL, PL = [], [], [], []
    for grid in levels:
        w = np.zeros(100, dtype=bool); g = np.zeros(100, dtype=bool); b = np.zeros(100, dtype=bool)
        pl = -1
        for r in range(10):
            for c in range(10):
                ch = grid[r][c]
                i = r * 10 + c
                if ch == "#": w[i] = True
                if ch in (".", "*", "+"): g[i] = True
                if ch in ("$", "*"): b[i] = True
                if ch in ("@", "+"): pl = i
        if pl >= 0 and b.sum() == g.sum() == 4:
            WL.append(w); GL.append(g); BL.append(b); PL.append(pl)
    return np.array(WL), np.array(GL), np.array(BL), np.array(PL)


def run_sudoku_benchmark(ckpt_path="runs/rtx3050_v5_2/weights.pt", n=200):
    print("\n" + "=" * 78 + "\nTHIRD-PARTY BENCHMARK 1: sapientinc/sudoku-extreme (Official Test Split)\n" + "=" * 78)
    P, S, C = load_sudoku_extreme(n=n)
    # Leakage check against local training pools
    tr_sols = set()
    for npz in Path("cache").glob("pool_train*.npz"):
        z = np.load(npz)
        for s in z["s"]:
            tr_sols.add(np.asarray(s, dtype=np.int8).tobytes())
    leak = sum(np.asarray(s, dtype=np.int8).tobytes() in tr_sols for s in S)
    singles_ok = sum(su.singles_solve(p)[0] for p in P)
    print(f"[data] n={len(P)} puzzles | clues avg={C.mean():.1f} (min={C.min()}, max={C.max()}) | "
          f"singles-solvable={singles_ok}/{len(P)} | train leakage={leak}/{len(P)}")

    ck = torch.load(ckpt_path, map_location=DEV, weights_only=False)
    sa = ck["args"]
    net = su.SudokuNet(C=sa["width"], heads=sa["heads"], L=sa["layers"], T_max=24).to(DEV).eval()
    sd = {k: v for k, v in ck["model"].items() if not k.startswith(("ug_", "ud_", "inv_"))}
    net.load_state_dict(sd, strict=False)
    npar = ck["n_params"]

    # Pure System 1 (greedy + pass@2)
    mdp = su.MDP(max_steps=81, shaping=False)
    e = su.BatchEnv(P, S, DEV, mdp)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(81):
            live = ~e.done
            if not bool(live.any()): break
            li = torch.nonzero(live, as_tuple=True)[0]
            with su._amp(DEV):
                o = net(e.b[li], e.givens[li], T=4)
            lg_l, _ = su.policy_logits(o, e.b[li], lm=e.legal()[li])
            a = torch.zeros(len(P), dtype=torch.long, device=DEV)
            a[li] = lg_l.argmax(-1)
            e.step(a)
    s1_acc = float(e.solved.cpu().numpy().mean())

    # System 1 + System 2 (simulate, max 64 nodes)
    s2_ok, s2_sims = [], []
    for p in P:
        ok, sims = su.simulate_solve(net, p, DEV, T=4, conf_thresh=0.95, max_sim_nodes=64)
        s2_ok.append(ok); s2_sims.append(sims)
    s2_acc = float(np.mean(s2_ok))
    dt = time.time() - t0
    print(f"[RESULT] Sudoku-AGI ({npar:,} params, ZERO training on sudoku-extreme, {dt:.1f}s):")
    print(f"   Pure System 1 (0 search): {s1_acc*100:.2f}% ({int(e.solved.sum())}/{len(P)})")
    print(f"   System 1+2 (avg {np.mean(s2_sims):.1f} sims): {s2_acc*100:.2f}% ({sum(s2_ok)}/{len(P)})")
    print("   Published baselines: Frontier LLMs 0.0% | HRM(27M) 55.0% | TRM-Att(7M) 74.7% | TRM-MLP(5M) 87.4% | PTRM(7M) 98.75%")
    return dict(n=len(P), clues_avg=float(C.mean()), leakage=leak, sys1=s1_acc, sys2=s2_acc, avg_sims=float(np.mean(s2_sims)))


def run_boxoban_benchmark(ckpt_path="runs/soko_v2/weights.pt", n_unfilt=100, n_hard=50):
    print("\n" + "=" * 78 + "\nTHIRD-PARTY BENCHMARK 2: DeepMind Boxoban (Official 10x10 Test & Hard)\n" + "=" * 78)
    ck = torch.load(ckpt_path, map_location=DEV, weights_only=False)
    sa = ck["args"]
    net10 = sk.SokobanNet(H=10, W=10, C=sa["width"], heads=sa["heads"], L=sa["layers"], T_cyc=8).to(DEV).eval()
    sd = {k: v for k, v in ck["model"].items() if not k.startswith(("ug_", "ud_", "inv_"))}
    net10.load_state_dict(sd, strict=False)
    print(f"[model] Loaded 8x8 checkpoint ({ck['n_params']:,} params) -> ZERO-SHOT 10x10 transfer (0 extra steps!)")

    results = {}
    splits = [
        ("unfiltered_test", "https://raw.githubusercontent.com/google-deepmind/boxoban-levels/master/unfiltered/test/000.txt", n_unfilt),
        ("hard", "https://raw.githubusercontent.com/google-deepmind/boxoban-levels/master/hard/000.txt", n_hard),
    ]
    best_vid = None
    for name, url, lim in splits:
        wl, gl, bl, pl = parse_boxoban_file(url, f"boxoban_{name}.txt", limit=lim)
        mdp = sk.SokoMDP(max_steps=80, shaping=False)
        # Pure Sys1 greedy
        e = sk.SokoBatchEnv(10, 10, wl, gl, bl, pl, DEV, mdp)
        with torch.no_grad():
            for _ in range(80):
                live = ~e.done
                if not bool(live.any()): break
                li = torch.nonzero(live, as_tuple=True)[0]
                with sk._amp(DEV):
                    o = net10(e.walls[li], e.goals[li], e.boxes[li], e.players[li], T=4)
                lg_l, _ = sk.policy_logits(o, e.legal()[li])
                a = torch.zeros(len(wl), dtype=torch.long, device=DEV)
                a[li] = lg_l.argmax(-1)
                e.step(a)
        s1_acc = float(e.solved.cpu().numpy().mean())

        # Sys1+2 (max 64 neural nodes)
        s2_ok, s2_n = [], []
        for j in range(len(wl)):
            ok_j, n_j, tr_j = sk.simulate_solve(net10, 10, 10, wl[j], gl[j], bl[j], pl[j], DEV,
                                                T=4, max_steps=80, max_nodes=64)
            s2_ok.append(ok_j); s2_n.append(n_j)
            if ok_j and (best_vid is None or (name == "hard" and best_vid[0] != "hard") or len(tr_j) > best_vid[1]):
                best_vid = (name, len(tr_j), f"Boxoban {name} ({n_j}n)", wl[j], gl[j], tr_j)
        s2_acc = float(np.mean(s2_ok))
        print(f"   [{name:<15}] n={len(wl):3d} | Zero-Shot Sys1={s1_acc*100:5.1f}% | Sys1+2(64n)={s2_acc*100:5.1f}% (avg {np.mean(s2_n):.1f} nodes)")
        results[name] = dict(n=len(wl), sys1=s1_acc, sys2=s2_acc, avg_nodes=float(np.mean(s2_n)))
    print("   Published 10x10 baselines (trained on 900k Boxoban levels, 1e9 steps): MCTSnets 84.0% | DRC(3,3, 1.29M) 99.0% unfilt / 42.8% hard")
    if best_vid is not None:
        _nm, _ln, lbl, w_v, g_v, tr_v = best_vid
        sk.render_episode_video(10, 10, w_v, g_v, tr_v, lbl, OUT / "boxoban_play")
    return results


if __name__ == "__main__":
    su_res = run_sudoku_benchmark()
    bk_res = run_boxoban_benchmark()
    (OUT / "thirdparty_results.json").write_text(json.dumps(dict(sudoku_extreme=su_res, boxoban=bk_res), indent=2))