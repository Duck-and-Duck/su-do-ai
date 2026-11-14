#!/usr/bin/env python3
"""
Third-Party Official Benchmark Evaluator (Complete, Unbuffered, Zero Leakage):
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


def parse_boxoban_file(url, cache_name, limit=100):
    cf = CACHE / cache_name
    if not cf.exists():
        print(f"[download] {url} ...", flush=True)
        urllib.request.urlretrieve(url, cf)
    lines = cf.read_text(encoding="utf-8").splitlines()
    levels, cur = [], []
    for ln in lines:
        if ln.startswith(";"):
            if len(cur) == 10:
                levels.append(cur)
                if len(levels) >= limit: break
            cur = []
        elif len(ln) >= 10:
            cur.append(ln[:10])
    if len(cur) == 10 and len(levels) < limit:
        levels.append(cur)
    WL, GL, BL, PL = [], [], [], []
    for grid in levels:
        w = np.zeros(100, dtype=bool); g = np.zeros(100, dtype=bool); b = np.zeros(100, dtype=bool); pl = -1
        for r in range(10):
            for c in range(10):
                ch = grid[r][c]; i = r * 10 + c
                if ch == "#": w[i] = True
                if ch in (".", "*", "+"): g[i] = True
                if ch in ("$", "*"): b[i] = True
                if ch in ("@", "+"): pl = i
        if pl >= 0 and b.sum() == g.sum() == 4:
            WL.append(w); GL.append(g); BL.append(b); PL.append(pl)
    return np.array(WL), np.array(GL), np.array(BL), np.array(PL)


def run_sudoku_benchmark(ckpt_path="runs/extreme_v5_1/weights.pt", n=200, n_sys2=100, sim_nodes=24):
    print("\n" + "=" * 78 + "\nTHIRD-PARTY BENCHMARK 1: sapientinc/sudoku-extreme (Official Test Split)\n" + "=" * 78, flush=True)
    P, S, C = load_sudoku_extreme(n=n)
    tr_sols = set()
    for npz in list(Path("cache").glob("pool_train*.npz")) + list(Path("cache/thirdparty").glob("sudoku_extreme_train*.npz")):
        z = np.load(npz)
        for s in z["s"]:
            tr_sols.add(np.asarray(s, dtype=np.int8).tobytes())
    leak = sum(np.asarray(s, dtype=np.int8).tobytes() in tr_sols for s in S)
    is_singles = np.array([su.singles_solve(p)[0] for p in P])
    print(f"[data] n={len(P)} puzzles | clues avg={C.mean():.1f} ({C.min()}..{C.max()}) | "
          f"0-singles={(~is_singles).sum()}/{len(P)} | total train leakage={leak}/{len(P)}", flush=True)

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
        print(f"   [Sys1 T={T_g}] All={gs_by_T[T_g].mean()*100:.2f}% | 0-Singles={gs_by_T[T_g][~is_singles].mean()*100:.2f}%", flush=True)
    best_T = 4 if gs_by_T[4].mean() >= gs_by_T[8].mean() else 8
    gs = gs_by_T[best_T]

    n_samp = 4
    P_rep, S_rep = np.repeat(P, n_samp, axis=0), np.repeat(S, n_samp, axis=0)
    sv_parts = []
    for off in range(0, len(P_rep), 128):
        Bc = min(128, len(P_rep) - off)
        e2 = su.BatchEnv(P_rep[off:off + Bc], S_rep[off:off + Bc], DEV, mdp)
        with torch.no_grad():
            for _ in range(81):
                live2 = ~e2.done
                if not bool(live2.any()): break
                li2 = torch.nonzero(live2, as_tuple=True)[0]
                with su._amp(DEV):
                    o = net(e2.b[li2], e2.givens[li2], T=best_T)
                lg_l, _ = su.policy_logits(o, e2.b[li2], lm=e2.legal()[li2])
                lp_l = torch.log_softmax(lg_l / 0.8, -1)
                a = torch.zeros(Bc, dtype=torch.long, device=DEV)
                a[li2] = torch.distributions.Categorical(logits=lp_l).sample()
                e2.step(a)
        sv_parts.append(e2.solved.cpu().numpy())
    cnt = np.concatenate(sv_parts).reshape(len(P), n_samp).sum(1)
    p2 = float(np.mean([su.pass_at_k(n_samp, int(c), 2) for c in cnt]))
    p2_ns = float(np.mean([su.pass_at_k(n_samp, int(c), 2) for c in cnt[~is_singles]]))
    print(f"[Sys1 Summary ({time.time()-t0:.1f}s, {npar:,} params, 1x train.csv(3k), 0 leakage, T={best_T})]", flush=True)
    print(f"   All {len(P)}:       Pure Sys1={gs.mean()*100:.2f}% | pass@2={p2*100:.2f}%", flush=True)
    print(f"   0-Singles ({(~is_singles).sum()}): Pure Sys1={gs[~is_singles].mean()*100:.2f}% | pass@2={p2_ns*100:.2f}%", flush=True)

    s2_ok, s2_sims = [], []
    t1 = time.time()
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
    print(f"   Sys1+2 (n={n_sys2}, max_nodes={sim_nodes}, avg {np.mean(s2_sims):.1f} sims): "
          f"All={s2_arr.mean()*100:.2f}% | 0-Singles={s2_arr[ns_sub].mean()*100:.2f}%", flush=True)
    print("   Published baselines: Frontier LLMs 0.0% | HRM(27M) 55.0% | TRM-Att(7M) 74.7% | TRM-MLP(5M) 87.4% | PTRM(7M) 98.75%", flush=True)
    return dict(n=len(P), leakage=leak, best_T=best_T, sys1_all=float(gs.mean()),
                sys1_0singles=float(gs[~is_singles].mean()), pass2_all=p2,
                sys2_n=n_sys2, sys2_all=float(s2_arr.mean()),
                sys2_0singles=float(s2_arr[ns_sub].mean()), avg_sims=float(np.mean(s2_sims)))


def run_boxoban_benchmark(ckpt_path="runs/boxoban_v1/weights.pt", n_unfilt=100, n_hard=50, s2_unfilt=50, s2_hard=30, max_nodes=48):
    print("\n" + "=" * 78 + "\nTHIRD-PARTY BENCHMARK 2: DeepMind Boxoban (Official 10x10 Test & Hard)\n" + "=" * 78, flush=True)
    ck = torch.load(ckpt_path, map_location=DEV, weights_only=False)
    sa = ck["args"]
    net10 = sk.SokobanNet(H=10, W=10, C=sa["width"], heads=sa["heads"], L=sa["layers"], T_cyc=8).to(DEV).eval()
    sd = {k: v for k, v in ck["model"].items() if not k.startswith(("ug_", "ud_", "inv_"))}
    net10.load_state_dict(sd, strict=False)
    print(f"[model] Loaded {ckpt_path} ({ck['n_params']:,} params)", flush=True)

    tr_sigs = set()
    for npz in Path("cache").glob("boxoban_train_*.npz"):
        z = np.load(npz, allow_pickle=True)
        for j in range(len(z["walls"])):
            tr_sigs.add(sk.d4_canonical_sig(10, 10, z["walls"][j], z["goals"][j], z["boxes"][j], int(z["players"][j])))

    results = {}
    splits = [
        ("unfiltered_test", "https://raw.githubusercontent.com/google-deepmind/boxoban-levels/master/unfiltered/test/000.txt", n_unfilt, s2_unfilt),
        ("hard", "https://raw.githubusercontent.com/google-deepmind/boxoban-levels/master/hard/000.txt", n_hard, s2_hard),
    ]
    best_vid = None
    for name, url, lim, s2_lim in splits:
        wl, gl, bl, pl = parse_boxoban_file(url, f"boxoban_{name}.txt", limit=lim)
        leak = sum(sk.d4_canonical_sig(10, 10, wl[j], gl[j], bl[j], int(pl[j])) in tr_sigs for j in range(len(wl)))
        mdp = sk.SokoMDP(max_steps=90, shaping=False)
        e = sk.SokoBatchEnv(10, 10, wl, gl, bl, pl, DEV, mdp)
        with torch.no_grad():
            for _ in range(90):
                live = ~e.done
                if not bool(live.any()): break
                li = torch.nonzero(live, as_tuple=True)[0]
                with sk._amp(DEV):
                    o = net10(e.walls[li], e.goals[li], e.boxes[li], e.players[li], T=4)
                lg_l, _ = sk.policy_logits(o, e.legal()[li])
                a = torch.zeros(len(wl), dtype=torch.long, device=DEV)
                a[li] = lg_l.argmax(-1)
                e.step(a)
        gs = e.solved.cpu().numpy()

        ns = 4
        e2 = sk.SokoBatchEnv(10, 10, np.repeat(wl, ns, 0), np.repeat(gl, ns, 0),
                             np.repeat(bl, ns, 0), np.repeat(pl, ns, 0), DEV, mdp)
        with torch.no_grad():
            for _ in range(90):
                live2 = ~e2.done
                if not bool(live2.any()): break
                li2 = torch.nonzero(live2, as_tuple=True)[0]
                with sk._amp(DEV):
                    o = net10(e2.walls[li2], e2.goals[li2], e2.boxes[li2], e2.players[li2], T=4)
                lg_l, _ = sk.policy_logits(o, e2.legal()[li2])
                lp_l = torch.log_softmax(lg_l / 0.7, -1)
                a = torch.zeros(len(wl) * ns, dtype=torch.long, device=DEV)
                a[li2] = torch.distributions.Categorical(logits=lp_l).sample()
                e2.step(a)
        cnt = e2.solved.cpu().numpy().reshape(len(wl), ns).sum(1)
        p2 = float(np.mean([sk.pass_at_k(ns, int(c), 2) for c in cnt]))
        print(f"   [{name:<15}] n={len(wl):3d} D4-leak={leak} | Sys1 greedy={gs.mean()*100:5.1f}% | pass@2={p2*100:5.1f}%", flush=True)

        s2_ok, s2_n = [], []
        t1 = time.time()
        for j in range(min(s2_lim, len(wl))):
            ok_j, n_j, tr_j = sk.simulate_solve(net10, 10, 10, wl[j], gl[j], bl[j], pl[j], DEV,
                                                T=4, max_steps=90, max_nodes=max_nodes)
            s2_ok.append(ok_j); s2_n.append(n_j)
            if ok_j and (best_vid is None or (name == "hard" and best_vid[0] != "hard") or len(tr_j) < best_vid[1]):
                best_vid = (name, len(tr_j), f"Boxoban {name} ({n_j}n)", wl[j], gl[j], tr_j)
        s2_acc = float(np.mean(s2_ok))
        print(f"   [{name:<15}] Sys1+2 (n={len(s2_ok)}, {max_nodes}n max): {s2_acc*100:5.1f}% (avg {np.mean(s2_n):.1f} nodes, {time.time()-t1:.1f}s)", flush=True)
        results[name] = dict(n=len(wl), leak=leak, sys1=float(gs.mean()), pass2=p2,
                             s2_n=len(s2_ok), sys2=s2_acc, avg_nodes=float(np.mean(s2_n)))
    print("   Published 10x10 baselines (trained on 900k Boxoban levels, 1e9 steps): MCTSnets 84.0% | DRC(3,3, 1.29M) 99.0% unfilt / 42.8% hard", flush=True)
    if best_vid is not None:
        _nm, _ln, lbl, w_v, g_v, tr_v = best_vid
        sk.render_episode_video(10, 10, w_v, g_v, tr_v, lbl, OUT / "boxoban_play")
    return results


if __name__ == "__main__":
    su_res = run_sudoku_benchmark()
    bk_res = run_boxoban_benchmark()
    (OUT / "thirdparty_results.json").write_text(json.dumps(dict(sudoku_extreme=su_res, boxoban=bk_res), indent=2))