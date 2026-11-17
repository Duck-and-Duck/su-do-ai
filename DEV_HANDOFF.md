# SUDOKU-AGI / SOKOBAN-AGI / ARC-AGI — Developer Handoff Paper

**Repo:** `https://github.com/Duck-and-Duck/su-do-ai`
**Local Hardware Verified:** Windows 11, Python 3.12.10, PyTorch 2.6.0+cu124, NVIDIA RTX 3050 6GB Laptop GPU (6144 MiB VRAM).

---

## 1. File Map & Current State

| File | Version | Role & Status |
|---|---|---|
| `core.py` | `v8.1.0` | **Domain-agnostic AGI backbone.** Contains `SelfAttnSet` (QK-Norm + `>32768` CUDA gridDim chunking), `HyperGraphBlock` (zero-atomic `inv_idx` partition gather + `index_add_` fallback), `RelationalReasoner` (`FROG [mean, amax]` global context + `3`-head ensemble `imagine_step` + action-centered `coll_head` + `dead_head` + relative-noise PTRM), `MuonAdamW`, `EMA`, `wilson`/`ece`/`pass_at_k`. |
| `sudoku_ai.py` | `v8.1.0` | **Raw-Obs Sudoku + World Model + Level Engine.** `755,478` params. Sees ONLY 3 raw channels (`[digit_onehot, empty, given]`, zero `legal`/`has`/`lc` rule crutches). Includes `13/13 + P14` proofs, 2-step vectorized world model, `FocusDiff`, `PTRM`, `WM-Tree` (`0` Python rules inside imagination), `simulate_solve` (neuro-symbolic DFS), `inspect_imagination`, and `LevelPuzzleEngine (L1-L5)`. |
| `sokoban_ai.py` | `v1.2.1` | **D4-Equivariant Sokoban.** `531,940` params. `13/13 + P14` proofs, `bfs_solve` + `astar_solve` + `naive_bfs_solve`, reverse-pull generator (`0` boxes on goals at `t=0`), Boxoban `10x10` loader, GIF + PNG storyboard renderer. *Pending refactor:* not yet importing `core.py v8.1.0` or training `imagine_step` world model. |
| `arc_ai.py` | `v5.0.2` | **ARC-AGI-1 Object-Aware Per-Task MDL.** `D4SymConv2d` + tied `row/col` `SelfAttnSet`, 9 real object/neighbor features (4/8-nb counts, BFS connected components), shape router (`SAME_HW`, `INT_SCALE`, `CROP`, `CONST_GLOBAL`), KL variational bottleneck, `2x D4` + `demos_fit` filter. Speed measured at `6.18s/task`; `v5.0.2` fixes `CONST_GLOBAL` variable output-shape stacking (ready to run `n=25`). |
| `benchmark_thirdparty.py` | `v8.1.0` | Zero-shot, zero-leakage evaluator against `sapientinc/sudoku-extreme` (`n=200`). |

---

## 2. Verified Benchmarks (All Numbers Measured, Zero Benchmark Training, Zero Leakage)

### A. Sudoku (`runs/sudoku_v8_1/weights.pt`, `755,478` params, `RAW-OBS`, `170` iters / `12 min`)
* **Proofs:** `13/13` PASS + `P14` (`0/800` internal leakage, `0/200` `sudoku-extreme` leakage). Exact $G$-equivariance `P5`: `enc=5.36e-07`, `dyn=5.66e-07`, `val=1.34e-07`.
* **Rules Learned From Raw Pixels:** `rule_acc = 0.990`, `wm_cos = 0.947`.
* **Imagination Inspector (`78`-clue unique puzzle):** Conflicts `1..8`: `Coll=0.977, V=0.405, Dead=0.602`; True digit `9`: **`Coll=0.015, V=0.670, Dead=0.370, Policy=0.999`**.
* **Internal Unique Held-Out (`n=24/tier`):**
  * `T1 easy`: `1.000` (`FocusDiff` / `PTRM16` / `WM-Tree` / `pass@2` / `Sys1+2`)
  * `T2 medium`: `0.958` `FocusDiff`, **`1.000`** `PTRM16` / `WM-Tree` / `pass@2` / `Sys1+2`
  * `T3 hard`: `0.875` `FocusDiff` / `WM-Tree`, **`0.958`** `PTRM16`, `1.000` `Sys1+2`
  * `T4 expert`: `0.833` `FocusDiff`, **`0.917`** `PTRM16` / `WM-Tree`, `1.000` `Sys1+2`
  * `T5 extreme`: `0.833` `FocusDiff`, **`1.000`** `PTRM16`, `0.667` `WM-Tree`, `0.958` `Sys1+2`
* **Multi-Solution Level Engine (`L1-L5`, `n=12/level`, any valid completion verified):** `PTRM(16)` scored **`1.000` on ALL 5 levels** (`L1 70-79c` down to `L5 10-17c`); `WM-Tree`: `1.000 / 1.000 / 1.000 / 1.000 / 0.750`.
* **Zero-Shot Official `sapientinc/sudoku-extreme` (`n=200`, `165` 0-singles, `min=17` clues):**
  * `FocusDiff(1)`: `33.00%` (`19.39%` 0-singles)
  * `PTRM K=32` (0 tree search): **`58.50%`** (`49.70%` 0-singles)
  * `WM-Tree` (`n=50`, 0 rules in imagination): **`48.00%`** (`35.00%` 0-singles)
  * `Neuro-Symbolic Sys2` (`n=50, 16n`): `64.00%` (`55.00%` 0-singles) *(vs. TRM-MLP 87.4% at 5M params, 1000x aug, 50M views)*

### B. Sokoban (`runs/boxoban_v1/weights.pt`, `531,940` params, `701` Boxoban train maps, `18 min`)
* **Proofs:** `13/13` + `P14` (`0` D4-orbit leakage). `P5` D4 equivariance `6.11e-07`.
* **Official DeepMind Boxoban (`10x10`, 4 boxes):**
  * `unfiltered/test` (`n=100`): `Sys1 greedy = 53.0%`, `pass@2 = 67.3%`, `Sys1+2 (48n) = 80.0%` (`14.3` avg nodes).
  * `hard` (`n=50`): `Sys1 = 12.0%`, `Sys1+2 (48n) = 30.0%` (`38.6` nodes).
* **Zero-Shot `8x8 -> 10x10` Transfer (`runs/soko_v2`):** `1.000 / 1.000 / 1.000 / 0.917` on synthetic `T1-T4` `10x10`.

### C. ARC-AGI-1 (`fchollet/ARC-AGI`) — Current Honest Weakness
* `arc_v3` (`178K` params, CrossAttn over 15k tokens): `0.00%` untrained, `0.25%` (`1/400`) after 10m pretrain, `0.00%` (`0/60`, `40m`) TTT.
* `arc_v4` (`38K` params, per-task MDL, `45m`): **`2/100` (`2.00%`, `demos_fit=17/100`, solved `['0607ce86', '3b4c2228']`)**.
* `arc_v5` (`v5.0.2`): `6.18s/task` measured; pending first full `n=25` run after `CONST_GLOBAL` fix.

---

## 3. Brutal Honesty: Remaining Crutches & Next Engineering Priorities

1. **Neuro-Symbolic `Sys2` (`simulate_solve`)** in Sudoku/Sokoban is still classical DFS/Best-First + exact simulator heuristic—only `WM-Tree` (`imagine_step`) is pure latent world-model planning. Push `WM-Tree` deeper (`depth=3..4`, `beam=4`, trained multi-step rollouts) and refactor `sokoban_ai.py` onto `core.py v8.1.0` so Sokoban gets `FROG [mean,max]` + `coll_head` + `imagine_step`!
2. **Why `sudoku-extreme` `Pure Sys1` (`33%` / `58.5%` PTRM) trails TRM (`87.4%`):** Only `170` gradient steps (`~8,160` views) vs. TRM's `50,000,000` views, and our generator only averages `23.5` clues rather than `17-20`-clue forcing-chain puzzles.
3. **Why ARC-AGI-1 is hard for pure per-cell NNs:** Every task invents a new rule from 2-4 demos. Path forward: (a) verify `arc_ai.py v5.0.2` (`n=25`, `~2.6 min`), (b) scale up per-task `CompressARC` ELBO steps/capacity OR combine `HyperGraph` object perception with a small verifiable object-DSL program search (`Induce -> Verify 100% on demos -> Apply`), (c) eventually unify Sudoku + Sokoban + ARC into ONE shared checkpoint interface.