# SUDOKU-AGI & SOKOBAN-AGI

Domain-agnostic equivariant hypergraph reasoner (`RelationalReasoner` + `HyperGraphBlock` + `MuonAdamW` + `RLOO` + `System 2` simulation) with executable mathematical proofs (`P1-P14`).

- `sudoku_ai.py` (v5.0.2): 533K params, 14/14 proofs PASS, 0.9774 pass@2 gate PASS, 99.1% (108/109) on non-singles T3-T5 (100% on T5 extreme).
- `sokoban_ai.py` (v1.0.0): D4-equivariant spatial planning + deadlock avoidance + auto-rendered gameplay videos.