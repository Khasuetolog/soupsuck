# Speed Lab plan

## Measured starting point (Kaggle T4, logs in runs_kaggle/, not in Git)

Best known good: solution commit `0c58998` (full ×2, jury order): base 87.39/80.31 s, solution
70.99/71.04 s, speedup 1.181, memory 3.810, 10/10 checks, pytest solution 323/0/0.

Where the solution's 71 s go (solution_1 metrics, 0c58998):
- training loop 46.6 s = first step **9.88 s** + 63 steps × median 0.509 s;
- outside the loop ≈ 24.4 s (Soup startup, model load, save). Base first step 1.20 s.
GPU time per step (profile, e0c50a2 + offload): GEMM 35.9%, offload H2D 14.3% + D2H 14.0%
(on side streams, mostly overlapped), NF4 decode 7.5%, attention 5.4%, our SwiGLU 3.1%.

## Feasibility of the 35 s target

The ≈24 s outside the training loop is mostly model loading and Soup setup that the base pays too;
35 s total would leave ≈11 s for 64 steps (≈0.17 s/step) while the GEMMs alone take ≈0.16 s/step.
35 s is therefore not reachable without reducing real work. Realistic milestones: <60 s, then <55 s.

## Experiment rules

- Every candidate lives behind an opt-in `SOUP_EXP_*` env flag (default = best known good) until it
  wins a paired measurement and passes all checks; then it becomes default in a separate commit.
- Kaggle machines differ by up to ~10% between sessions: decisions only from paired runs in one
  session (EXP-003 builds this), full ×2 for finalists.
- Records: `experiments/registry.jsonl`, `experiments/results.csv`, `tasks/SPEED_LAB_RESULTS.md`.

## First wave

| ID | Owner | Family | Hypothesis |
|---|---|---|---|
| EXP-001 | Codex-1 | G/I | First step costs 8.7 s more than the base (Triton JIT + init). Compile our kernels for the known shapes in background threads during Soup's model load (CPU compile overlaps I/O-bound loading; still inside the measured run, no cache reuse). Target −5..8 s wall. |
| EXP-002 | Codex-2 | F | NF4 decode + dense-weight GEMM: fuse dequantization into a Triton GEMM for the M>1536 forward fallback and backward dX (no dense W round trip), compare against decode+cuBLAS per shape. |
| EXP-003 | Codex-3 | infra | Paired A/B runner: one Kaggle session runs N variants interleaved (env flags) with the base, records wall/loop/first-step/peak, writes registry/results; statistics to tell gains from noise. |
