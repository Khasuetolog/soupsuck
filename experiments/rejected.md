# Rejected experiments

(none yet)

## EXP-002 — Triton NF4-dequant GEMM (Codex-2) — REJECTED
T4 bench 73a1372: correctness 96/96 configs + 24 autograd scenarios PASS (max rel L2 3.7e-4), but the
best candidate per shape is 4.12–66.01× slower than bnb decode + cuBLAS. Allowlist empty, default
unchanged. Report: experiments/reports/EXP-002.md.
