# Final correctness report

**Final candidate:** solution commit `0c58998478973e2915ad6fc4624aeae52744b933`. Later commits
(`6c8d388`…) change only `tasks/` documents; solution/harness/tests/task/tools are byte-identical.

## Defects

None reproducible. Historical `test_unpad[(5,5,5,5),9,bnb]` failure: LoRA inputs bitwise equal,
pad gradients zero, upstream fp16 cancellation; now validated against a full fp64 Qwen2+LoRA oracle
with the original tolerances (S-034). See `ERROR_FIX_LOG.md`.

## Tests executed (Kaggle Tesla T4, torch 2.11, Triton 3.6.0)

| Run | Commit | Result |
|---|---|---|
| `pytest solution -q -rs` | b1dcb59 (solution = 0c58998) | 323 passed, 0 failed, 0 skipped |
| `SOUP_APPLY_SOLUTION=1 pytest tests -q` | b1dcb59 | 79 passed |
| `run_task.py --repeats 2` + `check_rules.py` + `score.py` | 0c58998 | ALL CHECKS PASSED; speedup 1.181, memory 3.810, eval +0.000%, own kernels 15.5% |

Logs: `runs_kaggle/20261030T185618838149Z_unit_b1dcb59_correctness/out/`,
`runs_kaggle/20261030T173912044822Z_full2_0c58998/out/` (local, not in Git).
Locally (macOS, no torch/CUDA) only syntax/CPU checks run; every CUDA test ran on T4.

## Acceptance gate

- [x] No known reproducible correctness defect.
- [x] All solution tests pass on T4, none skipped.
- [x] 79 Soup tests pass.
- [x] Forward/backward, NF4/bnb, LoRA gradients validated (kernel/integration tests on T4; reports).
- [x] MLP offload: no demonstrated stream/lifetime defect (S-030 review, S-033 stress tests, T4).
- [x] `apply()` works and is idempotent (integration tests; reload checks in Codex-2 report).
- [x] No prohibited libraries; no `torch.compile`.
- [x] `harness/`, `tests/`, `task/` unchanged since the starter kit.
- [x] All ten official checks pass on the final candidate (full ×2).
- [x] Final commit documented.

## Remaining (not blockers)

- P2-021-3: CE oracle checks matching chunk GEMMs, not a full-head reference at every shape.
- P1-023-1: fp16 input before LoRA dropout differs by ≤1 fp16 ulp from the base (checks 5–7 pass).
- Not run: real Qwen/PEFT/bnb reload script and compute-sanitizer probe (commands in the Codex reports).
- Submission links: team GitHub repository and public Kaggle notebook are still missing.
