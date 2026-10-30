# Error audit — correctness phase (no performance work)

Date: 2026-10-30. Coordinator: Claude. Repository: this checkout (the shared copy
`team-share/decentrathon-team/` is a snapshot of commit `0c58998` and is not edited).

## Repository state

- HEAD: `6c8d388` (status-only commit on top of solution commit `0c58998`).
- Modified tracked files: none. Untracked: `.DS_Store`, `team-share/` (packaging output, not source).
- `harness/`, `tests/`, `task/` unchanged since the starter kit import (`02c866f`).
- Banned-library scan of `solution/`: only docstring mentions and the `use_liger_kernel`
  refusal guard in `solution/patch.py:139`; no imports, no `torch.compile`.

## Last measured evidence (Kaggle T4, `runs_kaggle/`, not in Git)

| Run | Commit | Result |
|---|---|---|
| full ×2 (jury order) | 0c58998 | 10/10 checks PASS; speedup 1.181, memory 3.810, own kernels 15.5% |
| pytest solution (same run) | 0c58998 | 323 passed, 0 failed, 0 skipped on T4 |
| Soup tests with patches | 0c58998 | 79 passed |

## Known failures

None reproduced on the current snapshot. History:
- `test_unpad[(5,5,5,5),9,bnb]` failed on 8bc5dc1/42423bf; S-034 showed bitwise-equal LoRA
  inputs, zero pad gradients and fp16 cancellation in upstream dY; the test now checks against a
  full fp64 Qwen2+LoRA oracle with unchanged tolerances and passes on T4 (fe00959, 0c58998).
  Must be re-confirmed by Codex-2 (task requirement), not assumed.

## Open risks to verify (not demonstrated defects)

1. P1-023-1: fp16 cast before LoRA dropout gives ±1 fp16 ulp input differences (checks 6/7 pass).
2. Async MLP offload (default since 0c58998): stream/event/lifetime correctness verified by review
   (S-030/S-033) and full runs, but stress tests rely on T4 execution.
3. Review P2 items still open: P2-021-3 (CE oracle scope), P2-028-2 coverage, P2-030-x.
4. Tests that cannot execute locally: all CUDA tests (macOS x86, Python 3.14, no torch wheel);
   they run only on Kaggle T4.

## Plan

- Codex-1: kernels + patch/autograd/__init__ correctness audit → `tasks/CODEX1_CORRECTNESS_REPORT.md`.
- Codex-2: mlp/unpad/tests, offload lifetime, unpad bnb case → `tasks/CODEX2_CORRECTNESS_REPORT.md`.
- Coordinator: review, T4 unit + full2 runs, `ERROR_FIX_LOG.md`, `FINAL_CORRECTNESS_REPORT.md`.
- Performance ideas go to `tasks/OPTIMIZATION_BACKLOG.md` only.
