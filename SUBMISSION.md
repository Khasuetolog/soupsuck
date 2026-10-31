# Submission: <TEAM_NAME>

## Links

- Repository, final commit: `https://github.com/Khasuetolog/soupsuck/commit/<FINAL_COMMIT>`
- Public Kaggle notebook, saved version: `<KAGGLE_LINK>`
- Notebook `COMMIT`: `<FINAL_COMMIT>`; the same saved-version link is in [README.md](README.md).

## Team

| Name | Role | GitHub | Kaggle | Contact |
| --- | --- | --- | --- | --- |
| `<NAME>` | `<ROLE>` | `<GITHUB>` | `<KAGGLE>` | `<CONTACT>` |

## Final result

Two base and two solution repetitions in one Kaggle T4 session, solution commit
`96a890c`, `harness/check_rules.py`: **ALL CHECKS PASSED (10/10)**.

| Measurement | Base | Ours | Ratio |
| --- | ---: | ---: | --- |
| `soup train` wall time, mean of 2 | 87.61 s | 65.49 s | **1.338× speedup** |
| Peak GPU memory (nvidia-smi) | 14,345 MiB | 3,769 MiB | **3.806× base/ours** |
| Public eval loss | 2.04966 | 2.04964 | −0.001 % |
| Own Triton kernel share, lowest run | — | 13.3 % | check 4 PASS |
| Loss curve (check 6) | — | first step 0.02 %, mean 0.01 % | PASS |

Solution suite on T4: 518 passed, 1 skipped, 0 failed. Soup regression tests
with `apply()`: 79/79. Evidence (local, not in Git):
`runs_kaggle/20261031T133425573730Z_full2_96a890c_promote010/out/`.

Base wall time varies between Kaggle sessions (77.7–91.4 s observed for the
same code), so compare the in-session ratio. Paired same-session A/B
measurements for every optimization, and the time breakdown of a run, are in
[README.md](README.md).

## Reproduce

After installing pinned dependencies and preparing data, from the repository root:

```bash
git checkout <FINAL_COMMIT>
python -m pytest solution -q
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
python harness/run_task.py --repeats 2
python harness/check_rules.py
python harness/score.py runs/result.json
```

Use the [Kaggle template](templates/kaggle_notebook.ipynb), `COMMIT=<FINAL_COMMIT>`,
GPU T4 and one GPU through the harness. Run defaults without SOUP overrides.

## Limitations

The no-logits head supports Qwen2 labelled training with a frozen bias-free head
on T4/fp16; unsupported configurations use the original head/trainer paths. We
do not replace attention or the 4-bit GEMM. Lossless embedding storage applies
only when every element round-trips exactly through fp16. Early fp16 norm output
preserves dropout masks/RNG but can change values by fp16 rounding; public checks
5–7 pass on two repeats, which does not prove equality on hidden rows or seeds.
About 24 s of each run is Python/model setup outside the training loop that the
solution cannot change without editing the harness. See [tasks/REVIEW.md](tasks/REVIEW.md).
