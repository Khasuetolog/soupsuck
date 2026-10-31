# Submission

## Links

- Repository, final commit: `https://github.com/Khasuetolog/soupsuck/commit/f1cb5ac32ef7cbd1caf8455c833b67bd43992ed8`
- Public Kaggle notebook, saved version: https://www.kaggle.com/code/khasansueta/notebookf3ae4d2bef
- Notebook `COMMIT`: `f1cb5ac32ef7cbd1caf8455c833b67bd43992ed8`; the same saved-version link is in [README.md](README.md).

## Team

| Name | Role | GitHub | Kaggle | Contact |
| --- | --- | --- | --- | --- |
| Ахметжан Хасен | Engineer | — | — | — |
| Anuarbek Islam | Software Developer | — | — | — |
| Omirali Amirkhan | Engineer | — | — | — |
| Zholdykaraeva Amina | Software Developer | — | — | — |

## Final result

Two base and two solution repetitions in one Kaggle T4 development `full2` session, solution commit
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

Base wall time varies between Kaggle sessions (77.7–91.4 s in selected recent
sessions with identical base code), so compare the in-session ratio. Paired
same-session A/B results for whole-run experiments, separately labelled
microbenchmarks, estimates and historical comparisons, and a partial
instrumented time breakdown are in
[README.md](README.md).

## Reproduce

To reproduce the two-repeat `full2` evidence after installing pinned dependencies
and preparing data, from the repository root:

```bash
git checkout f1cb5ac32ef7cbd1caf8455c833b67bd43992ed8
python -m pytest solution -q
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
python harness/run_task.py --repeats 2
python harness/check_rules.py
python harness/score.py runs/result.json
```

Use the [Kaggle template](templates/kaggle_notebook.ipynb), `COMMIT=f1cb5ac32ef7cbd1caf8455c833b67bd43992ed8`,
GPU T4 and one GPU through the harness. Run defaults without SOUP overrides.

## Limitations

The no-logits head supports Qwen2 labelled training with a frozen bias-free head
on T4/fp16; unsupported configurations use the original head/trainer paths. We
do not replace attention or the 4-bit GEMM. Lossless embedding storage applies
only when every element round-trips exactly through fp16. Early fp16 norm output
preserves dropout masks/RNG but can change values by fp16 rounding; public checks
5–7 pass on two repeats, which does not prove equality on hidden rows or seeds.
The partial instrumented probe observed 23.97 s before the first training step
(`ok=false`, missing `import.bitsandbytes` coverage); the final `full2` wall time
minus training loop averaged 25.8 s, including startup and shutdown. See the
measurement details in [README.md](README.md) and [tasks/REVIEW.md](tasks/REVIEW.md).
