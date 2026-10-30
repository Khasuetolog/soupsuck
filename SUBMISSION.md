# Submission: <TEAM_NAME>

## Links

- Repository, final commit: `<REPOSITORY_URL>/commit/<FINAL_COMMIT>`
- Public Kaggle notebook, saved version: `<KAGGLE_LINK>`
- Notebook `COMMIT`: `<FINAL_COMMIT>`; use the same saved-version link in [README.md](README.md).
- Demo video, up to 3 minutes: `<VIDEO_LINK>`
- Presentation: `<PRESENTATION_LINK>`

Links and team details are placeholders. The result below belongs to development
commit `de836e0264d24fbd2e184eb853e2bb81bb0f0c11`; it is not yet the final
commit/notebook result.

## Team (up to 4 people)

| Name | Role | GitHub | Kaggle | Contact (captain) |
| --- | --- | --- | --- | --- |
| `<NAME>` | `<ROLE>` | `<GITHUB>` | `<KAGGLE>` | `<CONTACT>` |

## Our kernels

The table lists our actual `@triton.jit` names. Registered paths in the documented
committed snapshot `8f8ebbc` are the fused head/TRL bridge, Qwen2 norms/decoder,
PEFT LoRA epilogue and lazy frozen MLP. NF4 has an installer but is not registered
by `solution.apply()` in that snapshot. The supplied `example_*` SwiGLU kernels
are not our contribution and were not registered for the measured MLP.

Each measured share is `100 * own_kernel_ms[name] / (1000 * timed_loop_s)`;
`timed_loop_s = 39.556`, covering the last 62 of 64 training steps.
Source: `runs_kaggle/20261030T105526661826Z_full_de836e0/out/runs/solution_1/metrics.json`.
Shares refer to training time measured by the harness CUDA events, not whole
command time or profiler GPU-time percentages.

| Kernel (`@triton.jit` name) | File | What it replaces | CUDA-event ms | Share of timed training |
| --- | --- | --- | ---: | ---: |
| `vocabulary_statistics_kernel` | [solution/kernels/cross_entropy.py](solution/kernels/cross_entropy.py) | Chunk normalizer, target logit and argmax | 273.461 | 0.6913% |
| `vocabulary_outputs_kernel` | [solution/kernels/cross_entropy.py](solution/kernels/cross_entropy.py) | Per-token CE, entropy and LSE | 0.637 | 0.0016% |
| `vocabulary_derivative_kernel` | [solution/kernels/cross_entropy.py](solution/kernels/cross_entropy.py) | Softmax/CE derivative for dX | 328.581 | 0.8307% |
| `rmsnorm_forward_kernel` | [solution/kernels/rmsnorm.py](solution/kernels/rmsnorm.py) | Qwen2 RMSNorm forward | 144.228 | 0.3646% |
| `rmsnorm_input_gradient_kernel` | [solution/kernels/rmsnorm.py](solution/kernels/rmsnorm.py) | RMSNorm input backward | 194.386 | 0.4914% |
| `rmsnorm_weight_partials_kernel` | [solution/kernels/rmsnorm.py](solution/kernels/rmsnorm.py) | Optional weight-gradient partials; frozen in this task | Not measured | Not launched / not counted |
| `rmsnorm_weight_finalize_kernel` | [solution/kernels/rmsnorm.py](solution/kernels/rmsnorm.py) | Optional weight-gradient reduction; frozen in this task | Not measured | Not launched / not counted |
| `add_rmsnorm_forward_kernel` | [solution/kernels/add_rmsnorm.py](solution/kernels/add_rmsnorm.py) | Attention residual add + post-attention norm | 213.710 | 0.5403% |
| `add_rmsnorm_backward_kernel` | [solution/kernels/add_rmsnorm.py](solution/kernels/add_rmsnorm.py) | Combined residual/norm backward | 254.252 | 0.6428% |
| `lora_scale_add_kernel` | [solution/kernels/lora_epilogue.py](solution/kernels/lora_epilogue.py) | PEFT scale + base-result add | 212.380 | 0.5369% |
| `lora_scale_gradient_kernel` | [solution/kernels/lora_epilogue.py](solution/kernels/lora_epilogue.py) | LoRA scale/add backward | 108.677 | 0.2747% |
| `swiglu_tile_engine` | [solution/kernels/swiglu_fused.py](solution/kernels/swiglu_fused.py) | Tiled SwiGLU forward and both activation derivatives | 1659.851 | 4.1962% |
| `nf4_decode_kernel` | [solution/kernels/nf4.py](solution/kernels/nf4.py) | NF4 dequantization; available after measured commit, not registered | Not measured | Not launched / not counted |
| **Counted total** | | | **3390.163** | **8.5705%** |


Check 4 reports **8.6%** after rounding, above its 5% requirement. Optional norm
weight-gradient kernels were not launched because norm parameters are frozen;
this is an informational note in check 3, not a failed check. NF4 has no measured
contribution to these results. GEMMs remain library operations and their time
is not counted as our own kernels.

Ideas taken from other kernels, with links (our implementations are written here):

- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867): running max and rescaled sums.
- [Cut Your Losses](https://arxiv.org/abs/2411.09009): avoid retaining full vocabulary logits.
- [Liger](https://github.com/linkedin/Liger-Kernel) / [Apex normalization](https://github.com/NVIDIA/apex/tree/master/apex/normalization): fusion patterns and RMSNorm reductions.
- [PEFT v0.20.0](https://github.com/huggingface/peft/tree/v0.20.0): original LoRA control flow and scale/add rounding.
- [bitsandbytes 0.50.2](https://github.com/bitsandbytes-foundation/bitsandbytes/tree/0.50.2): NF4 format and nested scales.
- Standard RMSNorm and SiLU/SwiGLU derivatives. No ready-made kernel library is imported.

## Results on Kaggle T4

**First passing development comparison: one repeat, seed 1, measured commit
de836e0.** Evidence root:
`runs_kaggle/20261030T105526661826Z_full_de836e0/out/`.
`runs/result.json:summary` supplies the table and ratios; `runs/checks.json`
and `checks.log` supply checks; `score.log` corroborates the rounded ratios;
`source-manifest.json` supplies commit/archive hashes. These development files
are locally retained and excluded from Git; the final public notebook must
contain final evidence.

| | Base Soup | Ours | Ratio / result |
| --- | ---: | ---: | --- |
| Time of `soup train`, s | 77.47 | 70.05 | 1.1059× speedup |
| Peak GPU memory, MiB | 14,345 | 5,843 | 2.4551× base/ours |
| Eval loss, public rows | 2.04964 | 2.04962 | −0.001% vs base |
| Training loop, s | 53.42 | 47.59 | From each run's metrics.json |
| Own-kernel share | — | 8.6% | Check 4 PASS |

- [x] All 10 checks passed **on the development result above**, including 79 Soup regression tests.
- [x] README rewritten: implementation, commands, measured results and limitations.
- [ ] Final commit `<FINAL_COMMIT>` validated in the jury order: base, solution, solution, base.
- [ ] Final public saved notebook `<KAGGLE_LINK>` and all links/team details filled.
- [ ] Separate solution test suite fully passing on final commit. The development log has 171 passed and 1 failed (`pytest_solution.log`), despite all ten harness checks passing.

### Final two-repeat result — pending

| Field | Final value |
| --- | --- |
| Commit / saved notebook | `<FINAL_COMMIT>` / `<KAGGLE_LINK>` |
| Base / solution time, s | `<FINAL_BASE_TIME>` / `<FINAL_SOLUTION_TIME>` |
| Base / solution peak, MiB | `<FINAL_BASE_MEMORY>` / `<FINAL_SOLUTION_MEMORY>` |
| Base / solution eval loss | `<FINAL_BASE_LOSS>` / `<FINAL_SOLUTION_LOSS>` |
| Minimum own-kernel share across solution runs | `<FINAL_KERNEL_SHARE>` |
| All ten checks / separate solution tests | `<FINAL_CHECKS>` / `<FINAL_SOLUTION_TESTS>` |

To reproduce the final comparison after installing pinned dependencies and
preparing data, from the repository root:

```bash
git checkout <FINAL_COMMIT>
python -m pytest solution -q
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
python harness/run_task.py --repeats 2
python harness/check_rules.py
python harness/score.py runs/result.json
```

Use the [Kaggle template](templates/kaggle_notebook.ipynb), `COMMIT=<FINAL_COMMIT>`,
GPU T4 x2 and one GPU through the harness, as described in [README.md](README.md).
Run defaults without experimental SOUP overrides. Development delivery can use
`python3 tools/kaggle/run.py run --mode full2` after uploading committed HEAD.

## Presentation covers

Operations changed, kernels and mathematics, dtype/correctness contract, time,
GPU peak and held-out quality against base, the distinction between development
and final comparisons, and remaining limits. Slides: `<PRESENTATION_LINK>`;
video: `<VIDEO_LINK>`.

## Limitations

The no-logits head supports Qwen2 labelled training with a frozen bias-free head
on T4/fp16; unsupported configurations use the original head/trainer paths.
Layer patches have separate guards. We do not replace attention or 4-bit GEMM.
NF4 decoding is implemented but unregistered in the documented snapshot.
Default MLP mode stores fp16 gate/up; optional recomputation trades more
projection work for lower saved activation storage and has no claimed full-run
result here. Chunked CE uses library GEMMs and Python iteration.

[Review findings](tasks/REVIEW.md) in measured de836e0: the kernel-share blocker
is resolved for this pair (8.6%); the all-ignored/nonfinite-upstream gradient
edge and fp32 MLP without AMP fallback were open in that measured snapshot.
The latter caused the one separate solution-test failure. Current 8f8ebbc
changes CE to native fp16 logit GEMM and removes ignored rows from backward
before gain computation, addressing the NaN dX path in code with new regression
tests; its GPU validation and full comparison are pending. No fix or newer
precision experiment inherits the old checks/results automatically. Final
measurements must cover the final code and two repetitions; Kaggle host, seed
and hidden-row variation can affect ratios and quality.
