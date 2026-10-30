# Submission: <TEAM_NAME>

## Links

- Repository, final commit: `<REPOSITORY_URL>/commit/<FINAL_COMMIT>`
- Public Kaggle notebook, saved version: `<KAGGLE_LINK>`
- Notebook `COMMIT`: `<FINAL_COMMIT>`; use the same saved-version link in [README.md](README.md).
- Demo video, up to 3 minutes: `<VIDEO_LINK>`
- Presentation: `<PRESENTATION_LINK>`

Links and team details are placeholders. The result below belongs to development
commit `ce1ff09`; it is not yet the final
commit/notebook result.

## Team (up to 4 people)

| Name | Role | GitHub | Kaggle | Contact (captain) |
| --- | --- | --- | --- | --- |
| `<NAME>` | `<ROLE>` | `<GITHUB>` | `<KAGGLE>` | `<CONTACT>` |

## Our kernels

The table uses actual `@triton.jit` names. Current registration is committed
HEAD `585de28`: fused head/TRL bridge, Qwen2 norms/decoder, PEFT LoRA epilogue,
lazy frozen MLP, bitsandbytes NF4 decode, RoPE, and lossless tied-embedding setup.
RoPE and embedding registration postdate measured ce1ff09; they have no measured
credit in its result. Embedding storage introduces no new Triton kernel.
The supplied `example_*` kernels are not registered or counted.

Each share is `100 * own_kernel_ms[name] / (1000 * timed_loop_s)`;
`timed_loop_s = 43.035`, covering the last 62 of 64 training steps.
Source: `runs_kaggle/20261030T112204781815Z_full_ce1ff09/out/runs/solution_1/metrics.json`.
These are shares of timed training, not whole-command or profiler GPU time.

| Kernel (`@triton.jit` name) | File | What it replaces | CUDA-event ms | Share of timed training |
| --- | --- | --- | ---: | ---: |
| `vocabulary_statistics_kernel` | [solution/kernels/cross_entropy.py](solution/kernels/cross_entropy.py) | Chunk normalizer, target logit and argmax | 262.717 | 0.6105% |
| `vocabulary_outputs_kernel` | [solution/kernels/cross_entropy.py](solution/kernels/cross_entropy.py) | Per-token CE, entropy and LSE | 0.725 | 0.0017% |
| `vocabulary_derivative_kernel` | [solution/kernels/cross_entropy.py](solution/kernels/cross_entropy.py) | Softmax/CE derivative for compacted dX | 112.667 | 0.2618% |
| `rmsnorm_forward_kernel` | [solution/kernels/rmsnorm.py](solution/kernels/rmsnorm.py) | Qwen2 RMSNorm, optional half output | 119.613 | 0.2779% |
| `rmsnorm_input_gradient_kernel` | [solution/kernels/rmsnorm.py](solution/kernels/rmsnorm.py) | RMSNorm input backward | 165.313 | 0.3841% |
| `rmsnorm_weight_partials_kernel` | [solution/kernels/rmsnorm.py](solution/kernels/rmsnorm.py) | Optional norm-weight partials; weights frozen here | Not measured | Not counted |
| `rmsnorm_weight_finalize_kernel` | [solution/kernels/rmsnorm.py](solution/kernels/rmsnorm.py) | Optional norm-weight reduction; weights frozen here | Not measured | Not counted |
| `add_rmsnorm_forward_kernel` | [solution/kernels/add_rmsnorm.py](solution/kernels/add_rmsnorm.py) | Attention residual add + post-attention norm | 222.172 | 0.5163% |
| `add_rmsnorm_backward_kernel` | [solution/kernels/add_rmsnorm.py](solution/kernels/add_rmsnorm.py) | Combined residual/norm backward | 254.163 | 0.5906% |
| `lora_scale_add_kernel` | [solution/kernels/lora_epilogue.py](solution/kernels/lora_epilogue.py) | PEFT scale + base-result add | 148.014 | 0.3439% |
| `lora_scale_gradient_kernel` | [solution/kernels/lora_epilogue.py](solution/kernels/lora_epilogue.py) | LoRA scale/add backward | 84.083 | 0.1954% |
| `swiglu_tile_engine` | [solution/kernels/swiglu_fused.py](solution/kernels/swiglu_fused.py) | Tiled SwiGLU forward/backward variants | 1709.047 | 3.9713% |
| `nf4_decode_kernel` | [solution/kernels/nf4.py](solution/kernels/nf4.py) | Nested/flat NF4 decode; registered and measured | 2148.202 | 4.9918% |
| `rope_pair_forward_kernel` | [solution/kernels/rope.py](solution/kernels/rope.py) | Combined q/k rotary forward; added after this run | Not measured | Not counted |
| `rope_pair_backward_kernel` | [solution/kernels/rope.py](solution/kernels/rope.py) | Combined q/k rotary backward; added after this run | Not measured | Not counted |
| **Counted total** | | | **5226.716** | **12.1453%** |

Check 4 rounds to **12.1%**, above its 5% requirement. Optional norm-weight
kernels were absent because norms are frozen. RoPE's absence is because it was
added later. GEMMs and other library kernels receive no own-kernel credit.
Registration and guards are in [solution/patch.py](solution/patch.py); detailed
math, dtype boundaries and fallback behavior are in [README.md](README.md).

Ideas taken from other kernels, with links (our implementations are written here):

- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867): running max and rescaled sums.
- [Cut Your Losses](https://arxiv.org/abs/2411.09009): avoid retaining full vocabulary logits.
- [Liger](https://github.com/linkedin/Liger-Kernel) / [Apex normalization](https://github.com/NVIDIA/apex/tree/master/apex/normalization): fusion patterns and RMSNorm reductions.
- [PEFT v0.20.0](https://github.com/huggingface/peft/tree/v0.20.0): original LoRA control flow and scale/add rounding.
- [bitsandbytes 0.50.2](https://github.com/bitsandbytes-foundation/bitsandbytes/tree/0.50.2): NF4 format and nested scales.
- Standard RMSNorm and SiLU/SwiGLU derivatives. No ready-made kernel library is imported.

## Results on Kaggle T4

**Latest passing development comparison: one repeat, seed 1, ce1ff09.**
Evidence root: `runs_kaggle/20261030T112204781815Z_full_ce1ff09/out/`.
`runs/result.json:summary` supplies ratios; `runs/checks.json` supplies checks;
`source-manifest.json` supplies archive hashes. Development artifacts are local
and excluded from Git; the public final notebook must contain final evidence.

| | Base Soup | Ours | Ratio / result |
| --- | ---: | ---: | --- |
| Time of `soup train`, s | 82.25 | 73.54 | 1.1184× speedup |
| Peak GPU memory, MiB | 14,345 | 5,863 | 2.4467× base/ours |
| Eval loss, public rows | 2.04960 | 2.04969 | +0.004% vs base |
| Training loop, s | 57.51 | 51.52 | From each run's metrics.json |
| Own-kernel share | — | 12.1% | Check 4 PASS |

Historical de836e0 pair: 77.47 / 70.05 s, 14,345 / 5,843 MiB, eval loss
2.04964 / 2.04962, speedup 1.1059, memory ratio 2.4551, own share 8.6%,
10/10 checks. Evidence:
`runs_kaggle/20261030T105526661826Z_full_de836e0/out/runs/result.json`.

Default MLP **store** remains selected. Separate ce1ff09 recompute experiment
(`SOUP_MLP_RECOMPUTE=1`) passed 10/10 with 93.95 s / 4,123 MiB / 2.04968;
its own base was 97.75 s / 14,345 MiB / 2.04964. Paired ratios were 1.0404
speedup and 3.4793 memory, own share 10.6%. Its speedup is below the 1.05 score
threshold. Evidence:
`runs_kaggle/20261030T112337362378Z_full_ce1ff09_eb596ed_recompute/out/runs/result.json`.
See README for the earlier fp16-norm and store/recompute experiments; different
sessions must use their own paired bases.

- [x] All 10 checks passed on ce1ff09, including 79 Soup regression tests.
- [x] README describes implementation, commands, measured results and limits.
- [ ] Final commit `<FINAL_COMMIT>` validated in order: base, solution, solution, base.
- [ ] Final public saved notebook `<KAGGLE_LINK>` and team/link placeholders filled.
- [ ] Complete solution suite passing on final commit. ce1ff09 had 214 passed / 4 failed; subsequent test changes are not a passing GPU log.
- [ ] Validate later RoPE and tied-embedding registration in the final paired run.

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
NF4 decoding is registered and measured at ce1ff09, while RoPE and lossless
embedding registration were added later and need new GPU evidence. Embedding
conversion is conditional on every table element round-tripping exactly through
half; dense non-AMP fallback may allocate a temporary fp32 table. Unsupported
norm consumers keep the original promoted dtype. Default MLP stores fp16
gate/up; recomputation trades projection work for memory. Chunked CE keeps
library GEMMs and Python iteration; higher-order derivatives are unsupported.

[Review](tasks/REVIEW.md): P0-1 own-share blocker has passing paired evidence;
S-018 closes P1-1's ignored-row/nonfinite gradient path in code; S-016/S-023
close P1-2's dense-fp32-without-AMP guard issue statically. P1-023-1 remains a
numerical limitation: early half norm output preserves dropout masks/RNG but
changes dropout values and shared-branch gradient accumulation. Passing public
checks 5–7 on one pair does not guarantee exact equivalence or hidden-row/seed
robustness. S-021 also records functional NF4 `out=` allocation/copy, flat NF4
backend routing, and the limited scope of chunk-GEMM bitwise tests.

The four ce1ff09 solution-test failures concerned a stale fake-MLP dtype
expectation, a LoRA/norm gradient comparison and two strict random RMSNorm
half-store comparisons. Tests were updated later, but GPU revalidation remains
pending. Final links, final two-repeat measurements and final solution-suite
results remain unfilled. Unpadding (S-024) and MLP offload (S-025) are in progress;
no accepted-feature or performance claim is made for them.
