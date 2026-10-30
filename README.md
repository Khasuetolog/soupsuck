# Soup Kernel Challenge: less memory for the same LoRA task

Our Triton kernels reduce activation and output-head memory during the fixed
Qwen2.5-1.5B-Instruct NF4/LoRA training task on a Kaggle Tesla T4.
The latest complete development comparison passed all ten harness checks:
**73.54 s and 5,863 MiB**, versus **82.25 s and 14,345 MiB** for base Soup.

| Measurement | Base Soup | Solution | Comparison |
| --- | ---: | ---: | --- |
| Whole `soup train` command | 82.25 s | 73.54 s | 1.1184× speedup |
| Peak GPU memory, sampled by nvidia-smi | 14,345 MiB | 5,863 MiB | 2.4467× base/solution |
| Public held-out answer-token loss | 2.04960 | 2.04969 | +0.004% vs base |
| Own Triton kernel share, steps 3 onward | — | 12.1% | Check 4 PASS |
| Harness checks | — | 10/10 PASS | One paired repetition |

**Measured commit:** `ce1ff09`, seed 1, 64 training steps on each side.
These are development results from **one** base/solution pair, not the final
two-repeat jury comparison. The separate solution suite reported **214 passed,
4 failed**; check 10 passed all **79** Soup regression tests. Subsequent test
changes require a new GPU run; all ten checks do not certify the solution suite.

Evidence root, abbreviated **E** throughout this document:
`runs_kaggle/20261030T112204781815Z_full_ce1ff09/out/`.
Summary numbers come from `E/runs/result.json:summary`, checks from
`E/runs/checks.json`, and tests from `E/pytest_solution.log` and check 10.
`E/source-manifest.json` records commit/archive hashes. These local development
artifacts are excluded from Git; the final public notebook must publish final
evidence. The earlier passing `de836e0` pair remains below as history.

- Public Kaggle notebook, saved version: `<KAGGLE_LINK>`
- Final submission commit: `<FINAL_COMMIT>`
- Submission links and team: [SUBMISSION.md](SUBMISSION.md)

## What we did

The entry point is [`solution.apply()`](solution/__init__.py).
Registration below describes committed HEAD `585de28`. Compared with the measured
`ce1ff09`, HEAD also registers lossless tied-embedding storage and fused RoPE.
Their performance and GPU correctness are **not** established by the ce1ff09
results. No result in this document is presented as a final-HEAD comparison.

### Frozen linear cross-entropy, entropy and predictions

[`solution/kernels/cross_entropy.py`](solution/kernels/cross_entropy.py) defines:

| `@triton.jit` name | Replaces / produces |
| --- | --- |
| `vocabulary_statistics_kernel` | Chunk softmax statistics, target logit and first argmax |
| `vocabulary_outputs_kernel` | Per-token CE, entropy and log-sum-exp |
| `vocabulary_derivative_kernel` | Softmax/CE derivative before the hidden-gradient GEMM |

**Registered:** the Qwen2 labelled-training head and TRL metrics bridge in
[`solution/patch.py`](solution/patch.py).
Instead of keeping all `[batch × sequence, vocabulary]` logits, the head visits
vocabulary chunks. Matrix multiplication remains `torch.mm`/cuBLAS. The registered implementation
uses fp16 operands and native fp16 logit output, then promotes logits
to fp32 for statistics, following the baseline bias-free AMP linear operation
without changing backend reduction flags. Measured de836e0 instead used fp32
GEMM output followed by explicit fp16 rounding; these two GEMM paths need not
be bitwise identical because cuBLAS may choose different reduction algorithms.

For each token, maintain maximum `m`, exponential sum `s`, and weighted sum `t`.
When a new chunk raises the maximum to `m_new`, rescale both sums by
`exp(m - m_new)`, then add the chunk terms:

```text
s += sum(exp(z - m_new))
t += sum(exp(z - m_new) * z)
lse = m + log(s)
loss = lse - z_target
entropy = lse - t/s
```

The implementation applies rescaling before the additions. It tracks the first
argmax across chunks, including ties, and masks vocabulary tails.
[`solution/autograd.py`](solution/autograd.py) shifts labels separately in each
row with `pad(labels, (0, 1), -100)[..., 1:]`. Ignored targets have zero per-token
loss; the final position of each row is ignored. The scalar loss is the sum
normalized by the supplied `num_items_in_batch`, or the valid-target count.

Backward recomputes chunk logits rather than saving them. The registered backward
selects valid-target rows before the backward GEMMs, then scatters dX into a
zero-filled full output; with no valid targets it returns zero dX immediately.
Forward still computes entropy and predictions for every position. Compaction
adds a nonzero synchronization outside the chunk loop. With upstream gradient
`g`, it computes `dZ = g * (softmax(z) - one_hot(target))`, casts `dZ/gain` to
fp16 for GEMM, accumulates dX in fp32, and multiplies by
`gain = max(abs(g), 1e-30)/1024`. This honors GradScaler and reduces fp16
underflow; it does not promise bitwise equality with baseline fp16 dZ.
The frozen head has no weight gradient. Excluding ignored rows before gain
computation removes the reviewed ignored-row `0 * inf` gradient path in 8f8ebbc;
the ce1ff09 paired checks pass, while the dedicated edge-case tests must also
be checked in the final suite. Zero-denominator training loss retains baseline
NaN semantics.

### Qwen2 RMSNorm

[`solution/kernels/rmsnorm.py`](solution/kernels/rmsnorm.py) defines:

| `@triton.jit` name | Replaces / produces |
| --- | --- |
| `rmsnorm_forward_kernel` | RMSNorm reduction and affine output |
| `rmsnorm_input_gradient_kernel` | Input-gradient pointwise/reduction chain |
| `rmsnorm_weight_partials_kernel` | Optional partial norm-weight gradients |
| `rmsnorm_weight_finalize_kernel` | Optional second-stage norm-weight reduction |

**Registered:** real Qwen2RMSNorm instances, including input and final norms.
The last two kernels are conditional and do not run for the frozen norms of the
fixed task; their absence is an informational check 3 note.

For `r = rsqrt(mean(x.float()²) + eps)`, forward computes
`q = cast(x.float()*r, x.dtype)`, then `y = weight*q` in the promoted dtype.
The cast occurs **before** multiplying by the norm weight, matching Qwen2.
For incoming gradient `g`, input backward uses
`u = cast(g*weight, x.dtype)` and
`dx = r*(u - (x*r)*mean(u*(x*r)))`, with fp32 arithmetic around the casts.
Weight gradients use independent partial tiles and a final reduction, without
atomics. Batch-dependent sizes and strides are runtime arguments.

`out_dtype=torch.float16` writes the final promoted result rounded to half in
the same pass. Input norms use explicit decoder scopes and require fp16 CUDA
AMP plus certified q/k/v consumers. For PEFT q/v, the registered vanilla LoRA
fp16-ingress path must be ready; k uses its certified dense/bnb path. The final
norm is scoped by the causal head and requires a certified ordinary dense head
under fp16 AMP. Unsupported wrappers, hooks or consumers keep the original
promoted norm output. A custom embedding/head forward can conservatively disable
the final-norm half optimization. Standalone norms have no half-output scope.
Backward returns dX in the input dtype and retains the original promoted dtype
for the optional norm-weight gradient. Early half output changes rounding and
shared-branch gradient accumulation; it is not bitwise-equivalent training.

### Attention residual add + post-attention RMSNorm

[`solution/kernels/add_rmsnorm.py`](solution/kernels/add_rmsnorm.py) defines
`add_rmsnorm_forward_kernel` and `add_rmsnorm_backward_kernel`.
**Registered:** [`solution/decoder.py`](solution/decoder.py) replaces the real
Qwen2 decoder's attention residual addition followed by post-attention norm.

Forward forms `s = residual + attention_output` in the promoted input dtype,
then applies the Qwen2 RMSNorm cast order above. It returns both `s` for the
residual branch and the normalized MLP input. Backward combines the gradient
from the residual branch with the norm Jacobian and casts to each input dtype.
It retains `s`, the weight reference and one fp32 reciprocal RMS per row.

By default, this post-attention norm writes fp16 output to reduce activation
storage before the frozen fp16 MLP projections. The residual stream remains in
its original promoted dtype. This is a precision optimization, not a promise
of bitwise end-to-end equivalence. The consumer guard now requires certified half projections and keeps the
original promoted dtype for a dense fp32 MLP outside AMP (review P1-2).

### LoRA scale/add epilogue and fp16 ingress

[`solution/kernels/lora_epilogue.py`](solution/kernels/lora_epilogue.py) defines
`lora_scale_add_kernel` and `lora_scale_gradient_kernel`.
**Registered:** the eligible vanilla PEFT Linear4bit path through
[`solution/lora.py`](solution/lora.py).

The forward replaces separate `adapter_output * scaling` and base-result add.
It rounds the product in the adapter-output dtype **before** adding, and keeps
PEFT's optional conversion to the base dtype outside autocast. Backward casts
to the branch dtype before scaling, matching the original two operations.

The patch keeps the original base linear, A/B projections and Dropout module.
For fp16 ingress it promotes the activation to fp32 before dropout, preserving
its dispatch shape and RNG advancement relative to the aligned fp32 path;
it does not recover values lost by earlier fp16 rounding. Dropout is still
called once, after the base projection, with the original probability/order.
The current default enables eligible attention input norms to write half,
using this ingress path for q/v adapters. Preserved masks/RNG do not imply
equal dropout values or equal adapter/input gradients after early rounding. Mixed
adapters, variants and unsupported states use original PEFT.

### Frozen MLP with tiled SwiGLU and optional recomputation

[`solution/kernels/swiglu_fused.py`](solution/kernels/swiglu_fused.py) defines
`swiglu_tile_engine`, one two-dimensional engine with forward/backward variants.
**Registered:** a lazy Qwen2MLP trampoline installs
[`solution/mlp.py`](solution/mlp.py) on first CUDA use. The supplied
[`solution/swiglu.py`](solution/swiglu.py) example is not registered and its
kernels do not count toward our contribution.

For gate `g`, up projection `u`, and `p = sigmoid(g)`, the engine computes:

```text
a  = (g*p)*u
dg = (da*u)*(p + (g*p)*(1-p))
du = da*(g*p)
```

Arithmetic is fp32 within the tile, with output casts chosen for the projection
graph. Row/column tails are masked. This fuses the pointwise forward and both
activation derivatives while leaving the linear projections and their backward
policy with PyTorch/bitsandbytes.

Default **store** mode saves fp16 gate/up tensors. Optional **recompute** mode
saves fp16 input instead and rebuilds gate/up under the captured forward AMP
context during backward. It adds projection work in exchange for lower retained
activation memory. Local GradientEdges preserve projection backward without
retaining the entire activation buffer as a tensor leaf. Frozen, supported
projection types qualify; trainable projections, hooks and wrappers fall back.
Only first-order gradients are implemented. Recompute was not used for the summary result. Its separate paired measurement
is reported below; store remains the default.

### NF4 decoder

[`solution/kernels/nf4.py`](solution/kernels/nf4.py) defines `nf4_decode_kernel`.
**Registered:** [`solution/nf4_patch.py`](solution/nf4_patch.py) patches the
bitsandbytes functional and CUDA backend dequantization paths after their
post-import dependencies are ready. The exact bitsandbytes version gate is
0.50.2. NF4 was active in measured ce1ff09 and contributes to its own-kernel share.
The 4-bit matrix multiplication itself remains a library operation.

The supported nested format is fp16 NF4, block size 64, with double-quantized
scales in blocks of 256. Earlier values occupy the high nibble of each byte.
For output index `i`, nested scale reconstruction is
`scale = fp32(code2[qabs[i//64]] * absmax2[(i//64)//256]) + offset`;
output is `fp16(NF4[nibble(i)] * scale)`. A second variant accepts decoded
fp32 scales, including non-nested functional calls routed through the patched
backend. FMA fusion is disabled to preserve intermediate scale rounding.
Unsupported versions, formats, devices and layouts use the original paths.
Review notes an extra allocation/copy for supported functional `out=` calls.

### Lossless storage of the tied embedding/head

[`solution/embedding.py`](solution/embedding.py) is **registered** after Soup's
`SFTTrainerWrapper._setup_transformers`. It accepts only an eligible frozen,
tied Qwen2 embedding/head after PEFT setup, with unchanged LoRA-only saving.
Every table element must satisfy `half_weight.float() == original_weight`;
the check is chunked and conversion is declined if any element fails.
The existing shared Parameter, tying, dimensions and names are preserved.
No new Triton kernel is introduced.

Only the table's storage becomes half. Embedding lookup is restored to the
original fp32 output; the dense head uses the same represented weights under
AMP. Outside AMP it temporarily reconstructs fp32 weights, so this optimization
does not promise lower eval peak memory. It was registered after ce1ff09:
no separate measured memory or speed gain is claimed here.

### Fused q/k rotary position embedding

[`solution/kernels/rope.py`](solution/kernels/rope.py) defines
`rope_pair_forward_kernel` and `rope_pair_backward_kernel`.
**Registered:** [`solution/rope.py`](solution/rope.py) replaces Qwen2's module-global
`apply_rotary_pos_emb`, which attention resolves at call time. It combines q/k
rotation, products and sum, preserving PyTorch type promotion and separate
product-rounding boundaries rather than using a fused multiply-add.
For the two half-head slices, backward computes
`dx1 = g1*cos1 + g2*sin2`, `dx2 = g2*cos2 - g1*sin1`, with cast boundaries
matching the original graph. Only cos/sin and metadata are saved.

The fast path supports even head dimensions, supported CUDA fp16/fp32 tensors,
`unsqueeze_dim=1` and non-trainable cos/sin. Other cases use plain PyTorch math;
no optional hub kernel is requested or called, including on fallback.
Sequence size is a runtime argument. RoPE was added after ce1ff09 and has no
measured share in that run; final GPU tests and profiling remain required.

## How it plugs in

The unmodified harness calls `solution.apply()` before Soup builds its model.
Targeted post-import hooks patch Qwen2, TRL SFTTrainer, PEFT's bnb LoRA class,
bitsandbytes dequantization and Soup model setup
before the importing caller receives the module. Already loaded modules are
handled too; markers make repeated installation idempotent. Cold registration
does not import/compile the CUDA kernels or eagerly import the training stack.

The head fast path requires the exact Qwen2 causal-LM class, training with
labels, a frozen bias-free dense head, supported contiguous CUDA fp16/fp32
inputs and the fp16 head precision path. Unsupported loss options, custom loss,
label smoothing, virtual tokens, parallelism or logit selection use the
original head/trainer path. CPU, fake test modules and unsupported tensor
layouts use guarded fallbacks. Eval/generation return ordinary logits; other
eligible layer patches can still run, subject to their own precision guards.

The TRL bridge keeps the same metric names and formulas: entropy uses all
attention-mask tokens; accuracy uses shifted non-ignored labels; num_tokens
accumulates attention-mask tokens. A fresh detached Soup ChunkedStep supplies
per-token losses to the unchanged Rewind row recorder. It preserves empty-row
counts and resets per-step state between calls. Saved LoRA adapters are evaluated
by the unmodified external evaluator without importing solution.

| Experimental environment flag | Submitted default behavior in the documented snapshot | Override for comparison |
| --- | --- | --- |
| `SOUP_FUSED_DISABLE` | Unset: registered optimizations enabled | `1`: use their original paths |
| `SOUP_FUSED_NORM_FP16` | Unset: eligible input, post-attention and final norm outputs fp16 | `0`: original promoted norm-output dtype |
| `SOUP_MLP_RECOMPUTE` | Unset: store gate/up | `1`: recompute gate/up in backward |

Final measurements must use defaults with these overrides unset. Flags are for
separate experiments; no behavior is selected by a profiler, test name or
measurement phase. No ready-made kernel libraries, torch.compile or CUDA graphs
are used by this implementation.

## How to run

### Public Kaggle notebook

1. Use the repository assigned by the hackathon platform. Import
   [`templates/kaggle_notebook.ipynb`](templates/kaggle_notebook.ipynb) into Kaggle.
2. Select **GPU T4 x2**, enable Internet, and set `REPO_URL` to the repository
   URL and `COMMIT` to `<FINAL_COMMIT>`. The harness uses one GPU. For a private
   repository, configure the read-only `GITHUB_TOKEN` Kaggle secret as the
   template describes.
3. Run all cells. For the final comparison, use
   `python harness/run_task.py --repeats 2` in the training cell, then run the
   unchanged checks and score cells. This runs base, solution, solution, base.
4. Save the completed notebook version, publish it, and replace `<KAGGLE_LINK>`
   here and in SUBMISSION with the same saved-version link. Preserve result,
   checks, metrics and logs with that notebook.

### Local Linux CUDA machine

Only T4 results count for the challenge. Install dependencies before the measured
run, then execute the unchanged kit commands from the repository root:

```bash
python -m pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" \
  "peft==0.20.0" "trl==0.29.1" "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py
python -m pytest solution -q
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
python harness/run_task.py --repeats 2
python harness/check_rules.py
python harness/score.py runs/result.json
```

The measured runtime was Python 3.13.15, torch 2.11.0+cu128, Triton 3.6.0 and
Tesla T4 driver 580.178.04, with the package versions above; source:
`E/runs/result.json:versions`. The head's backward relies on CUDA `torch.mm` support for
`out_dtype=torch.float32`; its forward logits use fp16 GEMM output. A different runtime must be validated.
The harness sets the allocator and restricts training to one GPU; do not
substitute a different allocator or prepopulate compiler caches.

### Development runner

[`tools/kaggle/run.py`](tools/kaggle/run.py) sends committed HEAD through a
private Kaggle code dataset and collects artifacts. Install/configure Kaggle CLI
first; keep credentials outside the repository in `~/.kaggle/kaggle.json`.
`KAGGLE_BIN` selects the CLI executable and defaults to `kaggle`.

```bash
python3 tools/kaggle/run.py run --mode unit --dry-run
python3 tools/kaggle/run.py push-code
python3 tools/kaggle/run.py run --mode unit
python3 tools/kaggle/run.py run --mode solution
python3 tools/kaggle/run.py run --mode full2
```

Commit accepted changes before `push-code`: uncommitted files are not uploaded.
`unit` runs tests, `solution` produces a solution-only run and profile, and
`full2` performs the paired two-repeat comparison and checks. Inspect preserved
pytest/check exit codes; a completed Kaggle notebook alone does not imply success.
This private iteration notebook does not replace the public submission link.

## Results and kernel-time accounting

The latest passing pair is **one repeat at ce1ff09**. Base metrics report a
57.51 s training loop and 0.9107 s median step; solution reports 51.52 s and
0.6868 s. Whole-command timing includes startup, cold compilation and saving.

| Passing development pair | Base / solution wall, s | Base / solution peak, MiB | Base / solution eval loss | Speedup / memory ratio | Own share | Checks |
| --- | --- | --- | --- | --- | --- | --- |
| de836e0 (history) | 77.47 / 70.05 | 14,345 / 5,843 | 2.04964 / 2.04962 | 1.1059 / 2.4551 | 8.6% | 10/10 |
| ce1ff09 (latest) | 82.25 / 73.54 | 14,345 / 5,863 | 2.04960 / 2.04969 | 1.1184 / 2.4467 | 12.1% | 10/10 |

Historical evidence:
`runs_kaggle/20261030T105526661826Z_full_de836e0/out/runs/result.json`.
These pairs ran on different sessions; their difference is not an isolated
measurement of any single optimization.

Shares below are training-time shares, not percentages of GPU kernel time or
whole-command shares. Timed solution steps 3–64 total **43.035 s**.
Each share is `100 * own_kernel_ms[name] / (1000 * timed_loop_s)` from
`E/runs/solution_1/metrics.json`. Base has no own kernels.

| Kernel | CUDA-event time, ms | Share of timed training |
| --- | ---: | ---: |
| `rmsnorm_forward_kernel` | 119.613 | 0.2779% |
| `rmsnorm_input_gradient_kernel` | 165.313 | 0.3841% |
| `add_rmsnorm_forward_kernel` | 222.172 | 0.5163% |
| `add_rmsnorm_backward_kernel` | 254.163 | 0.5906% |
| `lora_scale_add_kernel` | 148.014 | 0.3439% |
| `lora_scale_gradient_kernel` | 84.083 | 0.1954% |
| `vocabulary_statistics_kernel` | 262.717 | 0.6105% |
| `vocabulary_outputs_kernel` | 0.725 | 0.0017% |
| `vocabulary_derivative_kernel` | 112.667 | 0.2618% |
| `swiglu_tile_engine` | 1709.047 | 3.9713% |
| `nf4_decode_kernel` | 2148.202 | 4.9918% |
| **Total counted own kernels** | **5226.716** | **12.1453%** |

Frozen norm weights mean the two optional norm-weight-gradient kernels are
absent. RoPE and tied-embedding storage were not registered in ce1ff09; they
receive no measured credit here. No `example_*`, cuBLAS or other library kernel
time is counted. Check 4 rounds to 12.1%; final scoring uses the lowest eligible
share across solution repetitions.

### Precision and memory experiments

| Snapshot / setting | Solution wall, s | Peak, MiB | Median step, s | Eval loss |
| --- | ---: | ---: | ---: | ---: |
| 0e12e01, original post-attention norm dtype | 89.81 | 8,003 | 0.8215 | 2.04967 |
| 0e12e01, `SOUP_FUSED_NORM_FP16=1` | 76.64 | 5,843 | 0.7188 | 2.04966 |

Evidence directories under `runs_kaggle/`:
`20261030T104335675818Z_solution_0e12e01_default/out/` and
`20261030T104354956814Z_solution_0e12e01_e4e1167_fp16norm/out/`
(`runs_try/result.json` and `solution_1/metrics.json`). These solution-only
experiments motivated the post-attention half default; they do not isolate
later input/final norm compression or prove exact LoRA equivalence.

| MLP experiment | Solution wall / loop, s | Peak, MiB | Eval loss | Paired speedup / memory ratio |
| --- | --- | ---: | ---: | --- |
| 8f8ebbc store | 79.57 / 47.05 | 5,843 | 2.04966 | Solution only |
| 8f8ebbc recompute | 86.39 / 53.63 | 4,143 | 2.04966 | Solution only |
| ce1ff09 store (default) | 73.54 / 51.52 | 5,863 | 2.04969 | 1.1184 / 2.4467 |
| ce1ff09 recompute | 93.95 / 67.82 | 4,123 | 2.04968 | 1.0404 / 3.4793 |

8f8ebbc evidence: `runs_kaggle/20261030T111238818312Z_solution_8f8ebbc_store/out/`
and `20261030T111258418092Z_solution_8f8ebbc_eb596ed_recompute/out/`.
Recompute paired evidence:
`runs_kaggle/20261030T112337362378Z_full_ce1ff09_eb596ed_recompute/out/runs/result.json`.
Its base was 97.75 s, 14,345 MiB, eval loss 2.04964; all ten checks passed and
own share was 10.6%. The extra projection work reduces memory but its speedup
is below the score's 1.05 threshold. **Store remains the default.** Separate
sessions have different base timings; memory and speed ratios use each pair's
own base rather than mixing sessions.

| Final two-repeat comparison | Value |
| --- | --- |
| Commit / saved public notebook | `<FINAL_COMMIT>` / `<KAGGLE_LINK>` |
| Base / solution wall time | `<FINAL_BASE_TIME>` / `<FINAL_SOLUTION_TIME>` |
| Base / solution peak memory | `<FINAL_BASE_MEMORY>` / `<FINAL_SOLUTION_MEMORY>` |
| Base / solution eval loss | `<FINAL_BASE_LOSS>` / `<FINAL_SOLUTION_LOSS>` |
| Lowest own-kernel share / all ten checks | `<FINAL_KERNEL_SHARE>` / `<FINAL_CHECKS>` |

These placeholders are deliberately unfilled: the final two-repeat result is
not the measured one-repeat result. Score points depend on other admitted teams;
we report measured ratios rather than claiming a leaderboard score.

## Limitations and review status

The optimized head targets Qwen2 labelled training with a frozen dense head on
T4/fp16. bf16, trainable heads, higher-order derivatives and unsupported model or
trainer features are outside that fast path. Layer fallbacks are separate from
the head fallback and need their own precision validation.

Attention matmul/softmax and NF4 matrix multiplication remain library
operations; only rotary position application and NF4 decoding are replaced.
Chunked CE still performs library GEMMs and Python chunk iteration. Cold
compilation and setup count toward wall time. Lossless embedding conversion
requires an exact half roundtrip and may decline on another checkpoint.
RoPE and embedding registration postdate the latest full comparison.

[Independent review](tasks/REVIEW.md) contains S-017, S-021 and S-023:

| Finding | Current status / evidence boundary |
| --- | --- |
| P0-1: own share below 5% | Both de836e0 and ce1ff09 full pairs pass check 4; ce1ff09 is 12.1%. Final two-repeat evidence remains pending. |
| P1-1: all-ignored/nonfinite upstream can produce NaN dX | Closed in code by S-018: backward exits before gain/GEMMs if no valid targets and excludes ignored rows otherwise. S-021 confirms the structure; final dedicated tests still required. |
| P1-2: half norm reaches dense fp32 MLP outside AMP | Closed statically in S-016/S-023 by consumer guards; unsupported MLP keeps the promoted dtype. |
| P1-023-1: input-norm half changes LoRA values and shared-gradient accumulation | Numerical limitation remains. ce1ff09 checks 5–7 pass for one public pair; this does not prove equality or hidden-row/seed robustness. Dropout masks/RNG alone do not establish numerical equivalence. |
| P2-021-1: functional NF4 `out=` allocation/copy | Remains a performance limitation; alias result is preserved. |
| P2-021-2: functional non-nested NF4 can reach own flat backend | Documented routing; unsupported-format fallback does not mean all non-nested calls avoid our kernel. |
| P2-021-3: chunk CE equality test is not whole-head bitwise proof | Remains a test-scope limitation. cuBLAS algorithms can differ with GEMM shape. |
| P2-1: description/reproducibility | Updated here; final links, final commit and two-repeat result remain pending. |

The ce1ff09 solution suite had four failures: a stale fake-MLP dtype expectation,
an input/final half-norm gradient comparison, and two strict random half-store
RMSNorm comparisons. Subsequent test changes at 585de28 distinguish the guarded
fallback, rounding-boundary behavior and contraction tolerances. They are not a
new passing GPU log. Require a fresh complete solution suite and paired run of
the final commit before submission. Unpadding (S-024) and MLP offload (S-025)
are in progress and are not claimed as accepted features or measured gains.

## Challenge rules retained

The full task, scoring and checks are in the
[upstream starter kit](https://github.com/MakazhanAlpamys/soup-kernel-challenge).
Only `solution/` is copied into the jury's fresh kit; our development tools and
these reported artifacts do not modify its measurement code.

| Must remain unchanged | Fixed contract |
| --- | --- |
| `task/task.yaml`, `task/prepare_data.py` | Model, dataset selection/order, batch size, max length, steps, optimizer/schedule and LoRA settings |
| `harness/` | Timing, sampled memory, evaluator, checks and scoring |
| `tests/` | Soup regression tests |
| Trainer settings and Soup outputs | Same adapter parameters/shapes, metrics, token counts and Rewind rows/losses within allowed tolerances |
| Measurement environment | One T4 GPU, harness allocator, no background GPU work, warm compiler cache or measurement-dependent behavior |

The task uses 256 public training rows, batch size 4 and maximum length 512,
one epoch with LoRA r=16, alpha=32 and the unchanged Soup dropout/target defaults;
source: `task/task.yaml` and the upstream kit. We do not add trainable model
parameters or load ready-made kernel libraries.

## Sources of ideas and license

These are algorithm/format references, not imported or vendored kernel libraries:

- [Milakov & Gimelshein, Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867): online max/sum updates.
- [Cut Your Losses in Large-Vocabulary Language Models](https://arxiv.org/abs/2411.09009): avoiding full-vocabulary logits.
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel) and [Apex normalization](https://github.com/NVIDIA/apex/tree/master/apex/normalization): linear/CE and normalization fusion ideas.
- [PEFT Linear4bit](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/bnb.py): original LoRA control flow and rounding order.
- [bitsandbytes 0.50.2](https://github.com/bitsandbytes-foundation/bitsandbytes/tree/0.50.2): NF4 format, nibble order and nested-scale reconstruction.
- Standard SiLU/SwiGLU derivatives and RMSNorm Jacobian; our tiled kernels are written in this repository.

Code: Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE).
Model: [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
Data: [AmanMussa/kazakh-instruction-v2](https://huggingface.co/datasets/AmanMussa/kazakh-instruction-v2)
(MIT on its card), a Kazakh translation of
[Stanford Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) (CC BY-NC 4.0).
