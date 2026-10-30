# Soup Kernel Challenge: bounded-memory output head and RMSNorm

Draft for the final solution README. GPU results are not yet measured.

- Public Kaggle notebook, saved version: `<KAGGLE_LINK>`
- Final repository commit: `<FINAL_COMMIT>`

## What changes

The solution targets the output head of the fixed Qwen2 LoRA training task.
It computes causal cross-entropy, entropy and predictions without retaining
the full token-by-vocabulary logits tensor. A separate RMSNorm implementation
preserves Qwen2's cast order and supports backward through the decoder.

The entry point is [`solution.apply()`](../solution/__init__.py).
The harness calls it before Soup constructs the model and trainer.
The jury copies the solution package; the helper tools support reproduction and review.

### Why the output head

Let `N = batch_size * sequence_length`, `H` be the hidden width and `V` the
vocabulary size. A dense head produces `[N,V]` logits. In the target AMP path,
the head produces fp16 values and Accelerate exposes an fp32 copy to the trainer.
Those full tensors scale with `N*V` and can occupy gigabytes for this workload.
This is a tensor-size argument, not a measured reduction in peak GPU memory.

Removing logits alone would break the task: TRL reads them for entropy and token
accuracy, and Soup Rewind records per-row supervised loss. The replacement
therefore supplies those statistics along with the differentiable training loss.
The decoder, LoRA layers, dropout, optimizer, schedule and data order are retained.

### Frozen linear cross-entropy

[`fused_linear_cross_entropy`](../solution/kernels/cross_entropy.py) receives
hidden states `[N,H]`, a frozen dense head `[V,H]` and already shifted targets.
It returns per-token loss, entropy, argmax and log-sum-exp `[N]`.
Only per-token loss is differentiable; statistics are detached.

Vocabulary chunks are processed sequentially. Each chunk uses `torch.mm` with
fp16 operands and fp32 accumulation/output; the implementation relies on CUDA
PyTorch support for `out_dtype=torch.float32`. These GEMMs use cuBLAS.
The custom Triton work is the fused vocabulary statistics and softmax derivative,
rather than a custom Triton matrix multiplication.

The chunk policy in `_chunk_size` bounds temporary logits and weight casts.
Forward retains the fp16 hidden states, a reference to the frozen head, targets
and LSE for backward. It does not retain vocabulary-sized logits for every token.
Backward recomputes one vocabulary chunk at a time and accumulates `dX` in fp32.
There is no head-weight gradient because the eligible head is frozen.

For a row, maintain running maximum `m`, exponential sum `s`, and weighted sum `t`.
When a new chunk changes the running maximum to `m_new`, update:

```text
alpha = exp(m_old - m_new)
s_new = alpha * s_old + sum_chunk(exp(z - m_new))
t_new = alpha * t_old + sum_chunk(exp(z - m_new) * z)
lse = m + log(s)
cross_entropy = lse - z_target
entropy = lse - t / s
```

The target logit and first argmax are tracked across chunks; masked tails do not contribute.
Ignored targets produce zero per-token loss and gradient, while entropy and
argmax are still computed for their positions.

For upstream per-row gradient `g`, backward computes:

```text
p = exp(z - lse)
dZ = g * (p - one_hot(target))
gain = max(abs(g), 1e-30) / 1024
dX = gain * sum_chunks((dZ / gain).to(fp16) @ W_chunk.to(fp16))
```

The final multiplication by `gain` follows fp32 accumulation of the chunk GEMMs.
This row scaling reduces fp16 underflow/overflow risk when casting the derivative
for tensor-core GEMM. It honors the actual incoming gradient, including an
external GradScaler multiplier, rather than assuming an unscaled mean loss.

### AMP dtype and loss alignment

The logit contract is `fp32(round_fp16(fp16(hidden) @ fp16(weight).T))`.
The vocabulary kernels explicitly round the fp32 GEMM result through fp16
before computing fp32 CE, entropy, argmax and LSE.
Using the unrounded fp32 GEMM output would describe a different loss path.
Head backward returns the input hidden dtype; the frozen fp32 head is cast by
chunk, without constructing a full fp16 copy of the vocabulary weight.

[`fused_head`](../solution/autograd.py) constructs targets as
`pad(labels, (0, 1), value=-100)[..., 1:]`, preserving each batch row's boundary.
The final target is ignored. Loss is the sum divided by `num_items_in_batch` when supplied;
otherwise it is divided by the number of valid shifted targets.
The denominator is not replaced by a batch-size constant or a mean of row means.
An all-ignored batch is not silently assigned a finite zero training loss.

### TRL and Rewind integration

[`patch.py`](../solution/patch.py) patches `Qwen2ForCausalLM.forward` between the
decoder output and the head. The eligible path executes the decoder once and
returns `CausalLMOutputWithPast(loss=..., logits=None, ...)`, preserving its other
output fields. Unsupported decoder layouts use the ordinary head on the decoder
result already computed.

The `SFTTrainer.compute_loss` bridge delegates the eligible forward and loss
handling to `transformers.Trainer.compute_loss`, retaining its loss-kwargs and
cross-device scaling behavior. It then reproduces TRL's three metrics:

- `entropy`: attention-mask-weighted entropy over all positions, including prompt
  and final positions; when position IDs replace the mask, use the position mean.
- `mean_token_accuracy`: compare argmax at `[:, :-1]` with labels at `[:, 1:]`,
  ignoring `-100`; report zero when there are no valid targets.
- `num_tokens`: accumulate training attention-mask tokens, including prompt
  tokens; the position-ID alternative follows TRL's token-count convention.

The bridge uses `accelerator.gather_for_metrics` for the corresponding reductions.
These metrics deliberately use different masks and counts.

A fresh Soup `ChunkedStep` carries detached `[B,S]` token losses and entropies,
plus correct/labelled counts, through Soup's existing `STEP_ATTR`.
Rewind's unchanged `row_losses_from_token_losses` computes row means and counts.
The last loss column is zero. Model and trainer step state are cleared between
calls so an unsupported next batch cannot reuse the previous batch's statistics.
Soup creates its Rewind subclass dynamically; its `super().compute_loss` resolves
the patched TRL class, so the bridge also works through that inheritance path.
The sampler, FIFO row identity and Rewind recording method are unchanged.

### RMSNorm

[`rms_norm`](../solution/kernels/rmsnorm.py) computes variance and reciprocal RMS
in fp32. It casts the normalized value back to the input dtype **before**
multiplication by the norm weight, then returns the promoted output dtype.
This matters when fp16 activations meet a frozen fp32 norm weight.

The input gradient uses the RMSNorm Jacobian with the corresponding cast order.
Weight gradients use staged partial reductions only when `weight.requires_grad`.
Frozen norm weights do not allocate partial-gradient buffers or launch those
weight-gradient kernels. Only first-order gradients are implemented.

### Own Triton kernels

These are the actual `@triton.jit` definitions; execution depends on requested gradients.

| File | Kernel | Purpose |
| --- | --- | --- |
| [cross_entropy.py](../solution/kernels/cross_entropy.py) | `vocabulary_statistics_kernel` | Online normalizer, weighted sum, target logit and argmax |
| [cross_entropy.py](../solution/kernels/cross_entropy.py) | `vocabulary_outputs_kernel` | Per-token loss, entropy and LSE |
| [cross_entropy.py](../solution/kernels/cross_entropy.py) | `vocabulary_derivative_kernel` | Recomputed softmax derivative with row scaling |
| [rmsnorm.py](../solution/kernels/rmsnorm.py) | `rmsnorm_forward_kernel` | RMSNorm forward and reciprocal RMS |
| [rmsnorm.py](../solution/kernels/rmsnorm.py) | `rmsnorm_input_gradient_kernel` | Gradient through the norm input |
| [rmsnorm.py](../solution/kernels/rmsnorm.py) | `rmsnorm_weight_partials_kernel` | Optional weight-gradient partials |
| [rmsnorm.py](../solution/kernels/rmsnorm.py) | `rmsnorm_weight_finalize_kernel` | Optional weight-gradient reduction |

The kit's [`swiglu.py`](../solution/swiglu.py) example remains enabled when its
dependencies are available. `example_swiglu_fwd_kernel` and
`example_swiglu_bwd_kernel` are supplied example kernels, not our contribution.
Their time does not count toward the required own-kernel share.

## How to reproduce

Import [`templates/kaggle_notebook.ipynb`](../templates/kaggle_notebook.ipynb) into
Kaggle. Select **GPU T4 x2**, enable Internet, set the repository URL and set
`COMMIT` to `<FINAL_COMMIT>`, then **Run All**. A private repository needs the
read-only `GITHUB_TOKEN` Kaggle secret described in the template.
Save the completed notebook version and replace `<KAGGLE_LINK>` above; put the
same link and commit in the final copy of [`SUBMISSION.md`](../templates/SUBMISSION.md).

On a Linux CUDA machine, use the kit's pinned dependencies and commands:

```bash
python -m pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" \
  "peft==0.20.0" "trl==0.29.1" "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py
python -m pytest solution -q -rs
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
python harness/run_task.py
python harness/check_rules.py
python harness/score.py runs/result.json
# Repeat the base/solution comparison in the jury order:
python harness/run_task.py --repeats 2
python harness/check_rules.py
python harness/score.py runs/result.json
```

For private delivery, keep CLI credentials at `~/.kaggle/kaggle.json`, outside the repo;
see [`tools/kaggle/README.md`](../tools/kaggle/README.md).
`KAGGLE_BIN` selects the installed CLI executable; it defaults to `kaggle`.

```bash
python3 tools/kaggle/run.py run --mode unit --dry-run
python3 tools/kaggle/run.py push-code
python3 tools/kaggle/run.py run --mode unit
python3 tools/kaggle/run.py run --mode solution
python3 tools/kaggle/run.py run --mode full2
```

`push-code` uploads committed HEAD only to a private dataset, with file hashes.
Commit accepted changes before upload. The notebook verifies the expected SHA.
`unit` runs solution and patched Soup tests; `solution` profiles only the solution;
`full` compares both sides, and `full2` repeats both sides in the jury order.
Commands run sequentially; no extra GPU workload runs beside the harness.
Logs and exit codes are preserved under `runs_kaggle/`, even after test failures.
The final public submission uses the jury template, not the private iteration URL.

## Results and limitations

No target-GPU result is claimed; fill these cells only from the final saved T4 run.

| Measurement | Base Soup | Solution | Ratio / status |
| --- | --- | --- | --- |
| Whole `soup train` wall time, s | `<не измерено>` | `<не измерено>` | `<не измерено>` |
| Peak GPU memory, MiB | `<не измерено>` | `<не измерено>` | `<не измерено>` |
| Held-out answer-token eval loss | `<не измерено>` | `<не измерено>` | `<не измерено>` |
| Eligible own-kernel time / timed training | `<не измерено>` | `<не измерено>` | `<не измерено>` |
| All ten harness checks | `<не измерено>` | `<не измерено>` | `<не измерено>` |

Use `runs/result.json` for wall time, sampled GPU peak, eval loss and ratios;
`runs/checks.json` for every check; and solution `metrics.json` plus check 4 for
the eligible kernel share. Compilation belongs in the measured training time.

The fast head requires labelled training on the real Qwen2 class with a frozen,
bias-free dense head, contiguous CUDA fp16/fp32 activations/weights, int64 labels
and the fp16 linear path.
The integration checks CUDA capability; the target remains T4/fp16.
CPU, eval, generation/no labels, unsupported dtypes/layouts and non-default
logit selection take fallback paths. Unsupported loss options, virtual-token
trainer paths, custom loss/smoothing and configured parallelism bypass the bridge.
RMSNorm has its own CUDA/dtype/layout guard and can run independently of the head.

GPU loss/gradient equivalence, metric agreement, saved-adapter quality, kernel
share and the full Soup regression suite remain **UNVERIFIED** on T4.
Chunked GEMMs and fp16 casts may trade throughput for memory; no speed gain is
assumed. The implementation still materializes bounded chunk logits, recomputes
head chunks in backward, and performs Python-level chunk iteration.
Attention, NF4 dequantization, LoRA projections, optimizer updates and padding
are not newly optimized here. Higher-order gradients and a bf16 head are unsupported.
`SOUP_FUSED_DISABLE=1` disables the new head/norm paths for debugging; leave it
unset for the submitted run. The supplied SwiGLU example has its own registration.

## Sources of ideas

These references describe algorithms or design patterns; their kernels are not
imported or vendored by this solution. No ready-made kernel library is required.

- [Milakov and Gimelshein, *Online normalizer calculation for softmax* (2018)](https://arxiv.org/abs/1805.02867): online maximum and normalizer updates.
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel): combining a linear head with CE and RMSNorm row reductions.
- [Wijmans et al., *Cut Your Losses in Large-Vocabulary Language Models* (2024)](https://arxiv.org/abs/2411.09009): avoiding full vocabulary-logit materialization.
- [NVIDIA Apex RMSNorm](https://github.com/NVIDIA/apex/blob/master/apex/normalization/fused_layer_norm.py): normalization and backward design context.

The kit and solution use Apache-2.0; see [LICENSE](../LICENSE) and [NOTICE](../NOTICE).
