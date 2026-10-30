"""Unpadded Qwen2 acceptance: row isolation, padded LoRA RNG and Soup statistics.

Pure layout/scope tests run without torch. Runtime tests run on CPU when torch
and pinned transformers/PEFT exist, and on CUDA/fp16 otherwise. bnb tests need
CUDA and are explicitly skipped elsewhere, never replaced by a dense oracle.
"""

from contextlib import nullcontext
import sys
from types import ModuleType, SimpleNamespace

import pytest

from solution import unpad


@pytest.mark.parametrize("rows,expected", [
    ([[1, 1, 1], [1, 0, 0]], (3, 1)),
    ([[1], [1]], (1, 1)),
    ([[1, 1], [1, 1]], (2, 2)),
    ([[True, False]], (1,)),
    ([[0, 1]], None),
    ([[1, 0, 1]], None),
    ([[0, 0]], None),
    ([[1, 2]], None),
    ([[1], [1, 0]], None),
    ([], None),
    ([[]], None),
])
def test_right_padding_contract(rows, expected):
    assert unpad.right_padding_lengths(rows) == expected


def test_batch_scope_restores_after_nested_exception():
    first, second = object(), object()
    assert unpad._ACTIVE.get() is None
    with unpad._packing_scope(first):
        with pytest.raises(ValueError), unpad._packing_scope(second):
            assert unpad._ACTIVE.get() is second
            raise ValueError("sentinel")
        assert unpad._ACTIVE.get() is first
    assert unpad._ACTIVE.get() is None


def test_packed_head_capture_is_owned_consumed_once_and_exception_safe():
    owner, foreign, labels = object(), object(), SimpleNamespace(shape=(2, 3), device="cpu")
    batch = unpad.PackedBatch(2, 3, 4, (3, 1), SimpleNamespace(device="cpu"), frozenset(), frozenset())
    packed = SimpleNamespace(shape=(1, 4, 4), dtype="fp16", device="cpu")
    padded = SimpleNamespace(shape=(2, 3, 4), dtype="fp16", device="cpu")
    token = unpad._LABELS.set((labels, object()))
    try:
        with unpad.capture_packed_head(owner, labels) as captured:
            unpad._publish_packed_head(foreign, packed, batch, padded)
            assert captured.payload is None
            with pytest.raises(ValueError), unpad.capture_packed_head(foreign, labels):
                raise ValueError("restore nested head scope")
            assert unpad._HEAD_CAPTURE.get() is captured
            unpad._publish_packed_head(owner, packed, batch, padded)
        assert unpad._HEAD_CAPTURE.get() is None
        assert captured.take(padded, labels).hidden is packed
        assert captured.take(padded, labels) is None
        with unpad.capture_packed_head(owner, labels) as stale:
            unpad._publish_packed_head(owner, packed, batch, padded)
        assert stale.take(object(), labels) is None
        assert stale.take(padded, labels) is None
        with unpad.capture_packed_head(owner, labels) as wrong_labels:
            unpad._publish_packed_head(owner, packed, batch, padded)
        assert wrong_labels.take(padded, object()) is None
    finally:
        unpad._LABELS.reset(token)


def test_fake_module_registration_is_a_noop():
    assert not unpad.patch_qwen2_unpad(ModuleType(unpad._QWEN))
    assert not unpad.patch_qwen2_unpad(object())


def test_developer_flag(monkeypatch):
    monkeypatch.delenv("SOUP_FUSED_DISABLE", raising=False)
    monkeypatch.delenv("SOUP_UNPAD", raising=False)
    assert unpad.unpad_enabled()
    monkeypatch.setenv("SOUP_UNPAD", "0")
    assert not unpad.unpad_enabled()
    monkeypatch.setenv("SOUP_UNPAD", "1")
    monkeypatch.setenv("SOUP_FUSED_DISABLE", "1")
    assert not unpad.unpad_enabled()


def test_instance_binding_does_not_modify_dropout_class(monkeypatch):
    # Exercise instance ownership even on hosts without a torch wheel.
    class Dropout:
        def forward(self, x):
            return x
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32="float32"))
    selected, foreign = Dropout(), Dropout()
    native = Dropout.forward
    unpad._bind_dropout(selected)
    installed = selected.forward
    unpad._bind_dropout(selected)
    value = object()
    assert selected.forward is installed and selected.forward(value) is value
    assert foreign.forward(value) is value and "forward" not in foreign.__dict__
    assert Dropout.forward is native


def test_plan_preserves_row_order_and_rejects_supervised_padding(monkeypatch):
    """Control-flow check on metadata; no pretend tensor/numerical acceptance."""
    device = SimpleNamespace(type="cpu")
    transfers = []
    class Tensor:
        requires_grad = False
        def __init__(self, data, **kwargs):
            self.data, self.device = data, device
            self.shape = ((len(data), len(data[0])) if data and isinstance(data[0], list)
                          else (len(data),))
            self.ndim = len(self.shape)
        def detach(self):
            return self
        def tolist(self):
            transfers.append(self)
            return self.data
        def flat(self):
            return [x for row in self.data for x in row] if self.ndim == 2 else self.data
        def reshape(self, *shape):
            return Tensor(self.flat())
        def to(self, dtype):
            return self
        def __eq__(self, value):
            if self.ndim == 2:
                return Tensor([[x == value for x in row] for row in self.data])
            return Tensor([x == value for x in self.data])
        def __ne__(self, value):
            return self == (1 if value == 0 else 0)
        def combine(self, other, op):
            if self.ndim == 2:
                return Tensor([[op(x, y) for x, y in zip(a, b)] for a, b in zip(self.data, other.data)])
            return Tensor([op(x, y) for x, y in zip(self.data, other.data)])
        def __or__(self, other):
            return self.combine(other, lambda x, y: x or y)
        def __and__(self, other):
            return self.combine(other, lambda x, y: x and y)
        def __getitem__(self, selector):
            raise AssertionError("dynamic boolean selection is forbidden in preflight")
        def __bool__(self):
            raise AssertionError("separate scalar D2H is forbidden in preflight")
        def all(self):
            return Tensor([all(self.flat())])
        def narrow(self, axis, start, length):
            parent = self
            class Slice:
                def add_(self, value):
                    for index in range(start, start + length):
                        parent.data[index] += value
            return Slice()
    fake = SimpleNamespace(Tensor=Tensor, long="int64", uint8="uint8",
                           arange=lambda n, **kw: Tensor(list(range(n))),
                           cat=lambda values: Tensor([x for value in values for x in value.data]))
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(unpad, "_dropouts", lambda model, module: frozenset())
    monkeypatch.delenv("SOUP_UNPAD", raising=False)
    monkeypatch.delenv("SOUP_FUSED_DISABLE", raising=False)
    class Model:
        pass
    model = Model()
    model.training = True
    model._forward_hooks = model._forward_pre_hooks = {}
    model._backward_hooks = model._backward_pre_hooks = {}
    model.config = SimpleNamespace(use_cache=False, _attn_implementation="sdpa", hidden_size=64)
    model.rotary_emb = SimpleNamespace(rope_type="default")
    model.layers = [SimpleNamespace(self_attn=object())]
    module = SimpleNamespace(Qwen2Model=Model)
    mask = Tensor([[1, 0, 0], [1, 1, 1]])
    ids = Tensor([[5, 0, 0], [8, 6, 7]])
    labels = Tensor([[-100, -100, -100], [-100, 6, 7]])
    token = unpad._LABELS.set((labels, mask))
    try:
        plan = unpad._plan(model, module, ids, mask, None, None, None, False, {})
        assert plan.lengths == (1, 3) and plan.indices.data == [0, 3, 4, 5]
        assert len(transfers) == 1
        assert transfers[0].shape == (7,)  # B*S mask bits plus one validity bit
        labels.data[0][2] = 9
        assert unpad._plan(model, module, ids, mask, None, None, None, False, {}) is None
        assert len(transfers) == 2
    finally:
        unpad._LABELS.reset(token)


@pytest.fixture
def runtime(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import solution

    # Install the fused head before the outer unpad labels wrapper, matching
    # the real post-import callback order (also works when already installed).
    solution.apply()
    from transformers.models.qwen2 import modeling_qwen2 as module

    for cls in (module.Qwen2ForCausalLM, module.Qwen2Model, module.Qwen2Attention):
        monkeypatch.setattr(cls, "forward", cls.forward)
    native_dropout = torch.nn.Dropout.forward
    assert unpad.patch_qwen2_unpad(module)
    installed = tuple(cls.forward for cls in (
        module.Qwen2ForCausalLM, module.Qwen2Model, module.Qwen2Attention))
    assert unpad.patch_qwen2_unpad(module)
    assert installed == tuple(cls.forward for cls in (
        module.Qwen2ForCausalLM, module.Qwen2Model, module.Qwen2Attention))
    assert torch.nn.Dropout.forward is native_dropout  # no global Dropout patch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch, module, device


def _rng(torch, device):
    return torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()


def _set_rng(torch, device, state):
    (torch.cuda.set_rng_state if device == "cuda" else torch.set_rng_state)(state)


def _amp(torch, device):
    return torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext()


def _batch(torch, device, lengths, sequence=None):
    sequence = sequence or max(lengths)
    ids = torch.randint(1, 127, (len(lengths), sequence), device=device)
    mask = torch.arange(sequence, device=device)[None, :] < torch.tensor(lengths, device=device)[:, None]
    ids = ids.masked_fill(~mask, 0)
    labels = ids.masked_fill(~mask, -100)
    labels[:, 0] = -100
    return ids, labels, mask.long()


def _model(torch, module, device, backend):
    peft = pytest.importorskip("peft")
    from transformers import Qwen2Config

    torch.manual_seed(24024)
    config = Qwen2Config(vocab_size=127, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        attention_dropout=0.0, use_cache=False)
    config._attn_implementation = "sdpa"
    model = module.Qwen2ForCausalLM(config).train()
    model.requires_grad_(False)
    if backend == "bnb":
        if device != "cuda":
            pytest.skip("bnb acceptance needs CUDA")
        bnb = pytest.importorskip("bitsandbytes")
        if bnb.__version__ != "0.50.2":
            pytest.skip("bnb acceptance pinned to 0.50.2")
        for block in model.model.layers:
            for name in ("q_proj", "k_proj", "v_proj"):
                old = getattr(block.self_attn, name)
                quantized = bnb.nn.Linear4bit(old.in_features, old.out_features,
                                             bias=old.bias is not None, compute_dtype=torch.float16,
                                             compress_statistics=True, quant_type="nf4")
                quantized.weight = bnb.nn.Params4bit(old.weight.detach().clone(), requires_grad=False,
                                                    compress_statistics=True, quant_type="nf4")
                if old.bias is not None:
                    quantized.bias = torch.nn.Parameter(old.bias.detach().clone(), requires_grad=False)
                setattr(block.self_attn, name, quantized)
        model.is_loaded_in_4bit = True
    model.to(device)
    lora = peft.LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=8,
                           lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
    if backend == "dense":
        from peft.tuners.lora.layer import Linear

        lora._register_custom_module({torch.nn.Linear: Linear})
    wrapped = peft.get_peft_model(model, lora).train()
    if backend == "bnb":
        from solution.lora import patch_lora_4bit

        patch_lora_4bit()
    with torch.no_grad():
        for name, param in wrapped.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 0.03)
    return wrapped, model


def _statistics(torch, output, raw, labels):
    if output.logits is None:
        step = raw._solution_fused_step
        return tuple(t.detach().clone() for t in (
            step.token_loss, step.token_entropy, step.correct, step.labelled))
    with torch.no_grad():
        logits = output.logits.float()
        targets = torch.nn.functional.pad(labels, (0, 1), value=-100)[:, 1:]
        losses = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                   targets.reshape(-1), reduction="none").reshape(labels.shape)
        logp = logits.log_softmax(-1)
        entropy = -(logp.exp() * logp).sum(-1)
        valid = labels[:, 1:] != -100
        correct = ((logits[:, :-1].argmax(-1) == labels[:, 1:]) & valid).sum()
        return losses, entropy, correct, valid.sum()


@pytest.mark.parametrize("backend", ["dense", "bnb"])
@pytest.mark.parametrize("lengths,sequence", [
    ((7, 4, 1, 6), 7), ((7,), 7), ((7, 7, 7, 7), 7), ((5, 5, 5, 5), 9),
])
def test_padded_equivalence_loss_statistics_lora_gradients_rng(runtime, monkeypatch, backend, lengths, sequence):
    torch, module, device = runtime
    wrapped, raw = _model(torch, module, device, backend)
    ids, labels, mask = _batch(torch, device, lengths, sequence)
    initial = _rng(torch, device)
    captures, plans = [], []
    dropout = torch.nn.functional.dropout
    plan = unpad._plan
    head_rows = []
    if device == "cuda":
        from solution.kernels import cross_entropy as ce

        head = ce.fused_linear_cross_entropy
        def observed_head(hidden, weight, targets, ignore_index=-100):
            head_rows.append(hidden.shape[0])
            return head(hidden, weight, targets, ignore_index=ignore_index)
        monkeypatch.setattr(ce, "fused_linear_cross_entropy", observed_head)

    def observed_dropout(input, p=0.5, training=True, inplace=False):
        output = dropout(input, p=p, training=training, inplace=inplace)
        if training and p == 0.05:
            assert input.shape == (len(lengths), sequence, 64)
            assert input.dtype == torch.float32 and input.is_contiguous()
            assert input.data_ptr() % 16 == 0
            captures.append((output.detach() != 0).clone())
        return output

    def observed_plan(*args, **kwargs):
        value = plan(*args, **kwargs)
        plans.append(value)
        return value

    monkeypatch.setattr(torch.nn.functional, "dropout", observed_dropout)
    monkeypatch.setattr(unpad, "_plan", observed_plan)
    monkeypatch.setenv("SOUP_UNPAD", "0")
    with _amp(torch, device):
        reference = wrapped(input_ids=ids, labels=labels, attention_mask=mask, use_cache=False)
    reference_loss = reference.loss.detach().clone()
    reference_stats = _statistics(torch, reference, raw, labels)
    reference_rng = _rng(torch, device)
    reference.loss.mul(128).backward()
    reference_grads = {name: p.grad.detach().clone() for name, p in wrapped.named_parameters()
                       if "lora_" in name and p.grad is not None}
    reference_masks = list(captures)
    del reference
    captures.clear()
    wrapped.zero_grad(set_to_none=True)
    _set_rng(torch, device, initial)
    monkeypatch.setenv("SOUP_UNPAD", "1")
    with _amp(torch, device):
        actual = wrapped(input_ids=ids, labels=labels, attention_mask=mask, use_cache=False)
    actual_stats = _statistics(torch, actual, raw, labels)
    assert torch.equal(_rng(torch, device), reference_rng)
    assert len(captures) == len(reference_masks) == 4  # q/v, two layers
    for got, expected in zip(captures, reference_masks):
        assert torch.equal(got[mask.bool()], expected[mask.bool()])
    expect_packed = sum(lengths) < len(lengths) * sequence
    assert (plans[-1] is not None) == expect_packed
    if expect_packed:
        assert plans[-1].tokens == sum(lengths)
    if device == "cuda":
        assert head_rows == [len(lengths) * sequence, sum(lengths)]
    tolerance = dict(rtol=3e-3, atol=3e-3) if device == "cuda" else dict(rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual.loss, reference_loss, **tolerance)
    losses, entropy, correct, count = actual_stats
    expected_losses, expected_entropy, expected_correct, expected_count = reference_stats
    torch.testing.assert_close(losses, expected_losses, **tolerance)
    torch.testing.assert_close(entropy[mask.bool()], expected_entropy[mask.bool()], **tolerance)
    torch.testing.assert_close((entropy * mask).sum() / mask.sum(),
                               (expected_entropy * mask).sum() / mask.sum(), **tolerance)
    assert count == expected_count
    assert correct == expected_correct
    # Rewind's row mean/count formula, including the length-one row.
    counts = (labels[:, 1:] != -100).sum(-1)
    torch.testing.assert_close(losses.sum(-1) / counts.clamp_min(1),
                               expected_losses.sum(-1) / counts.clamp_min(1), **tolerance)
    actual.loss.mul(128).backward()
    for name, p in wrapped.named_parameters():
        if name in reference_grads:
            ref = reference_grads[name]
            assert p.grad is not None and torch.isfinite(p.grad).all()
            torch.testing.assert_close(p.grad, ref, rtol=3e-2 if device == "cuda" else 2e-4,
                                       atol=1e-3 if device == "cuda" else 2e-6, msg=name)
            assert (p.grad - ref).norm() <= (0.03 if device == "cuda" else 2e-4) * ref.norm() + 2e-6
    assert unpad._ACTIVE.get() is None and unpad._LABELS.get() is None
    assert unpad._HEAD_CAPTURE.get() is None


@pytest.mark.parametrize("reason", ["eval", "cache", "left", "no_mask", "hidden", "pad_label"])
def test_unsupported_calls_fall_back_before_packing(runtime, monkeypatch, reason):
    torch, module, device = runtime
    wrapped, _ = _model(torch, module, device, "dense")
    ids, labels, mask = _batch(torch, device, (7, 4))
    kwargs = dict(input_ids=ids, labels=labels, attention_mask=mask, use_cache=False)
    if reason == "eval":
        wrapped.eval()
    elif reason == "cache":
        kwargs["use_cache"] = True
    elif reason == "left":
        kwargs["attention_mask"] = mask.flip(-1)
    elif reason == "no_mask":
        kwargs.pop("attention_mask")
    elif reason == "hidden":
        kwargs["output_hidden_states"] = True
    else:
        labels[mask == 0] = 1
    observed = []
    plan = unpad._plan
    def spy(*args, **options):
        result = plan(*args, **options)
        observed.append(result)
        return result
    monkeypatch.setattr(unpad, "_plan", spy)
    with torch.no_grad(), _amp(torch, device):
        wrapped(**kwargs)
    assert observed and all(value is None for value in observed)
    assert unpad._ACTIVE.get() is None and unpad._LABELS.get() is None


def test_instance_dropout_only_and_exact_padded_rng(runtime):
    torch, _, device = runtime
    selected, other = torch.nn.Dropout(0.05).train(), torch.nn.Dropout(0.05).train()
    original_class = torch.nn.Dropout.forward
    unpad._bind_dropout(selected)
    first = selected.forward
    unpad._bind_dropout(selected)
    assert selected.forward is first
    assert torch.nn.Dropout.forward is original_class and "forward" not in other.__dict__
    indices = torch.tensor([0, 1, 2, 3], device=device)
    layout = unpad.PackedBatch(2, 3, 64, (3, 1), indices, frozenset({id(selected)}), frozenset())
    # Actual row-major indices for lengths (3,1) are 0,1,2,3.
    x = torch.rand(2, 3, 64, device=device) + 1
    initial = _rng(torch, device)
    expected = original_class(selected, x)
    expected_rng = _rng(torch, device)
    _set_rng(torch, device, initial)
    with unpad._packing_scope(layout):
        actual = selected(unpad._pack(x, layout))
    torch.testing.assert_close(actual, unpad._pack(expected, layout), rtol=0, atol=0)
    assert torch.equal(_rng(torch, device), expected_rng)


@pytest.mark.parametrize("denominator", [None, 17, "tensor"])
@pytest.mark.parametrize("all_ignored", [False, True])
def test_packed_fused_head_targets_metrics_and_gradients(monkeypatch, denominator, all_ignored):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("fused CE requires CUDA")
    pytest.importorskip("triton")
    pytest.importorskip("soup_cli")
    from solution.autograd import fused_head
    from solution.kernels import cross_entropy as ce

    torch.manual_seed(27027)
    lengths, sequence, width = (7, 3, 1), 7, 32
    _, labels, mask = _batch(torch, "cuda", lengths, sequence)
    # Nonignored FIRST labels expose a mistaken shift across packed row
    # boundaries: none is a target of the preceding row's last real token.
    labels[:, 0] = 5
    if all_ignored:
        labels.fill_(-100)
    indices = torch.tensor([row * sequence + col for row, length in enumerate(lengths)
                            for col in range(length)], device="cuda")
    layout = unpad.PackedBatch(3, sequence, width, lengths, indices, frozenset(), frozenset())
    full = torch.randn(3, sequence, width, device="cuda").masked_fill(~mask.bool().unsqueeze(-1), 0)
    reference = full.detach().clone().requires_grad_()
    packed = unpad._pack(full, layout).detach().requires_grad_()
    padded = unpad._scatter(packed, layout)
    payload = unpad.PackedHead(packed, layout, labels, padded)
    weight = torch.randn(127, width, device="cuda") / width**0.5
    divisor = torch.tensor(17, device="cuda") if denominator == "tensor" else denominator
    rows, seen_targets = [], []
    native = ce.fused_linear_cross_entropy
    def observed(hidden, weight, targets, ignore_index=-100):
        rows.append(hidden.shape[0])
        seen_targets.append(targets.detach().clone())
        return native(hidden, weight, targets, ignore_index=ignore_index)
    monkeypatch.setattr(ce, "fused_linear_cross_entropy", observed)
    expected_loss, expected = fused_head(reference, weight, labels, divisor)
    actual_loss, actual = fused_head(padded, weight, labels, divisor, packed_head=payload)
    assert rows == [3 * sequence, sum(lengths)]
    shifted = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
    assert torch.equal(seen_targets[-1], shifted.reshape(-1).index_select(0, indices))
    last = 0
    for length in lengths:
        last += length
        assert seen_targets[-1][last - 1] == -100
    tolerance = dict(rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(actual_loss, expected_loss, equal_nan=True, **tolerance)
    torch.testing.assert_close(actual.token_loss, expected.token_loss, **tolerance)
    torch.testing.assert_close(actual.token_entropy[mask.bool()],
                               expected.token_entropy[mask.bool()], **tolerance)
    assert torch.count_nonzero(actual.token_entropy[~mask.bool()]) == 0
    assert torch.count_nonzero(actual.token_loss[~mask.bool()]) == 0
    torch.testing.assert_close((actual.token_entropy * mask).sum() / mask.sum(),
                               (expected.token_entropy * mask).sum() / mask.sum(), **tolerance)
    assert actual.correct == expected.correct and actual.labelled == expected.labelled
    expected_loss.mul(128).backward()
    actual_loss.mul(128).backward()
    assert torch.isfinite(packed.grad).all()
    torch.testing.assert_close(packed.grad, unpad._pack(reference.grad, layout), rtol=1e-2, atol=2e-3)
    assert torch.count_nonzero(reference.grad[~mask.bool()]) == 0
    # Stale output identity must fall back to the complete current head.
    other = padded.detach().clone().requires_grad_()
    fallback_loss, fallback = fused_head(other, weight, labels, divisor, packed_head=payload)
    assert rows[-1] == labels.numel()
    torch.testing.assert_close(fallback_loss, expected_loss.detach(), equal_nan=True, **tolerance)
    torch.testing.assert_close(fallback.token_entropy, expected.token_entropy, **tolerance)


def test_scatter_has_one_destination_and_preserves_gradient(runtime, monkeypatch):
    torch, _, device = runtime
    lengths, sequence = (1, 4, 2), 4
    indices = unpad._indices_from_lengths(lengths, sequence, device)
    assert indices.tolist() == [0, 4, 5, 6, 7, 8, 9]
    batch = unpad.PackedBatch(3, sequence, 64, lengths, indices, frozenset(), frozenset())
    x = torch.randn(1, sum(lengths), 64, device=device, requires_grad=True)
    copies, allocations = [], []
    native_zeros, native_copy = torch.zeros, torch.Tensor.index_copy_
    def zeros(*args, **kwargs):
        out = native_zeros(*args, **kwargs)
        allocations.append(out.data_ptr())
        return out
    def copy(destination, *args, **kwargs):
        copies.append(destination.data_ptr())
        return native_copy(destination, *args, **kwargs)
    monkeypatch.setattr(torch, "zeros", zeros)
    monkeypatch.setattr(torch.Tensor, "index_copy_", copy)
    result = unpad._scatter(x, batch)
    assert copies == allocations == [result.data_ptr()]
    weights = torch.arange(result.numel(), device=device, dtype=result.dtype).reshape(result.shape)
    (result * weights).sum().backward()
    torch.testing.assert_close(x.grad, unpad._pack(weights, batch), rtol=0, atol=0)


def _run_joint_acceptance(backend):
    """Called in a fresh GPU subprocess: no dependence on earlier test patches."""
    from collections import defaultdict
    import os
    import torch
    import solution
    import pytest

    os.environ.pop("SOUP_FUSED_DISABLE", None)
    os.environ["SOUP_FUSED_NORM_FP16"] = "1"
    os.environ.pop("SOUP_MLP_RECOMPUTE", None)
    os.environ.pop("SOUP_MLP_OFFLOAD", None)
    os.environ.pop("SOUP_MLP_RECOMPUTE_LAYERS", None)
    solution.apply()
    solution.apply()
    from transformers.models.qwen2 import modeling_qwen2 as module
    from trl import SFTTrainer
    from soup_cli.trainer.rewind_hf import make_rewind_trainer_class
    from soup_cli.utils.stream_loss_chunk import STEP_ATTR
    from solution.rope import _MARKER as rope_marker
    from solution.kernels import cross_entropy as ce, rmsnorm, rope

    assert hasattr(module.apply_rotary_pos_emb, rope_marker)
    wrapped, raw = _model(torch, module, "cuda", backend)
    trainer = object.__new__(make_rewind_trainer_class(SFTTrainer))
    trainer.model = wrapped
    trainer.accelerator = SimpleNamespace(gather_for_metrics=lambda x: x, parallelism_config=None)
    trainer.args = SimpleNamespace(use_liger_kernel=False, loss_type="nll",
                                   average_tokens_across_devices=False, n_gpu=1)
    trainer.num_virtual_tokens = 0
    trainer.aux_loss_enabled = False
    trainer.label_smoother = trainer.compute_loss_func = None
    trainer.model_accepts_loss_kwargs = True
    trainer.state = SimpleNamespace(global_step=0)
    records = []
    def fail(exc):
        raise AssertionError("Rewind recorder failed") from exc
    trainer._soup_rewind_state = SimpleNamespace(
        failed=False, on_micro_batch=lambda **kw: records.append(kw), fail=fail)
    ids, labels, mask = _batch(torch, "cuda", (7, 4, 1, 4))
    denominator = (labels[:, 1:] != -100).sum() + 3
    initial = torch.cuda.get_rng_state()
    rows, norm_dtypes, rope_calls = [], [], []
    native_ce, native_norm, native_rope = ce.fused_linear_cross_entropy, rmsnorm.rms_norm, rope.fused_rope
    def observed_ce(hidden, weight, targets, ignore_index=-100):
        rows.append(hidden.shape[0])
        return native_ce(hidden, weight, targets, ignore_index=ignore_index)
    def observed_norm(x, weight, eps, out_dtype=None):
        norm_dtypes.append(out_dtype)
        return native_norm(x, weight, eps, out_dtype)
    def observed_rope(*args, **kwargs):
        rope_calls.append(args[0].shape)
        return native_rope(*args, **kwargs)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ce, "fused_linear_cross_entropy", observed_ce)
        patch.setattr(rmsnorm, "rms_norm", observed_norm)
        patch.setattr(rope, "fused_rope", observed_rope)
        saved = []
        for packed in (False, True):
            os.environ["SOUP_UNPAD"] = "1" if packed else "0"
            torch.cuda.set_rng_state(initial)
            wrapped.zero_grad(set_to_none=True)
            records.clear()
            norm_dtypes.clear()
            rope_calls.clear()
            trainer._metrics = {mode: defaultdict(list) for mode in ("train", "eval")}
            trainer._total_train_tokens = 0
            with torch.autocast("cuda", dtype=torch.float16):
                loss, outputs = trainer.compute_loss(
                    wrapped, dict(input_ids=ids, labels=labels, attention_mask=mask),
                    return_outputs=True, num_items_in_batch=denominator)
            assert outputs.logits is None
            step = getattr(trainer, STEP_ATTR)
            assert step is not None and raw._solution_fused_step is None
            assert torch.float16 in norm_dtypes and len(rope_calls) == 2
            assert len(records) == 1 and step.token_loss.shape == labels.shape
            entropy = trainer._metrics["train"]["entropy"][-1]
            accuracy = trainer._metrics["train"]["mean_token_accuracy"][-1]
            tokens = trainer._metrics["train"]["num_tokens"][:]
            rng = torch.cuda.get_rng_state()
            loss.mul(128).backward()
            grads = {name: p.grad.detach().clone() for name, p in wrapped.named_parameters()
                     if "lora_" in name and p.grad is not None}
            assert grads and all(torch.isfinite(g).all() for g in grads.values())
            saved.append((loss.detach(), step.token_loss.detach().clone(),
                          step.token_entropy.detach().clone(), entropy, accuracy,
                          tokens, records[0].copy(), rng, grads))
        expected, actual = saved
        assert rows == [labels.numel(), sum((7, 4, 1, 4))]
        for index in (0, 1):
            torch.testing.assert_close(actual[index], expected[index], rtol=3e-3, atol=3e-3)
        torch.testing.assert_close(actual[2][mask.bool()], expected[2][mask.bool()], rtol=3e-3, atol=3e-3)
        assert actual[3] == pytest.approx(expected[3], rel=3e-3, abs=3e-3)
        assert actual[4:6] == expected[4:6]
        assert actual[5] == [mask.sum().item()]
        assert actual[6]["row_tokens"] == expected[6]["row_tokens"]
        torch.testing.assert_close(torch.tensor(actual[6]["row_loss"]),
                                   torch.tensor(expected[6]["row_loss"]), rtol=3e-3, atol=3e-3)
        assert torch.equal(actual[7], expected[7])
        assert actual[8].keys() == expected[8].keys()
        for name, grad in actual[8].items():
            reference = expected[8][name]
            torch.testing.assert_close(grad, reference, rtol=3e-2, atol=1e-3, msg=name)
            assert (grad - reference).norm() <= 0.03 * reference.norm() + 2e-6, name
    assert unpad._ACTIVE.get() is None and unpad._HEAD_CAPTURE.get() is None
    assert unpad._LABELS.get() is None
    print(f"joint {backend}: RoPE + fp16 norm + packed CE + TRL/Rewind + LoRA gradients/RNG PASS")


@pytest.mark.parametrize("backend", ["dense", "bnb"])
def test_joint_patches_and_bridge_in_fresh_process(backend):
    import os
    from pathlib import Path
    import subprocess

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("joint fused-kernel acceptance needs CUDA")
    for dependency in ("transformers", "peft", "trl", "soup_cli", "triton"):
        pytest.importorskip(dependency)
    if backend == "bnb":
        pytest.importorskip("bitsandbytes")
    code = ("from solution.tests.test_unpad import _run_joint_acceptance\n"
            f"_run_joint_acceptance({backend!r})\n")
    result = subprocess.run([sys.executable, "-c", code],
                            cwd=Path(__file__).resolve().parents[2],
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                            capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    print(result.stdout)
