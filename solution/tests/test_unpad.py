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
            return self.data
        def __eq__(self, value):
            if self.ndim == 2:
                return Tensor([[x == value for x in row] for row in self.data])
            return Tensor([x == value for x in self.data])
        def __getitem__(self, selector):
            return Tensor([x for row, flags in zip(self.data, selector.data)
                           for x, flag in zip(row, flags) if flag])
        def all(self):
            return all(self.data)
    fake = SimpleNamespace(Tensor=Tensor, tensor=Tensor, long="int64")
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
    mask = Tensor([[1, 1, 1], [1, 0, 0]])
    ids = Tensor([[5, 6, 7], [8, 0, 0]])
    labels = Tensor([[-100, 6, 7], [-100, -100, -100]])
    token = unpad._LABELS.set((labels, mask))
    try:
        plan = unpad._plan(model, module, ids, mask, None, None, None, False, {})
        assert plan.lengths == (3, 1) and plan.indices.data == [0, 1, 2, 3]
        labels.data[1][2] = 9
        assert unpad._plan(model, module, ids, mask, None, None, None, False, {}) is None
    finally:
        unpad._LABELS.reset(token)


@pytest.fixture
def runtime(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
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
