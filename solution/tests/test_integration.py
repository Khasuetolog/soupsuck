"""CPU-safe registration and GPU end-to-end head/metrics/Rewind contracts."""

from collections import defaultdict
import copy
import sys
import importlib
from pathlib import Path
import subprocess
from types import ModuleType, SimpleNamespace

import pytest


def test_cold_apply_does_not_import_training_dependencies():
    code = """
import sys
import solution
solution.apply()
solution.apply()
assert not any(name in sys.modules for name in
               ('torch', 'triton', 'transformers', 'trl', 'peft', 'soup_cli'))
"""
    subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2], check=True)


def test_import_hook_patches_before_import_returns(monkeypatch, tmp_path):
    import solution.patch as patch

    name = "solution_lazy_fixture"
    (tmp_path / f"{name}.py").write_text("class Example:\n    value = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(patch, "_TARGET_MODULES", {name})
    seen = []
    def register(fullname, module):
        if not getattr(module.Example, "patched", False):
            module.Example.value = 2
            module.Example.patched = True
            seen.append(fullname)
    monkeypatch.setattr(patch, "_register_module", register)
    patch.install()
    module = importlib.import_module(name)
    try:
        assert module.Example.value == 2 and seen == [name]
        patch.install()
        assert seen == [name]
        importlib.reload(module)
        assert module.Example.value == 2 and seen == [name, name]
    finally:
        sys.modules.pop(name, None)


def test_lazy_mlp_registers_own_implementation_without_example(monkeypatch):
    import solution.patch as patch

    class MLP:
        def forward(self, x):
            return "original"
    original = MLP.forward
    fake = ModuleType("solution.mlp")
    calls = []
    def install_fused():
        assert MLP.forward is original  # trampoline marker must not block it
        def forward(self, x):
            return "own-fused"
        setattr(forward, patch._MARKER, original)
        MLP.forward = forward
        calls.append("own-fused")
    fake.patch_qwen2_mlp_fused = install_fused
    monkeypatch.setitem(sys.modules, "solution.mlp", fake)
    before = sys.modules.get("solution.swiglu")
    patch._patch_lazy_mlp(MLP)
    trampoline = MLP.forward
    patch._patch_lazy_mlp(MLP)
    assert MLP.forward is trampoline
    assert MLP().forward(SimpleNamespace(is_cuda=False)) == "original"
    assert MLP().forward(SimpleNamespace(is_cuda=True)) == "own-fused"
    patch._patch_lazy_mlp(MLP)
    assert MLP().forward(SimpleNamespace(is_cuda=True)) == "own-fused"
    assert calls == ["own-fused"] and sys.modules.get("solution.swiglu") is before


def test_trl_import_callback_finds_hf_trainer_through_base_trainer(monkeypatch):
    import solution.patch as patch

    trainer = type("Trainer", (), {"__module__": "transformers.trainer"})
    base = type("BaseTrainer", (trainer,), {"__module__": "trl.trainer.base_trainer"})
    def original(*args, **kwargs):
        return "original"
    cls = type("SFTTrainer", (base,), {"__module__": patch._TRL_MODULE, "compute_loss": original})
    root = ModuleType("trl")
    root.__path__ = []
    module = ModuleType(patch._TRL_MODULE)
    module.SFTTrainer, module.BaseTrainer = cls, base
    monkeypatch.setitem(sys.modules, "trl", root)
    patch._register_module(patch._TRL_MODULE, module)
    patched = cls.compute_loss
    assert getattr(patched, patch._MARKER) is original
    patch._register_module(patch._TRL_MODULE, module)
    assert cls.compute_loss is patched


def test_half_norm_scope_is_owned_and_restored():
    from solution.patch import _HALF_NORM, _half_norm_scope

    first, second = object(), object()
    assert _HALF_NORM.get() is None
    with _half_norm_scope(first):
        assert _HALF_NORM.get() is first
        with _half_norm_scope(second):
            assert _HALF_NORM.get() is second
        assert _HALF_NORM.get() is first
        with pytest.raises(ValueError), _half_norm_scope(second):
            raise ValueError("restore on failure")
        assert _HALF_NORM.get() is first
    assert _HALF_NORM.get() is None


def test_post_norm_guard_rejects_dense_fp32_without_autocast(monkeypatch):
    """CPU metadata regression for REVIEW P1-2; no torch installation needed."""
    import solution.patch as patch

    class Module:
        def __init__(self):
            self._forward_hooks = {}
            self._forward_pre_hooks = {}
            self._backward_hooks = {}
            self._backward_pre_hooks = {}
    class Linear(Module):
        def __init__(self):
            super().__init__()
            self.weight = SimpleNamespace(device="cuda:0")
    precision = {"amp": False}
    fake = SimpleNamespace(nn=SimpleNamespace(Module=Module, Linear=Linear), float16="fp16",
                           is_autocast_enabled=lambda device: precision["amp"],
                           get_autocast_dtype=lambda device: "fp16")
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(patch, "supported_tensor", lambda tensor: True)
    monkeypatch.setattr(patch, "_half_norm_enabled", lambda norm: True)
    cls = type("Qwen2MLP", (Module,), {"__module__": patch._QWEN_MODULE})
    mlp = cls()
    mlp.gate_proj, mlp.up_proj, mlp.down_proj = Linear(), Linear(), Linear()
    layer = SimpleNamespace(post_attention_layernorm=object(), mlp=mlp)
    x = SimpleNamespace(device="cuda:0")
    assert not patch._post_norm_half_ready(layer, x)
    precision["amp"] = True
    assert patch._post_norm_half_ready(layer, x)
    mlp.gate_proj._forward_pre_hooks["unknown"] = object()
    assert not patch._post_norm_half_ready(layer, x)
    precision["amp"] = False
    class Linear4bit(Module):
        __module__ = "bitsandbytes.nn.modules"
        def __init__(self):
            super().__init__()
            self.compute_dtype, self.compute_type_is_set = "fp16", True
            self.weight = SimpleNamespace(device="cuda:0", quant_state=object(), requires_grad=False)
        def parameters(self):
            return (self.weight,)
    mlp.gate_proj, mlp.up_proj, mlp.down_proj = Linear4bit(), Linear4bit(), Linear4bit()
    assert patch._post_norm_half_ready(layer, x)  # bnb itself casts without AMP
    mlp.down_proj.compute_dtype = "fp32"
    assert not patch._post_norm_half_ready(layer, x)
    assert not patch._input_norm_half_ready(
        SimpleNamespace(input_layernorm=object(), self_attn=object()), x, {})
    assert not patch._final_norm_half_ready(
        SimpleNamespace(model=SimpleNamespace(norm=object())), {})


def test_nf4_registration_waits_for_both_modules_and_root(monkeypatch):
    import solution.patch as patch

    root = ModuleType("bitsandbytes")
    root.__path__ = []
    root.__spec__ = SimpleNamespace(_initializing=True)
    functional = ModuleType("bitsandbytes.functional")
    backend = ModuleType("bitsandbytes.backends.cuda.ops")
    functional.dequantize_4bit = lambda: None
    backend._dequantize_4bit_impl = lambda: None
    monkeypatch.setitem(sys.modules, "bitsandbytes", root)
    monkeypatch.setitem(sys.modules, functional.__name__, functional)
    monkeypatch.delitem(sys.modules, backend.__name__, raising=False)
    fake = ModuleType("solution.nf4_patch")
    calls = []
    def install():
        if not hasattr(functional.dequantize_4bit, patch._MARKER):
            setattr(functional.dequantize_4bit, patch._MARKER, True)
            setattr(backend._dequantize_4bit_impl, patch._MARKER, True)
            calls.append("installed")
    fake.patch_bnb_dequant = install
    monkeypatch.setitem(sys.modules, "solution.nf4_patch", fake)
    patch._register_module(functional.__name__, functional)
    assert not calls
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    patch._register_module(backend.__name__, backend)
    assert not calls  # root is still initializing
    patch._register_module(root.__name__, root)
    assert calls == ["installed"]
    root.__spec__._initializing = False
    patch._register_module(functional.__name__, functional)
    assert calls == ["installed"]


def test_apply_with_fake_modules(monkeypatch):
    # Mirrors tests/test_trainer_init.py: these are not importable packages.
    import solution

    fake = SimpleNamespace(AutoModelForCausalLM=object, AutoTokenizer=object)
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(get_peft_model=lambda m, c: m))
    solution.apply()
    solution.apply()
    assert sys.modules["transformers"] is fake


def test_head_diagnostics_are_once_per_process(monkeypatch, capsys):
    import solution.patch as patch

    monkeypatch.setattr(patch, "_DIAGNOSTICS", set())
    patch._report_head("bridge: trainer.compute_loss_func is configured")
    patch._report_head("head: another refusal")
    patch._report_head()
    patch._report_head()
    assert capsys.readouterr().err.splitlines() == [
        "[solution] fused head disabled: bridge: trainer.compute_loss_func is configured",
        "[solution] fused head active",
    ]


@pytest.mark.parametrize("axis", [None, "dp_replicate_size", "dp_shard_size", "tp_size", "cp_size", "sp_size"])
def test_parallelism_guard_checks_active_sizes(axis):
    from solution.utils import parallelism_ineligibility

    config = SimpleNamespace(dp_replicate_size=1, dp_shard_size=1, tp_size=1,
                             cp_size=1, sp_size=1)
    if axis:
        setattr(config, axis, 2)
        assert axis in parallelism_ineligibility(config)
    else:
        assert parallelism_ineligibility(config) is None
        assert parallelism_ineligibility(None) is None
    assert parallelism_ineligibility(SimpleNamespace()) is not None


@pytest.fixture
def runtime():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("transformers")
    pytest.importorskip("trl")
    pytest.importorskip("soup_cli")
    pytest.importorskip("triton")
    from transformers.models.qwen2 import modeling_qwen2 as qwen
    from trl import SFTTrainer
    from soup_cli.trainer.rewind_hf import make_rewind_trainer_class
    import solution

    # Deliberately create the dynamic subclass BEFORE installation as well.
    rewind_cls = make_rewind_trainer_class(SFTTrainer)
    solution.apply()
    forwards = (qwen.Qwen2ForCausalLM.forward, qwen.Qwen2RMSNorm.forward,
                qwen.Qwen2MLP.forward, SFTTrainer.compute_loss)
    solution.apply()
    assert forwards == (qwen.Qwen2ForCausalLM.forward, qwen.Qwen2RMSNorm.forward,
                        qwen.Qwen2MLP.forward, SFTTrainer.compute_loss)
    return torch, qwen, SFTTrainer, rewind_cls


def _model(torch, qwen, device="cuda"):
    from transformers import Qwen2Config

    torch.manual_seed(71)
    config = Qwen2Config(vocab_size=1009, hidden_size=80, intermediate_size=176,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, attention_dropout=0.0,
                        tie_word_embeddings=False, use_cache=False)
    config._attn_implementation = "eager"
    model = qwen.Qwen2ForCausalLM(config).to(device).train()
    model.lm_head.weight.requires_grad_(False)
    return model


def _trainer(cls, model):
    # Bypass costly dataset/accelerator initialization; execute the actual
    # inherited compute_loss chain, including transformers.Trainer.
    trainer = object.__new__(cls)
    trainer.model = model
    trainer.accelerator = SimpleNamespace(
        gather_for_metrics=lambda t: t, parallelism_config=None,
    )
    trainer.args = SimpleNamespace(use_liger_kernel=False, loss_type="nll",
                                   average_tokens_across_devices=False, n_gpu=1)
    trainer.num_virtual_tokens = 0
    trainer.aux_loss_enabled = False
    trainer.label_smoother = None
    trainer.compute_loss_func = None
    trainer.model_accepts_loss_kwargs = True
    trainer._metrics = {m: defaultdict(list) for m in ("train", "eval")}
    trainer._total_train_tokens = 0
    return trainer


def _batch(torch):
    ids = torch.randint(0, 1009, (2, 9), device="cuda")
    labels = ids.clone()
    labels[:, :3] = -100  # prompt mask is distinct from padding mask
    labels[1, 7:] = -100
    mask = torch.ones_like(ids)
    mask[1, 7:] = 0
    return ids, labels, mask


@pytest.mark.parametrize("explicit_denominator", [False, True])
def test_loss_gradients_metrics_and_rewind(runtime, monkeypatch, explicit_denominator):
    torch, qwen, sft, rewind_cls = runtime
    from soup_cli.utils.stream_loss_chunk import STEP_ATTR
    from soup_cli.trainer.rewind_hf import row_losses
    from trl.trainer.utils import entropy_from_logits
    from solution.patch import _MARKER, _MODEL_STEP

    model = _model(torch, qwen)
    reference = copy.deepcopy(model)
    ids, labels, mask = _batch(torch)
    x = model.model.embed_tokens(ids).detach().requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)
    denominator = (labels[:, 1:] != -100).sum() + 3 if explicit_denominator else None
    trainer = _trainer(rewind_cls, model)
    records = []
    def fail(exc):
        raise AssertionError("Rewind failed") from exc
    trainer._soup_rewind_state = SimpleNamespace(
        failed=False, on_micro_batch=lambda **kw: records.append(kw), fail=fail,
    )
    trainer.state = SimpleNamespace(global_step=4)
    with torch.autocast("cuda", dtype=torch.float16):
        loss, output = trainer.compute_loss(
            model, dict(inputs_embeds=x, labels=labels, attention_mask=mask),
            return_outputs=True, num_items_in_batch=denominator,
        )
    assert output.logits is None
    step = getattr(trainer, STEP_ATTR)
    assert step is not None and getattr(model, _MODEL_STEP) is None
    assert step.token_loss.shape == labels.shape
    assert not step.token_loss.requires_grad and not step.token_entropy.requires_grad
    assert torch.count_nonzero(step.token_loss[:, -1]) == 0
    # Use a scaler-sized multiplier to exercise upstream gradients explicitly.
    (loss * 128).backward()
    with monkeypatch.context() as m:
        for cls in (qwen.Qwen2ForCausalLM, qwen.Qwen2RMSNorm, qwen.Qwen2MLP):
            m.setattr(cls, "forward", getattr(cls.forward, _MARKER))
        with torch.autocast("cuda", dtype=torch.float16):
            ref_output = reference(inputs_embeds=x_ref, labels=labels, attention_mask=mask,
                                   num_items_in_batch=denominator)
        (ref_output.loss * 128).backward()
    torch.testing.assert_close(loss, ref_output.loss, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=4e-2, atol=2e-2)
    ref_params = dict(reference.named_parameters())
    for name, param in model.named_parameters():
        other = ref_params[name]
        if param.grad is None or other.grad is None:
            assert param.grad is None and other.grad is None, name
        else:
            torch.testing.assert_close(param.grad, other.grad, rtol=5e-2, atol=2e-2, msg=name)
    logits = ref_output.logits.float()  # Accelerate's fp32 output conversion
    with torch.no_grad():
        entropy = (entropy_from_logits(logits) * mask).sum() / mask.sum()
        valid = labels[:, 1:] != -100
        accuracy = (((logits[:, :-1].argmax(-1) == labels[:, 1:]) & valid).sum() / valid.sum())
        expected_row_loss, expected_row_tokens = row_losses(logits, labels)
    metrics = trainer._metrics["train"]
    assert metrics["entropy"][-1] == pytest.approx(entropy.item(), rel=3e-3, abs=3e-3)
    assert metrics["mean_token_accuracy"][-1] == pytest.approx(accuracy.item(), abs=0.01)
    assert metrics["num_tokens"] == [mask.sum().item()]
    assert len(records) == 1 and records[0]["step"] == 5
    torch.testing.assert_close(torch.as_tensor(records[0]["row_loss"]),
                               torch.as_tensor(expected_row_loss), rtol=3e-3, atol=3e-3)
    assert records[0]["row_tokens"] == expected_row_tokens
    # An unsupported next batch must never leave the preceding fused step.
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        _, fallback = trainer.compute_loss(
            model, dict(input_ids=ids, labels=labels, attention_mask=mask), return_outputs=True,
        )
    assert fallback.logits is not None
    assert getattr(trainer, STEP_ATTR) is None and getattr(model, _MODEL_STEP) is None


def test_forward_fallbacks(runtime):
    torch, qwen, _, _ = runtime
    from solution.patch import _MODEL_STEP

    model = _model(torch, qwen)
    ids, labels, mask = _batch(torch)
    with torch.autocast("cuda", dtype=torch.float16):
        assert model(input_ids=ids, attention_mask=mask).logits is not None
        assert model(input_ids=ids, labels=labels, attention_mask=mask,
                     logits_to_keep=0).logits is None
        model.eval()
        assert model(input_ids=ids, labels=labels, attention_mask=mask).logits is not None
    assert getattr(model, _MODEL_STEP) is None
    cpu = _model(torch, qwen, "cpu")
    assert cpu(input_ids=ids.cpu(), labels=labels.cpu(), attention_mask=mask.cpu()).logits is not None


def test_accelerate_peft_forward_enables_amp_after_trainer_preflight(runtime, monkeypatch, capsys):
    """Regression: real Trainer is outside AMP; Accelerate wraps PEFT.forward.

    transformers/trainer.py:2075-2080 uses a nullcontext, whereas
    accelerate/accelerator.py:1818-1830 enters autocast inside model.forward.
    Outer-autocast tests do not exercise this boundary.
    """
    torch, qwen, _, rewind_cls = runtime
    peft = pytest.importorskip("peft")
    from accelerate import Accelerator
    from soup_cli.utils.stream_loss_chunk import STEP_ATTR
    import solution.patch as patch
    from solution.utils import head_ineligibility, qwen_model

    monkeypatch.setattr(patch, "_DIAGNOSTICS", set())
    base = _model(torch, qwen)
    base.config.tie_word_embeddings = True
    base.tie_weights()
    config = peft.LoraConfig(
        task_type="CAUSAL_LM", r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
    )
    # PEFT model.py:382-400 puts custom mappings before optional quantizer
    # dispatchers (including the incompatible package preinstalled on Kaggle).
    # This changes only this test's config, without importing that package or
    # modifying any global availability probe.
    from peft.tuners.lora.layer import Linear
    config._register_custom_module({torch.nn.Linear: Linear})
    wrapped = peft.get_peft_model(base, config)
    accelerator = Accelerator(mixed_precision="fp16")
    prepared = accelerator.prepare_model(wrapped)
    assert qwen_model(prepared, qwen.Qwen2ForCausalLM) is base
    assert type(base) is qwen.Qwen2ForCausalLM
    assert base.lm_head.weight is base.model.embed_tokens.weight
    assert base.lm_head.weight.dtype == torch.float32 and not base.lm_head.weight.requires_grad
    ids, labels, mask = _batch(torch)
    assert not torch.is_autocast_enabled("cuda")
    assert "autocast is disabled" in head_ineligibility(base, labels)
    # Structural preflight must allow Accelerate to enter its forward context.
    assert head_ineligibility(base, labels, _check_precision=False) is None
    trainer = _trainer(rewind_cls, prepared)
    trainer.accelerator = accelerator
    records = []
    def fail(exc):
        raise AssertionError("Rewind failed") from exc
    trainer._soup_rewind_state = SimpleNamespace(
        failed=False, on_micro_batch=lambda **kw: records.append(kw), fail=fail,
    )
    trainer.state = SimpleNamespace(global_step=0)
    def dense_head_must_not_run(module, inputs):
        raise AssertionError("Dense lm_head ran; the fused head was not activated")
    hook = base.lm_head.register_forward_pre_hook(dense_head_must_not_run)
    try:
        # Deliberately NO outer torch.autocast, matching Trainer.training_step.
        for _ in range(2):
            loss, outputs = trainer.compute_loss(
                prepared, dict(input_ids=ids, labels=labels, attention_mask=mask),
                return_outputs=True, num_items_in_batch=(labels[:, 1:] != -100).sum(),
            )
            assert outputs.logits is None and torch.isfinite(loss)
            assert getattr(trainer, STEP_ATTR) is not None
            assert getattr(base, patch._MODEL_STEP) is None
            (loss * 128).backward()
    finally:
        hook.remove()
    grads = [p.grad for n, p in prepared.named_parameters() if "lora_" in n and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(torch.count_nonzero(g) > 0 for g in grads)
    assert len(records) == 2
    assert len(trainer._metrics["train"]["entropy"]) == 2
    assert len(trainer._metrics["train"]["mean_token_accuracy"]) == 2
    assert trainer._metrics["train"]["num_tokens"] == [2 * mask.sum().item()]
    diagnostic = capsys.readouterr().err
    assert diagnostic.count("[solution] fused head active") == 1
    assert "[solution] fused head disabled" not in diagnostic


def test_bridge_without_amp_preserves_dense_fallback(runtime, monkeypatch, capsys):
    """Skipping outer precision checks must not force fp16 math without AMP."""
    torch, qwen, sft, _ = runtime
    import solution.patch as patch
    from soup_cli.utils.stream_loss_chunk import STEP_ATTR

    monkeypatch.setattr(patch, "_DIAGNOSTICS", set())
    model = _model(torch, qwen)
    trainer = _trainer(sft, model)
    ids, labels, mask = _batch(torch)
    assert not torch.is_autocast_enabled("cuda")
    loss, output = trainer.compute_loss(
        model, dict(input_ids=ids, labels=labels, attention_mask=mask), return_outputs=True,
    )
    assert output.logits is not None and output.logits.dtype == torch.float32
    assert torch.isfinite(loss)
    assert getattr(trainer, STEP_ATTR) is not None
    assert len(trainer._metrics["train"]["entropy"]) == 1
    diagnostic = capsys.readouterr().err
    assert "fused head disabled: head: CUDA autocast is disabled" in diagnostic
    assert "fused head active" not in diagnostic


def test_nf4_hooks_registered_after_bnb_import(runtime):
    pytest.importorskip("bitsandbytes")
    import solution
    from bitsandbytes import functional
    from bitsandbytes.backends.cuda import ops
    from solution.nf4_patch import _MARKER

    solution.apply()
    assert hasattr(functional.dequantize_4bit, _MARKER)
    assert hasattr(ops._dequantize_4bit_impl, _MARKER)
    installed = functional.dequantize_4bit, ops._dequantize_4bit_impl
    solution.apply()
    assert installed == (functional.dequantize_4bit, ops._dequantize_4bit_impl)


@pytest.mark.parametrize("training", [True, False])
def test_dense_qwen_without_amp_keeps_all_norm_outputs_full_precision(runtime, monkeypatch, training):
    torch, qwen, _, _ = runtime
    import solution.kernels.rmsnorm as norms
    import solution.kernels.add_rmsnorm as adds

    model = _model(torch, qwen).train(training)
    ids, labels, mask = _batch(torch)
    requested = []
    normal, additive = norms.rms_norm, adds.add_rms_norm
    def normal_spy(x, weight, eps, out_dtype=None):
        requested.append(out_dtype)
        return normal(x, weight, eps, out_dtype)
    def add_spy(r, d, weight, eps, out_dtype=None):
        requested.append(out_dtype)
        return additive(r, d, weight, eps, out_dtype)
    monkeypatch.setattr(norms, "rms_norm", normal_spy)
    monkeypatch.setattr(adds, "add_rms_norm", add_spy)
    monkeypatch.delenv("SOUP_FUSED_NORM_FP16", raising=False)
    assert not torch.is_autocast_enabled("cuda")
    with torch.no_grad():
        output = model(input_ids=ids, labels=labels, attention_mask=mask)
    assert output.logits.dtype == torch.float32 and torch.isfinite(output.loss)
    assert requested and all(dtype is None for dtype in requested)


def test_input_final_half_norm_with_real_bnb_lora_preserves_dropout(runtime, monkeypatch):
    """Compare the full fp32 norm path with scoped fp16 input/final norms."""
    torch, qwen, _, _ = runtime
    bnb = pytest.importorskip("bitsandbytes")
    peft = pytest.importorskip("peft")
    import solution
    import solution.lora as lora
    import solution.kernels.rmsnorm as norm_module
    from solution.patch import _input_norm_half_ready

    base = _model(torch, qwen)
    # Convert precisely the three consumers of input_layernorm. Marking the
    # model 4-bit makes PEFT select bnb before optional dense quantizer probes.
    for block in base.model.layers:
        for key in ("q_proj", "k_proj", "v_proj"):
            old = getattr(block.self_attn, key)
            quantized = bnb.nn.Linear4bit(old.in_features, old.out_features,
                                         bias=old.bias is not None, compute_dtype=torch.float16,
                                         compress_statistics=True, quant_type="nf4")
            quantized.weight = bnb.nn.Params4bit(old.weight.detach().cpu(), requires_grad=False,
                                                compress_statistics=True, quant_type="nf4")
            if old.bias is not None:
                quantized.bias = torch.nn.Parameter(old.bias.detach().cpu(), requires_grad=False)
            setattr(block.self_attn, key, quantized.cuda())
    base.is_loaded_in_4bit = True
    wrapped = peft.get_peft_model(base, peft.LoraConfig(
        task_type="CAUSAL_LM", r=4, lora_alpha=8, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )).train()
    solution.apply()
    with torch.no_grad():
        for name, param in wrapped.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 0.03)
    ids, labels, mask = _batch(torch)
    masks, requests, ingress = [], [], []
    hooks = []
    for index, block in enumerate(base.model.layers):
        for key in ("q_proj", "v_proj"):
            projection = getattr(block.self_attn, key)
            def capture(module, args, output, label=(index, key)):
                masks.append((label, args[0].dtype, (output.detach() != 0).clone()))
            hooks.append(projection.lora_dropout["default"].register_forward_hook(capture))
    norm_forward = norm_module.rms_norm
    def observed_norm(x, weight, eps, out_dtype=None):
        requests.append((id(weight), out_dtype))
        return norm_forward(x, weight, eps, out_dtype=out_dtype)
    monkeypatch.setattr(norm_module, "rms_norm", observed_norm)
    eligibility = lora._eligible
    def observed_ingress(layer, x, kwargs, **options):
        ok = eligibility(layer, x, kwargs, **options)
        if not options:  # actual forward, rather than prospective norm guard
            ingress.append((x.dtype, ok))
        return ok
    monkeypatch.setattr(lora, "_eligible", observed_ingress)
    initial = torch.cuda.get_rng_state()
    try:
        monkeypatch.setenv("SOUP_FUSED_NORM_FP16", "0")
        with torch.autocast("cuda", dtype=torch.float16):
            expected = wrapped(input_ids=ids, labels=labels, attention_mask=mask).loss
        after_ref = torch.cuda.get_rng_state()
        (expected * 128).backward()
        gradients = {name: p.grad.detach().clone() for name, p in wrapped.named_parameters()
                     if "lora_" in name and p.grad is not None}
        reference_masks = list(masks)
        masks.clear()
        requests.clear()
        ingress.clear()
        wrapped.zero_grad(set_to_none=True)
        torch.cuda.set_rng_state(initial)
        monkeypatch.setenv("SOUP_FUSED_NORM_FP16", "1")
        with torch.autocast("cuda", dtype=torch.float16):
            for block in base.model.layers:
                assert _input_norm_half_ready(block, base.model.embed_tokens(ids), {})
            actual = wrapped(input_ids=ids, labels=labels, attention_mask=mask).loss
        after_actual = torch.cuda.get_rng_state()
        (actual * 128).backward()
        assert torch.equal(after_ref, after_actual)
        assert len(reference_masks) == len(masks) == 2 * len(base.model.layers)
        for ref, got in zip(reference_masks, masks):
            assert ref[:2] == got[:2] and got[1] == torch.float32
            assert torch.equal(ref[2], got[2])
        assert ingress and all(dtype == torch.float16 and ok for dtype, ok in ingress)
        norm_ids = {id(block.input_layernorm.weight) for block in base.model.layers}
        norm_ids.add(id(base.model.norm.weight))
        assert all((key, torch.float16) in requests for key in norm_ids)
        torch.testing.assert_close(actual, expected, rtol=5e-3, atol=3e-3)
        for name, param in wrapped.named_parameters():
            if name not in gradients:
                continue
            ref = gradients[name]
            assert param.grad is not None and torch.isfinite(param.grad).all()
            torch.testing.assert_close(param.grad, ref, rtol=3e-2, atol=5e-4, msg=name)
            assert torch.linalg.vector_norm(param.grad - ref) <= 3e-2 * torch.linalg.vector_norm(ref) + 1e-6, name
        # If LoRA cannot promise fp32 dropout, its input norm MUST stay full
        # precision, so the original PEFT path never sees an unsafe half input.
        first = base.model.layers[0]
        rejected = first.self_attn.q_proj
        monkeypatch.setattr(lora, "_eligible", lambda layer, x, kwargs, **opts:
                            False if layer is rejected else eligibility(layer, x, kwargs, **opts))
        requests.clear()
        masks.clear()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            assert not _input_norm_half_ready(first, base.model.embed_tokens(ids), {})
            wrapped(input_ids=ids, labels=labels, attention_mask=mask)
        assert (id(first.input_layernorm.weight), None) in requests
        assert all(dtype == torch.float32 for _, dtype, _ in masks)
    finally:
        for hook in hooks:
            hook.remove()


def test_empty_accuracy_and_distinct_masks(runtime):
    torch, _, _, _ = runtime
    from solution.patch import _log_metrics
    from soup_cli.utils.stream_loss_chunk import ChunkedStep

    trainer = SimpleNamespace(model=SimpleNamespace(training=True),
                              accelerator=SimpleNamespace(gather_for_metrics=lambda x: x),
                              _metrics={"train": defaultdict(list)}, _total_train_tokens=7)
    entropy = torch.tensor([[1., 2., 3.]], device="cuda")
    zero = torch.tensor(0, device="cuda")
    step = ChunkedStep(entropy, torch.zeros_like(entropy), zero, zero)
    _log_metrics(trainer, {"attention_mask": torch.tensor([[1, 1, 0]], device="cuda")}, step)
    assert trainer._metrics["train"]["entropy"] == [1.5]
    assert trainer._metrics["train"]["mean_token_accuracy"] == [0.0]
    assert trainer._metrics["train"]["num_tokens"] == [9]
