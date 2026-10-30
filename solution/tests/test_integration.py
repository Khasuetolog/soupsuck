"""CPU-safe registration and GPU end-to-end head/metrics/Rewind contracts."""

from collections import defaultdict
import copy
import sys
from types import SimpleNamespace

import pytest


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
    wrapped = peft.get_peft_model(base, peft.LoraConfig(
        task_type="CAUSAL_LM", r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
    ))
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
