"""Qwen2 head/RMSNorm and the TRL metrics bridge.

Pinned source anchors (paths relative to the installed site-packages):
  transformers/models/qwen2/modeling_qwen2.py:452-477 (decoder/head boundary)
  transformers/trainer.py:1974-2061 (loss scaling and num_items_in_batch)
  trl/trainer/sft_trainer.py:1200-1328 (metrics)
  soup_cli/trainer/rewind_hf.py:338-354 (super call then STEP_ATTR)
The dynamic Rewind subclass resolves super().compute_loss on the patched TRL
class, so no setup wrapper or dynamically generated class needs replacing.
"""

from contextvars import ContextVar
from functools import wraps
import importlib
import sys
from types import ModuleType

from .utils import (enabled, head_ineligibility, parallelism_ineligibility,
                    qwen_model, supported_tensor, tensor_ineligibility)

_DISABLE_HEAD = ContextVar("soup_fused_disable_head", default=False)
_MODEL_STEP = "_solution_fused_step"
_MARKER = "_solution_fused_original"
_DIAGNOSTICS = set()


def _report_head(reason=None):
    """One refusal and one successful activation per process, independent of measurement."""
    key = "disabled" if reason is not None else "active"
    if key not in _DIAGNOSTICS:
        _DIAGNOSTICS.add(key)
        message = f"disabled: {reason}" if reason is not None else "active"
        print(f"[solution] fused head {message}", file=sys.stderr, flush=True)


def _bridge_ineligibility(trainer, qwen, inputs):
    if qwen is None:
        return "could not unwrap an exact Qwen2ForCausalLM"
    if not trainer.model.training:
        return "trainer.model.training=False"
    if getattr(trainer, "num_virtual_tokens", 0):
        return f"trainer.num_virtual_tokens={trainer.num_virtual_tokens!r}"
    if getattr(trainer.args, "use_liger_kernel", False):
        return "trainer.args.use_liger_kernel=True"
    if getattr(trainer.args, "loss_type", "nll") != "nll":
        return f"trainer.args.loss_type={trainer.args.loss_type!r}"
    if getattr(trainer, "aux_loss_enabled", False):
        return "trainer.aux_loss_enabled=True"
    if getattr(trainer, "label_smoother", None) is not None:
        return "trainer.label_smoother is configured"
    if getattr(trainer, "compute_loss_func", None) is not None:
        return "trainer.compute_loss_func is configured"
    reason = parallelism_ineligibility(getattr(trainer.accelerator, "parallelism_config", None))
    if reason is not None:
        return reason
    # Trainer is outside Accelerate's model.forward autocast boundary. Only
    # structural conditions belong here; forward rechecks actual precision.
    return head_ineligibility(qwen, _check_precision=False, **inputs)


def _package(name):
    try:
        module = importlib.import_module(name)
    except ImportError:
        return None
    return module if isinstance(module, ModuleType) and hasattr(module, "__path__") else None


def _mark(wrapper, original):
    setattr(wrapper, _MARKER, original)
    return wrapper


def _patch_head(cls):
    original = cls.forward
    if hasattr(original, _MARKER):
        return

    @wraps(original)
    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=None, logits_to_keep=0, **kwargs):
        import torch
        from transformers.modeling_outputs import CausalLMOutputWithPast
        from .autograd import fused_head

        setattr(self, _MODEL_STEP, None)
        arguments = dict(input_ids=input_ids, attention_mask=attention_mask,
                         position_ids=position_ids, past_key_values=past_key_values,
                         inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        if type(self) is not cls:
            reason = f"model type is {type(self).__name__} (requires exact Qwen2ForCausalLM)"
        elif _DISABLE_HEAD.get():
            reason = str(_DISABLE_HEAD.get())
        else:
            reason = head_ineligibility(self, labels, logits_to_keep, **kwargs)
        if reason is not None:
            if self.training:
                _report_head(f"head: {reason}")
            return original(self, labels=labels, logits_to_keep=logits_to_keep, **arguments)
        outputs = self.model(**arguments)
        hidden = outputs.last_hidden_state
        # Unsupported decoder layouts still use its already-computed result.
        reason = tensor_ineligibility(hidden, "hidden")
        if reason is None:
            if hidden.shape[:2] != labels.shape:
                reason = f"hidden shape={tuple(hidden.shape)} does not align with labels={tuple(labels.shape)}"
            elif hidden.device != self.lm_head.weight.device:
                reason = "hidden and lm_head.weight are on different devices"
            elif hidden.shape[-1] != self.lm_head.weight.shape[1]:
                reason = "hidden width differs from lm_head.weight width"
            elif not torch.is_autocast_enabled("cuda") and hidden.dtype != self.lm_head.weight.dtype:
                reason = "hidden and lm_head.weight dtypes differ outside CUDA autocast"
        if reason is not None:
            _report_head(f"head: {reason}")
            logits = self.lm_head(hidden)
            loss = self.loss_function(logits=logits, labels=labels,
                                      vocab_size=self.config.vocab_size, **kwargs)
        else:
            loss, step = fused_head(hidden, self.lm_head.weight, labels,
                                    kwargs.get("num_items_in_batch"))
            setattr(self, _MODEL_STEP, step)
            logits = None
            _report_head()
        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        )

    cls.forward = _mark(forward, original)


def _patch_norm(cls):
    original = cls.forward
    if hasattr(original, _MARKER):
        return

    @wraps(original)
    def forward(self, hidden_states):
        if (enabled() and supported_tensor(hidden_states) and supported_tensor(self.weight)
                and hidden_states.device == self.weight.device
                and hidden_states.shape[-1] == self.weight.numel()):
            from .kernels.rmsnorm import rms_norm

            return rms_norm(hidden_states, self.weight, self.variance_epsilon)
        return original(self, hidden_states)

    cls.forward = _mark(forward, original)


def _log_metrics(trainer, inputs, step):
    """TRL 0.29.1 formulas, including distinct entropy/accuracy/token masks."""
    import torch

    mode = "train" if trainer.model.training else "eval"
    gather = trainer.accelerator.gather_for_metrics
    with torch.no_grad():
        if "attention_mask" in inputs:
            mask = inputs["attention_mask"]
            entropy = (step.token_entropy * mask).sum() / mask.sum()
        elif "position_ids" in inputs:
            entropy = step.token_entropy.mean()
        else:
            raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
        trainer._metrics[mode]["entropy"].append(gather(entropy).mean().item())
        if mode == "train":
            if "attention_mask" in inputs:
                tokens = inputs["attention_mask"].sum()
            else:
                pos = inputs["position_ids"]
                tokens = torch.tensor(pos.size(1), device=pos.device)
            trainer._total_train_tokens += gather(tokens).sum().item()
        trainer._metrics[mode]["num_tokens"] = [trainer._total_train_tokens]
        correct = gather(step.correct)
        total = gather(step.labelled).sum()
        accuracy = (correct.sum() / total).item() if total > 0 else 0.0
        trainer._metrics[mode]["mean_token_accuracy"].append(accuracy)


def _step_from_logits(logits, labels):
    """Rare decoder-layout fallback after the bridge has delegated to HF."""
    import torch
    from torch.nn import functional as F
    from trl.trainer.utils import entropy_from_logits
    from soup_cli.utils.stream_loss_chunk import ChunkedStep

    with torch.no_grad():
        targets = F.pad(labels, (0, 1), value=-100)[..., 1:]
        losses = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                 targets.reshape(-1), reduction="none").reshape(labels.shape)
        mask = labels[:, 1:] != -100
        correct = ((logits[:, :-1].argmax(-1) == labels[:, 1:]) & mask).sum()
        return ChunkedStep(entropy_from_logits(logits), losses, correct, mask.sum())


def _patch_trainer(cls, qwen_cls, trainer_cls, step_attr):
    original = cls.compute_loss
    if hasattr(original, _MARKER):
        return

    @wraps(original)
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        setattr(self, step_attr, None)
        qwen = qwen_model(model, qwen_cls)
        if qwen is not None:
            setattr(qwen, _MODEL_STEP, None)
        reason = _bridge_ineligibility(self, qwen, inputs)
        if reason is not None:
            if self.model.training:
                _report_head(f"bridge: {reason}")
            token = _DISABLE_HEAD.set(f"bridge: {reason}")
            try:
                return original(self, model, inputs, return_outputs=return_outputs,
                                num_items_in_batch=num_items_in_batch)
            finally:
                _DISABLE_HEAD.reset(token)
        inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False
        loss, outputs = trainer_cls.compute_loss(
            self, model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch,
        )
        step = getattr(qwen, _MODEL_STEP, None)
        setattr(qwen, _MODEL_STEP, None)
        if step is None:
            if outputs.logits is None:
                raise RuntimeError("Fused Qwen forward returned neither logits nor fresh statistics")
            step = _step_from_logits(outputs.logits, inputs["labels"])
        setattr(self, step_attr, step)
        _log_metrics(self, inputs, step)
        return (loss, outputs) if return_outputs else loss

    cls.compute_loss = _mark(compute_loss, original)


def install():
    # Fake SimpleNamespaces in Soup tests may lack __spec__ and package paths.
    if _package("transformers") is None:
        return
    try:
        from transformers.models.qwen2 import modeling_qwen2 as qwen
        from transformers import Trainer
        from trl import SFTTrainer
        from soup_cli.utils.stream_loss_chunk import STEP_ATTR
    except (ImportError, AttributeError):
        return
    if not all(isinstance(c, type) for c in
               (qwen.Qwen2ForCausalLM, qwen.Qwen2RMSNorm, SFTTrainer, Trainer)):
        return
    _patch_trainer(SFTTrainer, qwen.Qwen2ForCausalLM, Trainer, STEP_ATTR)
    _patch_head(qwen.Qwen2ForCausalLM)
    _patch_norm(qwen.Qwen2RMSNorm)
    # Preserve the supplied example, guarding its registration externally.
    mlp = qwen.Qwen2MLP
    if not hasattr(mlp.forward, _MARKER):
        try:
            from .swiglu import patch_qwen2_mlp
        except ImportError:
            return  # e.g. CPU-only torch without Triton
        original = mlp.forward
        patch_qwen2_mlp()
        _mark(mlp.forward, original)
