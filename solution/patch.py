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
from types import ModuleType

from .utils import enabled, head_eligible, qwen_model, supported_tensor

_DISABLE_HEAD = ContextVar("soup_fused_disable_head", default=False)
_MODEL_STEP = "_solution_fused_step"
_MARKER = "_solution_fused_original"


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
        eligible = (type(self) is cls and not _DISABLE_HEAD.get()
                    and head_eligible(self, labels, logits_to_keep, **kwargs))
        if not eligible:
            return original(self, labels=labels, logits_to_keep=logits_to_keep, **arguments)
        outputs = self.model(**arguments)
        hidden = outputs.last_hidden_state
        # Unsupported decoder layouts still use its already-computed result.
        if (not supported_tensor(hidden) or hidden.shape[:2] != labels.shape
                or hidden.device != self.lm_head.weight.device
                or hidden.shape[-1] != self.lm_head.weight.shape[1]
                or (not torch.is_autocast_enabled("cuda")
                    and hidden.dtype != self.lm_head.weight.dtype)):
            logits = self.lm_head(hidden)
            loss = self.loss_function(logits=logits, labels=labels,
                                      vocab_size=self.config.vocab_size, **kwargs)
        else:
            loss, step = fused_head(hidden, self.lm_head.weight, labels,
                                    kwargs.get("num_items_in_batch"))
            setattr(self, _MODEL_STEP, step)
            logits = None
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
        use_bridge = (
            qwen is not None and self.model.training
            and not getattr(self, "num_virtual_tokens", 0)
            and not getattr(self.args, "use_liger_kernel", False)
            and getattr(self.args, "loss_type", "nll") == "nll"
            and not getattr(self, "aux_loss_enabled", False)
            and getattr(self, "label_smoother", None) is None
            and getattr(self, "compute_loss_func", None) is None
            and getattr(self.accelerator, "parallelism_config", None) is None
            and head_eligible(qwen, **inputs)
        )
        if not use_bridge:
            token = _DISABLE_HEAD.set(True)
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
