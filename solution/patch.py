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
from importlib.machinery import PathFinder
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
        # TRL can load before Qwen2; do not import the architecture just to
        # register its trainer bridge. It exists by the first model forward.
        actual_cls = qwen_cls
        if actual_cls is None:
            module = sys.modules.get("transformers.models.qwen2.modeling_qwen2")
            actual_cls = getattr(module, "Qwen2ForCausalLM", None)
        qwen = qwen_model(model, actual_cls)
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


def _patch_lazy_mlp(cls):
    """Load the own fused MLP on first CUDA use; never register the example."""
    original = cls.forward
    if hasattr(original, _MARKER):
        return

    @wraps(original)
    def forward(self, x):
        if not x.is_cuda or not enabled() or type(self) is not cls:
            return original(self, x)
        from .mlp import patch_qwen2_mlp_fused

        # Both patches use _MARKER. Restore the original before asking the
        # own MLP installer to register, otherwise it would see our trampoline
        # marker and incorrectly treat the implementation as already installed.
        if cls.forward is forward:
            cls.forward = original
            patch_qwen2_mlp_fused()
        return cls.forward(self, x)

    cls.forward = _mark(forward, original)


_QWEN_MODULE = "transformers.models.qwen2.modeling_qwen2"
_TRL_MODULE = "trl.trainer.sft_trainer"
_BNB_MODULE = "peft.tuners.lora.bnb"
_TARGET_MODULES = {_QWEN_MODULE, _TRL_MODULE, _BNB_MODULE}
_IMPORT_FINDER = None


def _register_module(name, module):
    # Fake regression-test modules need not be packages or expose real types.
    root = sys.modules.get(name.split(".")[0])
    if (not isinstance(module, ModuleType) or not isinstance(root, ModuleType)
            or not hasattr(root, "__path__")):
        return
    if name == _QWEN_MODULE:
        names = ("Qwen2ForCausalLM", "Qwen2RMSNorm", "Qwen2MLP", "Qwen2DecoderLayer")
        if not all(isinstance(getattr(module, key, None), type)
                   and getattr(module, key).__module__ == name for key in names):
            return
        _patch_head(module.Qwen2ForCausalLM)
        _patch_norm(module.Qwen2RMSNorm)
        from .decoder import patch_qwen2_decoder

        patch_qwen2_decoder()
        _patch_lazy_mlp(module.Qwen2MLP)
    elif name == _TRL_MODULE:
        cls = getattr(module, "SFTTrainer", None)
        # TRL 0.29.1 imports BaseTrainer, not Trainer into this module.
        # Resolve the actual HF base from MRO without another eager import.
        trainer = next((base for base in cls.__mro__[1:]
                        if base.__module__ == "transformers.trainer" and base.__name__ == "Trainer"),
                       None) if isinstance(cls, type) else None
        if isinstance(cls, type) and isinstance(trainer, type) and cls.__module__ == name:
            # Pinned Soup stream_loss_chunk.py:298; importing that module at
            # registration time is unnecessary. fused_head imports ChunkedStep
            # only when training actually uses the no-logits head.
            _patch_trainer(cls, None, trainer, "_soup_chunked_step")
    elif name == _BNB_MODULE:
        cls = getattr(module, "Linear4bit", None)
        if isinstance(cls, type) and cls.__module__ == name:
            from .lora import _patch_class

            _patch_class(cls, tuple(module.VARIANT_KWARG_KEYS))


class _PatchLoader:
    def __init__(self, loader, name):
        self.loader, self.name = loader, name

    def create_module(self, spec):
        create = getattr(self.loader, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        self.loader.exec_module(module)
        _register_module(self.name, module)

    def __getattr__(self, name):
        return getattr(self.loader, name)


class _PatchFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGET_MODULES:
            return None
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is not None and hasattr(spec.loader, "exec_module"):
            spec.loader = _PatchLoader(spec.loader, fullname)
        return spec


def install():
    """Register targeted post-import patches, without importing heavy packages.

    Python calls exec_module before returning the imported module to its caller;
    dynamic Soup subclasses therefore see patched TRL methods. Modules imported
    before apply are handled immediately; markers keep both paths idempotent.
    No CUDA kernel is imported or compiled here.
    """
    global _IMPORT_FINDER
    if _IMPORT_FINDER is None:
        _IMPORT_FINDER = _PatchFinder()
        sys.meta_path.insert(0, _IMPORT_FINDER)
    for name in _TARGET_MODULES:
        module = sys.modules.get(name)
        if module is not None:
            _register_module(name, module)
