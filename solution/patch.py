"""Qwen2 head/RMSNorm and the TRL metrics bridge.

Pinned source anchors (paths relative to the installed site-packages):
  transformers/models/qwen2/modeling_qwen2.py:452-477 (decoder/head boundary)
  transformers/trainer.py:1974-2061 (loss scaling and num_items_in_batch)
  trl/trainer/sft_trainer.py:1200-1328 (metrics)
  soup_cli/trainer/rewind_hf.py:338-354 (super call then STEP_ATTR)
The dynamic Rewind subclass resolves super().compute_loss on the patched TRL
class, so no setup wrapper or dynamically generated class needs replacing.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from importlib.machinery import PathFinder
import sys
import os
from types import ModuleType

from .utils import (enabled, head_ineligibility, parallelism_ineligibility,
                    qwen_model, supported_tensor, tensor_ineligibility)

_DISABLE_HEAD = ContextVar("soup_fused_disable_head", default=False)
_MODEL_STEP = "_solution_fused_step"
_MARKER = "_solution_fused_original"
_DIAGNOSTICS = set()
_HALF_NORM = ContextVar("soup_half_norm_owner", default=None)


@contextmanager
def _half_norm_scope(norm):
    """Select one explicitly owned norm, never all norms of the same class."""
    token = _HALF_NORM.set(norm)
    try:
        yield
    finally:
        _HALF_NORM.reset(token)


def _plain_half_consumer(module, x):
    import torch

    if (not isinstance(module, torch.nn.Module) or "forward" in module.__dict__
            or module._forward_hooks or module._forward_pre_hooks
            or module._backward_hooks or module._backward_pre_hooks):
        return False
    if type(module) is torch.nn.Linear:
        return (torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.float16
                and supported_tensor(module.weight) and module.weight.device == x.device)
    return (type(module).__module__ == "bitsandbytes.nn.modules"
            and type(module).__name__ == "Linear4bit" and module.compute_dtype == torch.float16
            and module.compute_type_is_set and getattr(module.weight, "quant_state", None) is not None
            and module.weight.device == x.device and not any(p.requires_grad for p in module.parameters()))


def _half_norm_enabled(norm):
    import torch

    return (isinstance(norm, torch.nn.Module) and enabled() and os.environ.get("SOUP_FUSED_NORM_FP16") != "0"
            and not (norm._forward_hooks or norm._forward_pre_hooks
                     or norm._backward_hooks or norm._backward_pre_hooks))


def _input_norm_half_ready(layer, x, kwargs):
    """Qwen2Attention.forward:206-208 has exactly three normalized-input consumers."""
    norm, attention = layer.input_layernorm, layer.self_attn
    import torch

    # Outside AMP, rotary cos/sin can promote q/k back to fp32 while v stays
    # half; eager attention's subsequent fp32@fp16 matmul would be invalid.
    # Require AMP for the entire attention, not only its input projections.
    if (not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.float16
            or not _half_norm_enabled(norm) or not supported_tensor(x)
            or type(attention).__module__ != _QWEN_MODULE or type(attention).__name__ != "Qwen2Attention"
            or "forward" in attention.__dict__ or attention._forward_hooks or attention._forward_pre_hooks
            or attention._backward_hooks or attention._backward_pre_hooks):
        return False
    if kwargs.get("output_hidden_states") or getattr(attention.config, "output_hidden_states", False):
        return False
    from .lora import fp16_ingress_ready

    return all(_plain_half_consumer(proj, x) or fp16_ingress_ready(proj, x)
               for proj in (attention.q_proj, attention.k_proj, attention.v_proj))


def _final_norm_half_ready(model, kwargs):
    """Scope Qwen2Model.norm only when the owning CausalLM consumes it as fp16."""
    norm = getattr(model.model, "norm", None)
    import torch

    if (not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.float16
            or type(model.lm_head) is not torch.nn.Linear):
        return False
    if not _half_norm_enabled(norm) or not supported_tensor(norm.weight):
        return False
    if (model.model._forward_hooks or model.model._forward_pre_hooks
            or model.model._backward_hooks or model.model._backward_pre_hooks):
        return False
    if kwargs.get("output_hidden_states") or getattr(model.config, "output_hidden_states", False):
        return False
    # Plain head under CUDA fp16 autocast, or the fused head that explicitly
    # converts hidden to fp16 (cross_entropy.py:124). Model-only calls are not scoped.
    return _plain_half_consumer(model.lm_head, norm.weight)


def _post_norm_half_ready(layer, x):
    """P1-2: never feed half norm output to fp32 dense MLP without autocast."""
    import torch

    norm, mlp = layer.post_attention_layernorm, layer.mlp
    if (not _half_norm_enabled(norm) or not supported_tensor(x)
            or type(mlp).__module__ != _QWEN_MODULE or type(mlp).__name__ != "Qwen2MLP"
            or "forward" in mlp.__dict__ or mlp._forward_hooks or mlp._forward_pre_hooks
            or mlp._backward_hooks or mlp._backward_pre_hooks):
        return False
    if torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") != torch.float16:
        return False
    projections = (mlp.gate_proj, mlp.up_proj, mlp.down_proj)
    return len({type(proj) for proj in projections}) == 1 and all(
        _plain_half_consumer(proj, x) for proj in projections)


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
        from .unpad import capture_packed_head

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
            norm = self.model.norm if type(self) is cls and _final_norm_half_ready(self, kwargs) else None
            with _half_norm_scope(norm):
                return original(self, labels=labels, logits_to_keep=logits_to_keep, **arguments)
        norm = self.model.norm if _final_norm_half_ready(self, kwargs) else None
        with capture_packed_head(self.model, labels) as captured, _half_norm_scope(norm):
            outputs = self.model(**arguments)
        hidden = outputs.last_hidden_state
        packed_head = captured.take(hidden, labels)
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
                                    kwargs.get("num_items_in_batch"), packed_head=packed_head)
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

            import torch

            out_dtype = torch.float16 if _HALF_NORM.get() is self and _half_norm_enabled(self) else None
            return rms_norm(hidden_states, self.weight, self.variance_epsilon, out_dtype=out_dtype)
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
_SOUP_MODULE = "soup_cli.trainer.sft"
_NF4_MODULES = {"bitsandbytes", "bitsandbytes.functional", "bitsandbytes.backends.cuda.ops"}
_TARGET_MODULES = {_QWEN_MODULE, _TRL_MODULE, _BNB_MODULE, _SOUP_MODULE, *_NF4_MODULES}
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
        from .rope import patch_qwen2_rope

        patch_qwen2_decoder()
        patch_qwen2_rope(module)
        _patch_lazy_mlp(module.Qwen2MLP)
        from .unpad import patch_qwen2_unpad

        patch_qwen2_unpad(module)
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
    elif name == _SOUP_MODULE:
        cls = getattr(module, "SFTTrainerWrapper", None)
        if (isinstance(cls, type) and cls.__module__ == name
                and callable(getattr(cls, "_setup_transformers", None))):
            from .embedding import patch_soup_setup_embedding

            patch_soup_setup_embedding(cls)
    elif name in _NF4_MODULES:
        loaded = {key: sys.modules.get(key) for key in _NF4_MODULES}
        if not all(isinstance(value, ModuleType) for value in loaded.values()):
            return
        # bnb's functional/backend imports can be circular. Never import the
        # other half early, or patch a module still executing. The currently
        # executing loader callback runs AFTER its underlying exec_module.
        if any(key != name and getattr(getattr(value, "__spec__", None), "_initializing", False)
               for key, value in loaded.items()):
            return
        functional, backend = loaded["bitsandbytes.functional"], loaded["bitsandbytes.backends.cuda.ops"]
        if not (callable(getattr(functional, "dequantize_4bit", None))
                and callable(getattr(backend, "_dequantize_4bit_impl", None))):
            return
        from .nf4_patch import patch_bnb_dequant

        patch_bnb_dequant()


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
    No CUDA kernel is imported or compiled here by default. The explicit
    EXP-001 flag starts measured-run preparation on background threads.
    """
    global _IMPORT_FINDER
    if _IMPORT_FINDER is None:
        _IMPORT_FINDER = _PatchFinder()
        sys.meta_path.insert(0, _IMPORT_FINDER)
    for name in _TARGET_MODULES:
        module = sys.modules.get(name)
        if module is not None:
            _register_module(name, module)
    if os.environ.get("SOUP_EXP_PRECOMPILE") == "1":
        from .precompile import start

        start()
