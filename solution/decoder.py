"""Fuse the attention residual add and post-attention norm in Qwen2 5.16.1.

Pinned source: transformers/models/qwen2/modeling_qwen2.py Qwen2DecoderLayer.
Its forward returns one Tensor; attention owns cache updates and receives all
position/cache kwargs unchanged. We keep the layer boundary and final MLP
residual add intact. Fusing that last add with the NEXT layer's input norm
would require replacing the model loop or passing hidden state across layers,
changing hooks/hidden-state observability and the final norm boundary.

The post-attention norm output feeds only the MLP (no LoRA, no dropout). When
all its projections accept fp16 (CUDA fp16 autocast, or frozen bnb fp16
compute), by default the kernel
writes round_fp16 of the exact fp32 norm output: the same values the
projections consume, at half the activation memory. On a Kaggle T4 this cut
the peak from 8003 to 5843 MiB with the same eval loss. SOUP_FUSED_NORM_FP16=0
keeps the exact Qwen2 dtype promotion (fp32 output) for comparison.
Expected removed work: one standalone attention residual add per layer, and
its backward edge sum/cast; the norm row pass reuses the add in registers.
The fp16 option additionally removes a downstream fp32-to-fp16 copy when the
consumer would otherwise autocast that tensor. Profile counts need measurement.
"""

from functools import wraps
import importlib
from types import ModuleType

from .utils import enabled, supported_tensor

_MARKER = "_solution_fused_original"


def patch_qwen2_decoder():
    """Install the real Qwen2DecoderLayer patch once; return whether it is active.

    Imports stay lazy and fake/non-package transformer modules are ignored.
    This function is intentionally not called here; install() owns registration.
    """
    try:
        import torch
        module = importlib.import_module("transformers.models.qwen2.modeling_qwen2")
    except (ImportError, AttributeError):
        return False
    if not isinstance(module, ModuleType):
        return False
    cls = getattr(module, "Qwen2DecoderLayer", None)
    norm_cls = getattr(module, "Qwen2RMSNorm", None)
    if (not isinstance(cls, type) or not issubclass(cls, torch.nn.Module)
            or cls.__module__ != module.__name__ or not isinstance(norm_cls, type)):
        return False
    original = cls.forward
    if hasattr(original, _MARKER):
        return True

    @wraps(original)
    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, position_embeddings=None, **kwargs):
        if not enabled() or type(self) is not cls or not supported_tensor(hidden_states):
            return original(self, hidden_states, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            use_cache=use_cache, position_embeddings=position_embeddings, **kwargs)
        residual = hidden_states
        # Explicit ownership scopes distinguish this norm from post-attention
        # and final norms without tagging instances or guessing module names.
        from .patch import _half_norm_scope, _input_norm_half_ready, _post_norm_half_ready

        norm = self.input_layernorm if _input_norm_half_ready(self, hidden_states, kwargs) else None
        with _half_norm_scope(norm):
            hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            use_cache=use_cache, position_embeddings=position_embeddings, **kwargs)
        norm = self.post_attention_layernorm
        if (type(norm) is norm_cls and supported_tensor(hidden_states)
                and supported_tensor(norm.weight) and hidden_states.shape == residual.shape
                and hidden_states.device == residual.device == norm.weight.device
                and norm.weight.ndim == 1 and norm.weight.numel() == residual.shape[-1]
                and not norm._forward_hooks and not norm._forward_pre_hooks
                and not norm._backward_hooks and not norm._backward_pre_hooks):
            from .kernels.add_rmsnorm import add_rms_norm

            output_dtype = torch.float16 if _post_norm_half_ready(self, residual) else None
            residual, hidden_states = add_rms_norm(
                residual, hidden_states, norm.weight, norm.variance_epsilon, output_dtype)
        else:
            residual = residual + hidden_states
            hidden_states = norm(residual)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    setattr(forward, _MARKER, original)
    cls.forward = forward
    return True
