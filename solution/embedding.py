"""Lossless fp16 storage for a frozen, tied Qwen2 embedding/head after PEFT setup.

This changes representation only after torch.equal(w_half.float(), w) holds
for EVERY table element. Checking is chunked to bound fp32 roundtrip scratch
at 32 MiB; the candidate fp16 table is never installed before full acceptance.
No dimensions are hardcoded: saving is weight.numel()*2 bytes, i.e.
466747392 bytes (445.125 MiB) at V=151936,H=1536.

The existing Parameter and both module identities are preserved. Embedding
gather is upcast to the original fp32 output. The ordinary Linear forward
uses fp16 storage under CUDA autocast, which casts the same numerical weight
as before; outside autocast it constructs a temporary fp32 weight and calls
F.linear, preserving dense fallback/eval/generation behavior and dtype errors.
There is no persistent fp32 copy. This fallback can temporarily allocate the
full fp32 weight, so the saving is a training-storage claim, not an eval peak
claim. Fused CE already treats weight.to(fp16) as a no-op for fp16 storage.

PEFT 0.20.0 utils/save_and_load.py:get_peft_model_state_dict decides auto-saving
embeddings from targets/vocabulary, not weight dtype. Before conversion we
invoke that exact auto policy and refuse if any adapter saves non-LoRA keys.
No config/target/parameter names or shapes change; no save method is patched.

Registration target: soup_cli.trainer.sft.SFTTrainerWrapper._setup_transformers
AFTER its original body (Soup 0.77.0 trainer/sft.py:1611). For the existing
lazy installer add soup_cli.trainer.sft to its import targets and invoke
patch_soup_setup_embedding(module.SFTTrainerWrapper) in the loader callback.
With no argument this function only checks an already-loaded Soup module;
it never imports Soup/torch/transformers merely to register a cold patch.
"""

from functools import wraps
import importlib
import sys
from types import MethodType, ModuleType

from .utils import enabled, qwen_model

_TARGET = "soup_cli.trainer.sft"
_MARKER = "_solution_lossless_embedding_original"
_ACTIVE = "_solution_lossless_embedding_storage"
_REPORTED = set()


def _decline(reason):
    if reason not in _REPORTED:
        _REPORTED.add(reason)
        print(f"[solution] lossless embedding skipped: {reason}", file=sys.stderr, flush=True)
    return False


def _lossless_candidate(weight):
    import torch

    # Full candidate is the eventual storage, not a second full fp32 copy.
    candidate = torch.empty_like(weight, dtype=torch.float16)
    rows_per_chunk = max(1, (32 * 1024**2) // (weight.shape[1] * 4))
    with torch.no_grad():
        for start in range(0, weight.shape[0], rows_per_chunk):
            source = weight[start:start + rows_per_chunk]
            part = candidate[start:start + rows_per_chunk]
            part.copy_(source)
            if not torch.equal(part.float(), source):
                return None
    return candidate


def compress_tied_embedding(model):
    """Return True only when the final real PEFT Qwen2 has lossless storage."""
    import torch

    if not enabled() or not isinstance(model, torch.nn.Module):
        return False
    try:
        module = importlib.import_module("transformers.models.qwen2.modeling_qwen2")
    except (ImportError, AttributeError):
        return False
    cls = getattr(module, "Qwen2ForCausalLM", None)
    if not isinstance(cls, type) or cls.__module__ != module.__name__:
        return False
    raw = qwen_model(model, cls)
    if raw is None:
        return _decline("model is not an exact Qwen2ForCausalLM")
    embedding, head = raw.get_input_embeddings(), raw.get_output_embeddings()
    if type(embedding) is not torch.nn.Embedding or type(head) is not torch.nn.Linear:
        return _decline("input/output embedding modules are wrapped or unsupported")
    weight = embedding.weight
    if head.weight is not weight:
        return _decline("lm_head and embed_tokens do not share one Parameter")
    if getattr(raw, _ACTIVE, False):
        return weight.dtype == torch.float16 and not weight.requires_grad
    if (weight.requires_grad or weight.dtype != torch.float32 or not weight.is_cuda
            or not weight.is_contiguous() or weight.ndim != 2):
        return _decline("tied weight must be frozen contiguous fp32 on CUDA after kbit preparation")
    if (embedding.max_norm is not None or head.bias is not None
            or "forward" in embedding.__dict__ or "forward" in head.__dict__):
        return _decline("custom/max-norm embedding or custom/biased head forward is unsupported")
    configs = getattr(model, "peft_config", None)
    if not isinstance(configs, dict) or not configs:
        return _decline("final PEFT adapter configuration is absent")
    for config in configs.values():
        targets = getattr(config, "target_modules", None)
        peft_type = getattr(config, "peft_type", None)
        if (getattr(peft_type, "value", peft_type) != "LORA"
                or not isinstance(targets, (tuple, list, set)) or not targets
                or not set(targets).issubset({"q_proj", "v_proj"})
                or getattr(config, "modules_to_save", None)
                or getattr(config, "trainable_token_indices", None)
                or getattr(config, "bias", "none") != "none"):
            return _decline("requires q_proj/v_proj-only LoRA with no extra saved modules or tokens")
    from peft import get_peft_model_state_dict

    # Match the SAME auto-save policy used by PeftModel.save_pretrained. In
    # particular a resized vocabulary can request embeddings despite q/v targets.
    for name in configs:
        adapter = get_peft_model_state_dict(model, adapter_name=name, save_embedding_layers="auto")
        if not adapter or any("lora_" not in key for key in adapter):
            return _decline("PEFT auto-save includes non-LoRA tensors (e.g. resized embeddings)")
    candidate = _lossless_candidate(weight)
    if candidate is None:
        return _decline("at least one table element fails fp16-to-fp32 exact roundtrip")
    original_embedding = type(embedding).forward
    original_head = type(head).forward

    @wraps(original_embedding)
    def embedding_forward(self, input):
        return original_embedding(self, input).to(torch.float32)

    @wraps(original_head)
    def head_forward(self, input):
        if torch.is_autocast_enabled("cuda"):
            return original_head(self, input)
        return torch.nn.functional.linear(input, self.weight.float(), self.bias)

    # Change storage of the SAME Parameter, preserving tying, names and any
    # external references to that Parameter. Closures capture no old table.
    with torch.no_grad():
        weight.data = candidate
    embedding.forward = MethodType(embedding_forward, embedding)
    head.forward = MethodType(head_forward, head)
    setattr(raw, _ACTIVE, True)
    return True


def patch_soup_setup_embedding(wrapper_cls=None):
    """Patch the loaded Soup wrapper, or a class supplied by its import callback."""
    if wrapper_cls is None:
        module = sys.modules.get(_TARGET)
        if not isinstance(module, ModuleType):
            return False
        wrapper_cls = getattr(module, "SFTTrainerWrapper", None)
    if (not isinstance(wrapper_cls, type) or wrapper_cls.__module__ != _TARGET
            or not callable(getattr(wrapper_cls, "_setup_transformers", None))):
        return False
    original = wrapper_cls._setup_transformers
    if hasattr(original, _MARKER):
        return True

    @wraps(original)
    def setup(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        import torch

        if not isinstance(getattr(self, "model", None), torch.nn.Module):
            return result
        compress_tied_embedding(self.model)
        return result

    setattr(setup, _MARKER, original)
    wrapper_cls._setup_transformers = setup
    return True
