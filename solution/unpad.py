"""Right-padding removal within one Qwen2 training batch, preserving LoRA RNG.

Register patch_qwen2_unpad(module) after the existing Qwen2 callbacks in
solution.patch. No trainer/config/data changes or lora.py edits are needed.
The original Qwen2Model loop runs on [1,T,H]; only attention is segmented.
Its output is scattered to [B,S,H], so existing head/TRL/Rewind code keeps its
original labels, shift, denominator and masks. Padded hidden states are zero;
their labels must be ignored. Observable hidden-state/attention requests fall
back. Eval, caches, non-SDPA, attention dropout, sliding/dynamic RoPE, left
padding and unknown adapter paths also fall back before any random operation.

Only the active q/v adapter's exact Dropout objects are intercepted. Their
input is scattered to a contiguous fp32 [B,S,H] allocation BEFORE calling
the unchanged native Dropout.forward, then real outputs are gathered. Shape,
dtype, alignment, p and call order therefore match padded native dropout;
pad values cannot affect its RNG. Input values/GEMM algorithms can still
differ by normal fp16 rounding, so end-to-end tolerances need T4 validation.

SOUP_UNPAD=0 disables the path for A/B comparisons. Production defaults on
once explicitly registered; this module imports no torch/Triton at import time.
No own gather kernel is necessary: index_select/index_copy preserve autograd.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
import importlib
import os
from types import MethodType, ModuleType

from .utils import enabled

_MARKER = "_solution_unpad_original"
_ACTIVE = ContextVar("solution_unpad_batch", default=None)
_LABELS = ContextVar("solution_unpad_labels", default=None)
_QWEN = "transformers.models.qwen2.modeling_qwen2"


def unpad_enabled():
    return enabled() and os.environ.get("SOUP_UNPAD", "1").lower() not in {"0", "false", "no"}


def right_padding_lengths(rows):
    """Validate binary nonempty right-padded rows; pure Python for CPU audits."""
    if not rows or not rows[0]:
        return None
    width, lengths = len(rows[0]), []
    for row in rows:
        if len(row) != width:
            return None
        count, padding = 0, False
        for value in row:
            if value not in (0, 1):
                return None
            if value == 0:
                padding = True
            elif padding:
                return None
            else:
                count += 1
        if not count:
            return None
        lengths.append(count)
    return tuple(lengths)


@dataclass
class PackedBatch:
    batch: int
    sequence: int
    hidden: int
    lengths: tuple
    indices: object
    dropout_ids: frozenset
    attention_ids: frozenset
    causal_masks: dict = field(default_factory=dict)

    @property
    def tokens(self):
        return sum(self.lengths)


@contextmanager
def _packing_scope(batch):
    token = _ACTIVE.set(batch)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def _pack(x, batch):
    return x.reshape(batch.batch * batch.sequence, *x.shape[2:]).index_select(0, batch.indices).unsqueeze(0)


def _scatter(x, batch):
    import torch

    tail = x.shape[2:]
    flat = torch.zeros((batch.batch * batch.sequence, *tail), device=x.device, dtype=x.dtype)
    return flat.index_copy(0, batch.indices, x.reshape(batch.tokens, *tail)).reshape(
        batch.batch, batch.sequence, *tail)


def _no_hooks(module):
    return not ("forward" in module.__dict__ or module._forward_hooks or module._forward_pre_hooks
                or module._backward_hooks or module._backward_pre_hooks)


def _dropouts(model, module):
    """Whitelist only native direct projections or one vanilla q/v LoRA."""
    import torch

    selected, dropout_modules = set(), []
    for layer in model.layers:
        attention = layer.self_attn
        if (type(layer) is not module.Qwen2DecoderLayer or not _no_hooks(layer)
                or type(attention) is not module.Qwen2Attention or not _no_hooks(attention)
                or type(layer.input_layernorm) is not module.Qwen2RMSNorm
                or layer.input_layernorm.weight.dtype != torch.float32
                or not _no_hooks(layer.input_layernorm)
                or type(layer.mlp) is not module.Qwen2MLP or not _no_hooks(layer.mlp)
                or attention.attention_dropout != 0 or attention.sliding_window is not None
                or not attention.is_causal):
            return None
        for key in ("q_proj", "k_proj", "v_proj", "o_proj"):
            projection = getattr(attention, key)
            if not _no_hooks(projection):
                return None
            kind = type(projection)
            if kind is torch.nn.Linear or (kind.__module__ == "bitsandbytes.nn.modules"
                                          and kind.__name__ == "Linear4bit"):
                continue
            if (key not in {"q_proj", "v_proj"}
                    or kind.__module__ not in {"peft.tuners.lora.layer", "peft.tuners.lora.bnb"}
                    or kind.__name__ not in {"Linear", "Linear4bit"}
                    or projection.disable_adapters or projection.merged
                    or len(projection.active_adapters) != 1):
                return None
            name = projection.active_adapters[0]
            if (name not in projection.lora_A or name not in projection.lora_B
                    or name in getattr(projection, "lora_variant", {})
                    or getattr(projection, "use_dora", {}).get(name, False)):
                return None
            a, b = projection.lora_A[name], projection.lora_B[name]
            dropout = projection.lora_dropout[name]
            if (type(a) is not torch.nn.Linear or type(b) is not torch.nn.Linear
                    or not _no_hooks(a) or not _no_hooks(b)
                    or a.weight.dtype != torch.float32 or b.weight.dtype != torch.float32
                    or type(dropout) not in {torch.nn.Dropout, torch.nn.Identity}
                    or getattr(dropout, "inplace", False)
                    or dropout._forward_hooks or dropout._forward_pre_hooks
                    or dropout._backward_hooks or dropout._backward_pre_hooks
                    or ("forward" in dropout.__dict__ and not hasattr(dropout.forward, _MARKER))):
                return None
            # A half ingress must use our fp32-dropout PEFT bnb wrapper.
            if kind.__name__ == "Linear4bit":
                from .lora import _MARKER as lora_marker

                if not hasattr(kind.forward, lora_marker):
                    return None
            if type(dropout) is torch.nn.Dropout:
                selected.add(id(dropout))
                dropout_modules.append(dropout)
    # Refuse any other stochastic module whose packed RNG geometry is unknown.
    for child in model.modules():
        if (child._forward_hooks or child._forward_pre_hooks
                or child._backward_hooks or child._backward_pre_hooks):
            return None
        if (isinstance(child, torch.nn.Dropout) and child.training and child.p
                and id(child) not in selected):
            return None
    for dropout in dropout_modules:
        _bind_dropout(dropout)
    return frozenset(selected)


def _plan(model, module, input_ids, attention_mask, position_ids, past_key_values,
          inputs_embeds, use_cache, kwargs):
    import torch

    if (not unpad_enabled() or type(model) is not module.Qwen2Model or not model.training
            or _ACTIVE.get() is not None or not _no_hooks(model)
            or getattr(model, "gradient_checkpointing", False)
            or past_key_values is not None
            or (use_cache if use_cache is not None else getattr(model.config, "use_cache", True))
            or getattr(model.config, "_attn_implementation", None) != "sdpa"
            or getattr(model, "has_sliding_layers", False)
            or getattr(model.rotary_emb, "rope_type", None) != "default"):
        return None
    if any(key not in {"num_items_in_batch", "output_hidden_states", "output_attentions", "return_dict"}
           for key in kwargs):
        return None
    for key in ("output_hidden_states", "output_attentions"):
        if kwargs.get(key) or getattr(model.config, key, False):
            return None
    if kwargs.get("return_dict") is False or getattr(model.config, "return_dict", True) is False:
        return None
    if (not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2
            or attention_mask.requires_grad):
        return None
    labels_call = _LABELS.get()
    if labels_call is None or labels_call[1] is not attention_mask:
        return None
    labels = labels_call[0]
    if not isinstance(labels, torch.Tensor) or labels.shape != attention_mask.shape:
        return None
    if (input_ids is None) == (inputs_embeds is None):
        return None
    source = input_ids if input_ids is not None else inputs_embeds
    if (not isinstance(source, torch.Tensor) or source.shape[:2] != attention_mask.shape
            or source.device != attention_mask.device or labels.device != source.device):
        return None
    if source.device.type == "cuda":
        if not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.float16:
            return None
    elif source.device.type != "cpu":
        return None
    batch, sequence = attention_mask.shape
    # One mask-to-host sync per batch, never per layer. Python validates left/
    # interior padding and obtains row boundaries without repeated GPU item().
    lengths = right_padding_lengths(attention_mask.detach().tolist())
    if lengths is None or sum(lengths) == batch * sequence:
        return None
    if not bool((labels[attention_mask == 0] == -100).all()):
        return None
    if position_ids is not None and (not isinstance(position_ids, torch.Tensor)
            or position_ids.device != source.device or position_ids.ndim != 2
            or position_ids.shape not in {(1, sequence), (batch, sequence)}):
        return None
    dropouts = _dropouts(model, module)
    if dropouts is None:
        return None
    flat_indices = [row * sequence + col for row, length in enumerate(lengths) for col in range(length)]
    indices = torch.tensor(flat_indices, device=source.device, dtype=torch.long)
    return PackedBatch(batch, sequence, model.config.hidden_size, lengths, indices, dropouts,
                       frozenset(id(layer.self_attn) for layer in model.layers))


def _bind_dropout(dropout):
    """Change only a certified model's active adapter instance, never nn.Dropout."""
    original = dropout.forward
    if hasattr(original, _MARKER):
        return

    @wraps(original)
    def forward(self, x):
        import torch

        batch = _ACTIVE.get()
        if batch is None or id(self) not in batch.dropout_ids:
            return original(x)
        if x.shape != (1, batch.tokens, batch.hidden) or x.dtype != torch.float32:
            raise RuntimeError("Unpadded LoRA dropout requires the certified packed fp32 activation")
        padded = _scatter(x, batch)
        # Call exactly the existing module's native forward once. No RNG calls
        # on compact tensors, no generated masks and no skip for padded rows.
        return _pack(original(padded), batch)

    setattr(forward, _MARKER, original)
    dropout.forward = MethodType(forward, dropout)


def _patch_attention(module):
    original = module.Qwen2Attention.forward
    if hasattr(original, _MARKER):
        return

    @wraps(original)
    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, **kwargs):
        batch = _ACTIVE.get()
        if batch is None or id(self) not in batch.attention_ids:
            return original(self, hidden_states, position_embeddings, attention_mask,
                            past_key_values=past_key_values, **kwargs)
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        import torch

        shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        query = self.q_proj(hidden_states).view(shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        query, key = module.apply_rotary_pos_emb(query, key, *position_embeddings)
        rows, start = [], 0
        for length in batch.lengths:
            if length not in batch.causal_masks:
                batch.causal_masks[length] = torch.ones(
                    (1, 1, length, length), device=query.device, dtype=torch.bool).tril_()
            stop = start + length
            # Preserve the pinned SDPA helper's masked-path GQA policy: it
            # repeats KV, rather than enable_gqa=True (math-only on some T4
            # paths). Backend enable/disable settings are never overridden.
            output, _ = sdpa_attention_forward(
                self, query[:, :, start:stop], key[:, :, start:stop], value[:, :, start:stop],
                batch.causal_masks[length], dropout=0.0, scaling=self.scaling, is_causal=False)
            rows.append(output)
            start = stop
        output = torch.cat(rows, dim=1).reshape(*hidden_states.shape[:-1], -1).contiguous()
        return self.o_proj(output), None

    setattr(forward, _MARKER, original)
    module.Qwen2Attention.forward = forward


def _patch_model(module):
    original = module.Qwen2Model.forward
    if hasattr(original, _MARKER):
        return

    @wraps(original)
    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
        args = dict(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                    past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                    **kwargs)
        batch = _plan(self, module, input_ids, attention_mask, position_ids,
                      past_key_values, inputs_embeds, use_cache, kwargs)
        if batch is None:
            return original(self, **args)
        import torch
        from transformers.modeling_outputs import BaseModelOutputWithPast

        positions = position_ids
        if positions is None:
            positions = torch.arange(batch.sequence, device=batch.indices.device).unsqueeze(0)
        positions = positions.expand(batch.batch, batch.sequence)
        packed_positions = positions.reshape(-1).index_select(0, batch.indices).unsqueeze(0)
        with _packing_scope(batch):
            output = original(
                self, input_ids=_pack(input_ids.unsqueeze(-1), batch).squeeze(-1) if input_ids is not None else None,
                inputs_embeds=_pack(inputs_embeds, batch) if inputs_embeds is not None else None,
                # Already prepared mapping skips creation of a packed causal
                # mask across row boundaries; segmented attention owns masks.
                attention_mask={"full_attention": None}, position_ids=packed_positions,
                past_key_values=None, use_cache=False, **kwargs)
        return BaseModelOutputWithPast(last_hidden_state=_scatter(output.last_hidden_state, batch),
                                       past_key_values=output.past_key_values)

    setattr(forward, _MARKER, original)
    module.Qwen2Model.forward = forward


def _patch_causal_labels(module):
    original = module.Qwen2ForCausalLM.forward
    if hasattr(original, _MARKER):
        return

    @wraps(original)
    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                logits_to_keep=0, **kwargs):
        token = _LABELS.set((labels, attention_mask))
        try:
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                            logits_to_keep=logits_to_keep, **kwargs)
        finally:
            _LABELS.reset(token)

    setattr(forward, _MARKER, original)
    module.Qwen2ForCausalLM.forward = forward


def patch_qwen2_unpad(module=None):
    """Call at the END of the real Qwen2 post-import callback; returns installed.

    Install after head/norm/decoder/MLP and optional RoPE patches. Repeated
    calls do not wrap twice. Fake modules are ignored before importing torch.
    No changes to lora.py: only the certified Dropout objects are intercepted.
    """
    if module is None:
        try:
            module = importlib.import_module(_QWEN)
        except (ImportError, AttributeError):
            return False
    names = ("Qwen2Model", "Qwen2Attention", "Qwen2ForCausalLM", "Qwen2DecoderLayer",
             "Qwen2RMSNorm", "Qwen2MLP")
    if (not isinstance(module, ModuleType) or module.__name__ != _QWEN
            or not all(isinstance(getattr(module, name, None), type)
                       and getattr(module, name).__module__ == _QWEN for name in names)):
        return False
    _patch_attention(module)
    _patch_model(module)
    _patch_causal_labels(module)
    return True
