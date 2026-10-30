"""Lossless tied-table storage: exact values, model gradients and adapter saves."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _model(tmp_path):
    pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from peft.tuners.lora.layer import Linear

    torch.manual_seed(120)
    config = Qwen2Config(vocab_size=65, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                         tie_word_embeddings=True, attention_dropout=0.0, use_cache=False)
    config._attn_implementation = "eager"
    base_path = tmp_path / "base"
    config.save_pretrained(base_path)
    config._name_or_path = str(base_path)
    base = Qwen2ForCausalLM(config).cuda()
    assert base.lm_head.weight is base.model.embed_tokens.weight
    # Integers / 1024, with at most seven significant bits: all values are
    # bf16-representable AND safely fp16-representable, including exact zeros.
    with torch.no_grad():
        table = torch.randint(-64, 65, base.model.embed_tokens.weight.shape, device="cuda").float() / 1024
        assert torch.equal(table.bfloat16().float(), table)
        base.model.embed_tokens.weight.copy_(table)
    base = peft.prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    adapter = peft.LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=8,
                              lora_dropout=0.0, target_modules=["q_proj", "v_proj"], bias="none")
    # Use PEFT's documented custom mapping ahead of optional quantizers; the
    # Kaggle image can contain a torchao build incompatible with its torch.
    adapter._register_custom_module({torch.nn.Linear: Linear})
    model = peft.get_peft_model(base, adapter).train()
    model.peft_config["default"].base_model_name_or_path = str(base_path)
    with torch.no_grad():
        for name, value in model.named_parameters():
            if "lora_B" in name:
                value.normal_(0, 0.02)  # exercise nonzero A and B gradients
    return model, base


def _run(model, ids, labels, amp):
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        output = model(input_ids=ids, labels=labels, use_cache=False)
    logits, loss = output.logits.detach().clone(), output.loss.detach().clone()
    output.loss.backward()
    gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters()
                 if p.requires_grad and p.grad is not None}
    return logits, loss, gradients


@pytest.mark.parametrize("amp", [False, True])
def test_roundtrip_preserves_model_and_lora_gradients_bitwise(tmp_path, monkeypatch, amp):
    from solution.embedding import compress_tied_embedding

    model, base = _model(tmp_path)
    weight = base.lm_head.weight
    embedding, head = model.get_input_embeddings(), model.get_output_embeddings()
    original_values = weight.detach().clone()
    original_shapes = {name: tuple(t.shape) for name, t in model.state_dict().items()}
    ids = torch.randint(0, 65, (2, 7), device="cuda")
    labels = ids.clone()
    labels[:, :2] = -100
    # Isolate this representation change from unrelated global decoder/head
    # patches. Converted module forwards must still preserve semantics when
    # the global fused switch is disabled, since their storage already changed.
    monkeypatch.setenv("SOUP_FUSED_DISABLE", "1")
    expected_embedding = embedding(ids).detach().clone()
    expected = _run(model, ids, labels, amp)
    monkeypatch.delenv("SOUP_FUSED_DISABLE")
    assert compress_tied_embedding(model)
    assert compress_tied_embedding(model)  # no new storage/wrappers on repeat
    assert base.lm_head.weight is base.model.embed_tokens.weight is weight
    assert weight.dtype == torch.float16 and not weight.requires_grad
    assert model.get_input_embeddings() is embedding and model.get_output_embeddings() is head
    assert type(head) is torch.nn.Linear and type(embedding) is torch.nn.Embedding
    assert torch.equal(weight.float(), original_values)
    assert {name: tuple(t.shape) for name, t in model.state_dict().items()} == original_shapes
    monkeypatch.setenv("SOUP_FUSED_DISABLE", "1")
    actual_embedding = embedding(ids)
    assert actual_embedding.dtype == torch.float32
    assert torch.equal(actual_embedding, expected_embedding)
    actual = _run(model, ids, labels, amp)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert actual[2].keys() == expected[2].keys()
    assert actual[2] and any(torch.count_nonzero(g).item() for g in actual[2].values())
    for name in expected[2]:
        assert torch.equal(actual[2][name], expected[2][name]), name


def test_fused_ce_receives_identical_half_weights(tmp_path, monkeypatch):
    from solution.embedding import compress_tied_embedding
    from solution.kernels import fused_linear_cross_entropy

    monkeypatch.delenv("SOUP_FUSED_DISABLE", raising=False)
    model, base = _model(tmp_path)
    hidden = torch.randn((17, 32), device="cuda", requires_grad=True)
    labels = torch.randint(0, 65, (17,), device="cuda")
    labels[::3] = -100
    upstream = torch.randn(17, device="cuda")
    expected = fused_linear_cross_entropy(hidden, base.lm_head.weight, labels)
    expected[0].backward(upstream)
    expected_dx = hidden.grad.clone()
    expected = tuple(t.detach().clone() for t in expected)
    hidden.grad = None
    assert compress_tied_embedding(model)
    actual = fused_linear_cross_entropy(hidden, base.lm_head.weight, labels)
    actual[0].backward(upstream)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert torch.equal(hidden.grad, expected_dx)


@pytest.mark.parametrize("value", [1.0001, 2.0**-25, 65536.0])
def test_any_nonrepresentable_element_declines_without_mutation(tmp_path, monkeypatch, capsys, value):
    import solution.embedding as embedding

    monkeypatch.delenv("SOUP_FUSED_DISABLE", raising=False)
    monkeypatch.setattr(embedding, "_REPORTED", set())
    model, base = _model(tmp_path)
    weight = base.lm_head.weight
    with torch.no_grad():
        weight[0, 0] = value
    before = weight.clone()
    pointer = weight.data_ptr()
    assert not embedding.compress_tied_embedding(model)
    assert not embedding.compress_tied_embedding(model)
    assert weight.dtype == torch.float32 and weight.data_ptr() == pointer
    assert base.lm_head.weight is base.model.embed_tokens.weight is weight
    assert torch.equal(weight, before)
    assert "forward" not in base.model.embed_tokens.__dict__
    assert "forward" not in base.lm_head.__dict__
    assert capsys.readouterr().err.count("fails fp16-to-fp32 exact roundtrip") == 1


def test_adapter_save_auto_keys_config_and_values_unchanged(tmp_path, monkeypatch):
    pytest.importorskip("peft")
    pytest.importorskip("safetensors.torch")
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from solution.embedding import compress_tied_embedding

    monkeypatch.delenv("SOUP_FUSED_DISABLE", raising=False)
    model, _ = _model(tmp_path)
    expected = {name: value.clone() for name, value in
                get_peft_model_state_dict(model, save_embedding_layers="auto").items()}
    model.save_pretrained(tmp_path / "before", save_embedding_layers="auto")
    assert compress_tied_embedding(model)
    actual = get_peft_model_state_dict(model, save_embedding_layers="auto")
    model.save_pretrained(tmp_path / "after", save_embedding_layers="auto")
    assert expected and actual.keys() == expected.keys()
    assert all("lora_" in name for name in actual)
    assert all(torch.equal(actual[name], value) for name, value in expected.items())
    config_before = json.loads((tmp_path / "before" / "adapter_config.json").read_text())
    config_after = json.loads((tmp_path / "after" / "adapter_config.json").read_text())
    assert config_before == config_after
    stored_before = load_file(tmp_path / "before" / "adapter_model.safetensors")
    stored_after = load_file(tmp_path / "after" / "adapter_model.safetensors")
    assert stored_before.keys() == stored_after.keys()
    assert all(torch.equal(stored_before[name], stored_after[name]) for name in stored_before)


def test_resized_auto_saved_embedding_is_not_compressed(tmp_path, monkeypatch):
    from solution.embedding import compress_tied_embedding

    monkeypatch.delenv("SOUP_FUSED_DISABLE", raising=False)
    model, base = _model(tmp_path)
    config_path = tmp_path / "base" / "config.json"
    original_base_config = json.loads(config_path.read_text())
    original_base_config["vocab_size"] -= 1
    config_path.write_text(json.dumps(original_base_config))
    with pytest.warns(UserWarning, match="embedding layer has been resized"):
        assert not compress_tied_embedding(model)
    assert base.lm_head.weight.dtype == torch.float32


def test_fake_setup_model_is_noop_and_original_return_is_preserved(monkeypatch):
    import solution.embedding as embedding

    calls = []

    def original(self, cfg, tcfg):
        calls.append((cfg, tcfg))
        self.model = SimpleNamespace(fake=True)
        return "original-result"

    wrapper = type("SFTTrainerWrapper", (), {"__module__": "soup_cli.trainer.sft",
                                            "_setup_transformers": original})
    assert embedding.patch_soup_setup_embedding(wrapper)
    patched = wrapper._setup_transformers
    assert embedding.patch_soup_setup_embedding(wrapper) and wrapper._setup_transformers is patched

    def forbidden(_):
        raise AssertionError("fake setup model reached the real conversion helper")

    monkeypatch.setattr(embedding, "compress_tied_embedding", forbidden)
    owner = wrapper()
    assert owner._setup_transformers("cfg", "tcfg") == "original-result"
    assert calls == [("cfg", "tcfg")] and owner.model.fake


def test_cold_registration_does_not_import_training_packages():
    code = """
import sys
from solution.embedding import patch_soup_setup_embedding
assert not patch_soup_setup_embedding()
assert not any(name in sys.modules for name in ('torch','transformers','peft','soup_cli','triton'))
"""
    subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2], check=True)
