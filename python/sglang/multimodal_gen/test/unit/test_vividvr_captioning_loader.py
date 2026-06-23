from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PACKAGE_NAME = "sglang.multimodal_gen.runtime.vividvr.caption_sidecar_backend"
VIVIDVR_ROOT = Path("/home/zhiheng/Vivid-VR").resolve()


def test_caption_sidecar_backend_package_is_importable():
    module = importlib.import_module(PACKAGE_NAME)

    assert hasattr(module, "create_captioner")


def test_create_captioner_uses_local_cogvlm2_loader(monkeypatch: pytest.MonkeyPatch):
    backend_module = importlib.import_module(f"{PACKAGE_NAME}.cogvlm2")
    captioner_module = importlib.import_module(f"{PACKAGE_NAME}.captioner")

    tokenizer_calls: list[tuple[str, dict[str, object]]] = []
    config_calls: list[tuple[str, dict[str, object]]] = []
    model_calls: list[tuple[str, object, object]] = []

    class DummyModel:
        def __init__(self) -> None:
            self.eval_called = False

        def eval(self) -> "DummyModel":
            self.eval_called = True
            return self

    dummy_model = DummyModel()

    def fake_tokenizer_from_pretrained(model_path: str, **kwargs):
        tokenizer_calls.append((model_path, kwargs))
        return object()

    def fake_config_from_pretrained(model_path: str, **kwargs):
        config_calls.append((model_path, kwargs))
        return {"config_path": model_path}

    def fake_model_from_pretrained(model_path: str, *, config, torch_dtype, **kwargs):
        model_calls.append((model_path, config, torch_dtype))
        assert kwargs == {}
        return dummy_model

    monkeypatch.setattr(
        backend_module.AutoTokenizer,
        "from_pretrained",
        staticmethod(fake_tokenizer_from_pretrained),
    )
    monkeypatch.setattr(
        backend_module.CogVLMConfig,
        "from_pretrained",
        staticmethod(fake_config_from_pretrained),
    )
    monkeypatch.setattr(
        backend_module.CogVLMVideoForCausalLM,
        "from_pretrained",
        staticmethod(fake_model_from_pretrained),
    )

    args = SimpleNamespace(
        caption_backend="cogvlm2",
        cogvlm2_ckpt_path="/tmp/cogvlm2",
        caption_torch_dtype="bfloat16",
    )

    captioner = captioner_module.create_captioner(args)

    assert captioner is not None
    assert tokenizer_calls == [
        (
            "/tmp/cogvlm2",
            {
                "trust_remote_code": False,
                "fix_mistral_regex": True,
            },
        )
    ]
    assert config_calls == [("/tmp/cogvlm2", {})]
    assert model_calls == [("/tmp/cogvlm2", {"config_path": "/tmp/cogvlm2"}, backend_module.torch.bfloat16)]
    assert dummy_model.eval_called is True
    assert all(Path(path).resolve() != VIVIDVR_ROOT for path in sys.path if path)
