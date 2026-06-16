import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("verify_qwen36_27b.py")
CHECK_LONG_CONTEXT_PATH = Path(__file__).with_name("check_long_context.py")
START_SCRIPT_PATH = Path(__file__).with_name("start_qwen36_27b.sh")


def load_module(path=SCRIPT_PATH, name="verify_qwen36_27b"):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def test_normalize_base_url_accepts_root_or_v1():
    module = load_module()

    assert module.normalize_base_url("http://127.0.0.1:18080") == (
        "http://127.0.0.1:18080",
        "http://127.0.0.1:18080/v1",
    )
    assert module.normalize_base_url("http://127.0.0.1:18080/v1/") == (
        "http://127.0.0.1:18080",
        "http://127.0.0.1:18080/v1",
    )


def test_build_long_prompt_reaches_target_and_keeps_final_question():
    module = load_module()

    prompt, measured = module.build_long_prompt(
        FakeTokenizer(),
        target_tokens=25,
        final_question="final question",
    )

    assert measured >= 25
    assert prompt.endswith("final question")


def test_stream_line_counter_requires_done_marker():
    module = load_module()

    count, done = module.count_stream_chunks(
        [
            "",
            "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}",
            "data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}",
            "data: [DONE]",
        ]
    )

    assert count == 3
    assert done is True


def test_chat_payload_uses_expected_model_and_limits():
    module = load_module()

    payload = module.chat_payload(
        model="qwen3.6-27b",
        prompt="hello",
        max_tokens=32,
        temperature=0,
        stream=True,
    )

    assert payload["model"] == "qwen3.6-27b"
    assert payload["stream"] is True
    assert payload["max_tokens"] == 32
    assert payload["temperature"] == 0
    assert payload["messages"] == [{"role": "user", "content": "hello"}]


def test_verify_cli_defaults_to_openai_base_url(monkeypatch):
    module = load_module()
    monkeypatch.setenv("OPENAI_BASE_URL", "http://external.example:18080/v1")
    monkeypatch.setenv("BASE_URL", "http://legacy.example:18080/v1")
    monkeypatch.setattr(sys, "argv", ["verify_qwen36_27b.py"])

    args = module.parse_args()

    assert args.base_url == "http://external.example:18080/v1"


def test_long_context_cli_defaults_to_openai_base_url(monkeypatch):
    module = load_module(CHECK_LONG_CONTEXT_PATH, "check_long_context")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://external.example:18080/v1")
    monkeypatch.setenv("BASE_URL", "http://legacy.example:18080/v1")
    monkeypatch.setattr(sys, "argv", ["check_long_context.py"])

    args = module.parse_args()

    assert args.base_url == "http://external.example:18080/v1"


def test_start_script_enables_qwen3_tool_call_parser():
    script = START_SCRIPT_PATH.read_text()

    assert 'TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"' in script
    assert '--tool-call-parser "$TOOL_CALL_PARSER"' in script
