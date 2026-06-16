#!/usr/bin/env python3
import argparse
import os
import time

import requests
from transformers import AutoTokenizer


DEFAULT_MODEL_PATH = "/mnt/shanhai-ai/wenhy/models/Qwen/Qwen/Qwen3___6-27B"
DEFAULT_API_KEY_FILE = "/etc/sglang/qwen36_openai_api_key"
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"


def default_base_url() -> str:
    return os.environ.get("OPENAI_BASE_URL") or os.environ.get("BASE_URL", DEFAULT_BASE_URL)


def default_api_key() -> str:
    if "OPENAI_API_KEY" in os.environ:
        return os.environ["OPENAI_API_KEY"]
    if os.path.exists(DEFAULT_API_KEY_FILE):
        with open(DEFAULT_API_KEY_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "EMPTY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a long-context OpenAI-compatible chat request to Qwen3.6-27B served by SGLang."
    )
    parser.add_argument("--base-url", default=default_base_url())
    parser.add_argument("--api-key", default=default_api_key())
    parser.add_argument("--model", default="qwen3.6-27b")
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--target-tokens", type=int, default=100_000)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--trust-env-proxy",
        action="store_true",
        help="Honor HTTP_PROXY/HTTPS_PROXY from the environment. Disabled by default for local service checks.",
    )
    return parser.parse_args()


def build_prompt(tokenizer: AutoTokenizer, target_tokens: int) -> tuple[str, int]:
    unit = "这是长上下文验收片段。请只记住最后的问题。\n"
    unit_tokens = len(tokenizer.encode(unit, add_special_tokens=False))
    repeat = max(1, target_tokens // max(1, unit_tokens) + 1)
    body = unit * repeat
    actual_tokens = len(tokenizer.encode(body, add_special_tokens=False))
    prompt = body + "\n最后的问题：请只回答：长上下文验收通过。"
    return prompt, actual_tokens


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    prompt, measured_tokens = build_prompt(tokenizer, args.target_tokens)

    url = args.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    print(
        {
            "base_url": args.base_url,
            "model": args.model,
            "target_tokens": args.target_tokens,
            "measured_prompt_tokens_before_chat_template": measured_tokens,
            "max_tokens": args.max_tokens,
        }
    )

    start = time.time()
    session = requests.Session()
    session.trust_env = args.trust_env_proxy
    response = session.post(
        url,
        headers={"Authorization": f"Bearer {args.api_key}"},
        json=payload,
        timeout=args.timeout,
    )
    elapsed = time.time() - start

    print("status", response.status_code)
    print("elapsed_sec", round(elapsed, 2))
    print("raw_prefix", response.text[:1200])
    response.raise_for_status()

    obj = response.json()
    content = obj["choices"][0]["message"]["content"]
    usage = obj.get("usage") or {}
    print("content_prefix", content[:500])
    print("usage", usage)

    if not content.strip():
        raise AssertionError("empty response")
    if usage.get("prompt_tokens", 0) < args.target_tokens:
        raise AssertionError(
            f"prompt_tokens below target: {usage.get('prompt_tokens')} < {args.target_tokens}"
        )


if __name__ == "__main__":
    main()
