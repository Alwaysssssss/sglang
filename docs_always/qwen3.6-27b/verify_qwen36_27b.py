#!/usr/bin/env python3
import argparse
import concurrent.futures
import os
import time
from typing import Iterable

import requests


DEFAULT_MODEL = "qwen3.6-27b"
DEFAULT_BASE_URL = "http://127.0.0.1:18080/v1"
DEFAULT_MODEL_PATH = "/mnt/shanhai-ai/wenhy/models/Qwen/Qwen/Qwen3___6-27B"
DEFAULT_API_KEY_FILE = "/etc/sglang/qwen36_openai_api_key"


def default_base_url() -> str:
    return os.environ.get("OPENAI_BASE_URL") or os.environ.get("BASE_URL", DEFAULT_BASE_URL)


def normalize_base_url(base_url: str) -> tuple[str, str]:
    value = base_url.rstrip("/")
    if value.endswith("/v1"):
        root = value[:-3]
        v1 = value
    else:
        root = value
        v1 = value + "/v1"
    return root.rstrip("/"), v1.rstrip("/")


def default_api_key() -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    if os.path.exists(DEFAULT_API_KEY_FILE):
        with open(DEFAULT_API_KEY_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "EMPTY"


def auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def chat_payload(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stream: bool = False,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if stream:
        payload["stream"] = True
    return payload


def count_stream_chunks(lines: Iterable[str]) -> tuple[int, bool]:
    count = 0
    done = False
    for raw in lines:
        line = raw.strip()
        if not line or not line.startswith("data:"):
            continue
        count += 1
        if line == "data: [DONE]":
            done = True
            break
    return count, done


def build_long_prompt(tokenizer, target_tokens: int, final_question: str) -> tuple[str, int]:
    unit = "这是 长上下文 验收 片段 请 只 记住 最后 的 问题\n"
    unit_tokens = len(tokenizer.encode(unit, add_special_tokens=False))
    repeat = max(1, target_tokens // max(1, unit_tokens) + 1)
    body = unit * repeat
    measured = len(tokenizer.encode(body, add_special_tokens=False))
    return body + "\n" + final_question, measured


def new_session(trust_env_proxy: bool) -> requests.Session:
    session = requests.Session()
    session.trust_env = trust_env_proxy
    return session


def require_status(response: requests.Response, label: str, expected: int = 200) -> None:
    if response.status_code != expected:
        raise AssertionError(
            f"{label} expected HTTP {expected}, got {response.status_code}: {response.text[:500]}"
        )


def check_health(session: requests.Session, root_url: str, headers: dict[str, str], timeout: float) -> None:
    response = session.get(root_url + "/health", headers=headers, timeout=timeout)
    require_status(response, "health")
    print("PASS health http=200")


def check_models(
    session: requests.Session,
    v1_url: str,
    headers: dict[str, str],
    model: str,
    expected_context_length: int,
    timeout: float,
) -> None:
    response = session.get(v1_url + "/models", headers=headers, timeout=timeout)
    require_status(response, "models")
    obj = response.json()
    matches = [item for item in obj.get("data", []) if item.get("id") == model]
    if not matches:
        raise AssertionError(f"model {model!r} not found in /v1/models: {obj}")
    max_model_len = matches[0].get("max_model_len")
    if expected_context_length and max_model_len != expected_context_length:
        raise AssertionError(
            f"model {model!r} max_model_len expected {expected_context_length}, got {max_model_len}"
        )
    print(f"PASS models id={model} max_model_len={max_model_len}")


def check_bad_key(session: requests.Session, v1_url: str, timeout: float) -> None:
    response = session.get(
        v1_url + "/models",
        headers=auth_headers("wrong-qwen36-key"),
        timeout=timeout,
    )
    require_status(response, "bad key", expected=401)
    print("PASS bad_key http=401")


def check_chat(
    session: requests.Session,
    v1_url: str,
    headers: dict[str, str],
    model: str,
    timeout: float,
) -> None:
    response = session.post(
        v1_url + "/chat/completions",
        headers={**headers, "Content-Type": "application/json"},
        json=chat_payload(
            model=model,
            prompt="请用一句话介绍诗人李白。",
            max_tokens=96,
            temperature=0,
        ),
        timeout=timeout,
    )
    require_status(response, "chat")
    content = response.json()["choices"][0]["message"]["content"]
    if not content.strip():
        raise AssertionError("chat returned empty content")
    print(f"PASS chat completion_tokens={response.json().get('usage', {}).get('completion_tokens')}")


def check_stream(
    session: requests.Session,
    v1_url: str,
    headers: dict[str, str],
    model: str,
    timeout: float,
) -> None:
    response = session.post(
        v1_url + "/chat/completions",
        headers={**headers, "Content-Type": "application/json"},
        json=chat_payload(
            model=model,
            prompt="请用两句话说明流式响应是什么。",
            max_tokens=96,
            temperature=0,
            stream=True,
        ),
        stream=True,
        timeout=timeout,
    )
    require_status(response, "stream")
    count, done = count_stream_chunks(response.iter_lines(decode_unicode=True))
    if count <= 1 or not done:
        raise AssertionError(f"stream did not finish correctly: chunks={count}, done={done}")
    print(f"PASS stream chunks={count} done={done}")


def check_concurrency(
    v1_url: str,
    api_key: str,
    model: str,
    concurrency: int,
    timeout: float,
    trust_env_proxy: bool,
) -> None:
    def one(index: int) -> tuple[int, int, str]:
        session = new_session(trust_env_proxy)
        response = session.post(
            v1_url + "/chat/completions",
            headers={**auth_headers(api_key), "Content-Type": "application/json"},
            json=chat_payload(
                model=model,
                prompt=f"请只用一句中文短句回答并发验收编号 {index}。",
                max_tokens=64,
                temperature=0,
            ),
            timeout=timeout,
        )
        if response.status_code != 200:
            return index, response.status_code, response.text[:200]
        content = response.json()["choices"][0]["message"]["content"]
        return index, response.status_code, content[:120]

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one, range(concurrency)))

    failures = [row for row in results if row[1] != 200 or not row[2]]
    if failures:
        raise AssertionError(f"concurrency failures: {failures}")
    print(f"PASS concurrency requests={concurrency}")


def check_long_context(
    session: requests.Session,
    v1_url: str,
    headers: dict[str, str],
    model: str,
    model_path: str,
    target_tokens: int,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    prompt, measured = build_long_prompt(
        tokenizer,
        target_tokens=target_tokens,
        final_question="最后的问题：请只回答：长上下文验收通过。",
    )
    start = time.time()
    response = session.post(
        v1_url + "/chat/completions",
        headers={**headers, "Content-Type": "application/json"},
        json=chat_payload(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        ),
        timeout=timeout,
    )
    elapsed = time.time() - start
    require_status(response, "long context")
    obj = response.json()
    content = obj["choices"][0]["message"]["content"]
    usage = obj.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    if not content.strip():
        raise AssertionError("long context returned empty content")
    if prompt_tokens < target_tokens:
        raise AssertionError(f"prompt_tokens below target: {prompt_tokens} < {target_tokens}")
    print(
        "PASS long_context "
        f"measured_prompt_tokens_before_chat_template={measured} "
        f"prompt_tokens={prompt_tokens} elapsed_sec={elapsed:.2f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the externally exposed Qwen3.6-27B SGLang OpenAI-compatible service."
    )
    parser.add_argument("--base-url", default=default_base_url())
    parser.add_argument("--api-key", default=default_api_key())
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--expected-context-length", type=int, default=131072)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--long-timeout", type=float, default=900.0)
    parser.add_argument("--target-tokens", type=int, default=100_000)
    parser.add_argument("--long-max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--skip-concurrency", action="store_true")
    parser.add_argument("--skip-long-context", action="store_true")
    parser.add_argument(
        "--trust-env-proxy",
        action="store_true",
        help="Honor HTTP_PROXY/HTTPS_PROXY from the environment. Disabled by default for local checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.api_key == "EMPTY":
        raise SystemExit(
            f"OPENAI_API_KEY is EMPTY. Create {DEFAULT_API_KEY_FILE} or pass --api-key."
        )

    root_url, v1_url = normalize_base_url(args.base_url)
    session = new_session(args.trust_env_proxy)
    headers = auth_headers(args.api_key)

    print(
        "verify_qwen36_27b "
        f"root_url={root_url} v1_url={v1_url} model={args.model} "
        f"target_tokens={args.target_tokens}"
    )
    check_health(session, root_url, headers, args.timeout)
    check_models(
        session,
        v1_url,
        headers,
        args.model,
        args.expected_context_length,
        args.timeout,
    )
    check_bad_key(session, v1_url, args.timeout)
    check_chat(session, v1_url, headers, args.model, args.timeout)
    check_stream(session, v1_url, headers, args.model, args.timeout)
    if not args.skip_concurrency:
        check_concurrency(
            v1_url,
            args.api_key,
            args.model,
            args.concurrency,
            args.timeout,
            args.trust_env_proxy,
        )
    if not args.skip_long_context:
        check_long_context(
            session,
            v1_url,
            headers,
            args.model,
            args.model_path,
            args.target_tokens,
            args.long_max_tokens,
            args.temperature,
            args.long_timeout,
        )
    print("PASS all requested checks")


if __name__ == "__main__":
    main()
