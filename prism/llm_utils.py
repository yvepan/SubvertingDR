"""LLM call helpers: JSON generation with retry, backoff, and trace logging."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request

from prism.pipeline_types import ensure_dir, write_json


Validator = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class LLMConfig:
    model: str
    api_key: str
    base_url: str
    temperature: float = 0.0
    max_retries: int = 10
    timeout_seconds: int = 600
    use_response_format: bool = True


class LLMJsonError(RuntimeError):
    pass


def resolve_api_key(api_key: str | None, api_key_env: str | None) -> str:
    if api_key:
        return api_key
    if api_key_env:
        env_value = os.environ.get(api_key_env)
        if env_value:
            return env_value
    raise LLMJsonError("Missing API key. Pass --api-key or set --api-key-env.")


def build_llm_config(
    model: str,
    base_url: str,
    api_key: str | None = None,
    api_key_env: str | None = None,
    max_retries: int = 10,
    timeout_seconds: int = 600,
) -> LLMConfig:
    return LLMConfig(
        model=model,
        api_key=resolve_api_key(api_key, api_key_env),
        base_url=base_url,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
    )


def completion_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/v1"):
        return f"{trimmed}/chat/completions"
    return f"{trimmed}/v1/chat/completions"


def extract_message_content(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not choices:
        raise LLMJsonError("LLM response does not contain choices.")

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    raise LLMJsonError("Unsupported message content format.")


def strip_code_fences(text: str) -> str:
    fenced = re.match(r"^\s*```(?:json|markdown)?\s*(.*?)\s*```\s*$", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def extract_first_json_block(text: str) -> str:
    stripped = strip_code_fences(text)
    if stripped.startswith("{") or stripped.startswith("["):
        return stripped

    match = re.search(r"(\{.*\}|\[.*\])", stripped, re.DOTALL)
    if match:
        return match.group(1)
    raise LLMJsonError("Could not find a JSON object or array in the model output.")


def parse_json_text(text: str) -> dict[str, Any]:
    payload = json.loads(extract_first_json_block(text))
    if not isinstance(payload, dict):
        raise LLMJsonError("Expected the model to return a JSON object.")
    return payload


def _http_post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _chat_completion_request(
    config: LLMConfig,
    system_prompt: str,
    user_prompt: str,
    use_response_format: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if use_response_format:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }
    response_payload = _http_post_json(
        url=completion_url(config.base_url),
        headers=headers,
        payload=payload,
        timeout_seconds=config.timeout_seconds,
    )
    return payload, response_payload


def generate_json(
    *,
    config: LLMConfig,
    system_prompt: str,
    user_prompt: str,
    validator: Validator | None,
    trace_path: Path,
) -> dict[str, Any]:
    ensure_dir(trace_path.parent)
    latest_error = "unknown error"
    response_format_enabled = config.use_response_format
    current_prompt = user_prompt

    for attempt in range(1, config.max_retries + 1):
        trace_payload: dict[str, Any] = {
            "attempt": attempt,
            "system_prompt": system_prompt,
            "user_prompt": current_prompt,
            "response_format_enabled": response_format_enabled,
        }
        try:
            request_payload, raw_response = _chat_completion_request(
                config=config,
                system_prompt=system_prompt,
                user_prompt=current_prompt,
                use_response_format=response_format_enabled,
            )
            raw_text = extract_message_content(raw_response)
            parsed = parse_json_text(raw_text)
            if validator is not None:
                validator(parsed)
            trace_payload["request_payload"] = request_payload
            trace_payload["raw_response"] = raw_response
            trace_payload["raw_text"] = raw_text
            trace_payload["parsed_response"] = parsed
            trace_payload["status"] = "success"
            write_json(trace_path, trace_payload)
            return parsed
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            latest_error = f"HTTP {exc.code}: {body}"
            trace_payload["status"] = "http_error"
            trace_payload["error"] = latest_error
            write_json(
                trace_path.with_name(f"{trace_path.stem}_attempt_{attempt}.json"),
                trace_payload,
            )
            if response_format_enabled and exc.code in {400, 404, 415, 422}:
                response_format_enabled = False
                continue
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.RemoteDisconnected) as exc:
            latest_error = f"Network Error: {exc}"
            trace_payload["status"] = "network_error"
            trace_payload["error"] = latest_error
            write_json(
                trace_path.with_name(f"{trace_path.stem}_attempt_{attempt}.json"),
                trace_payload,
            )
            backoff_seconds = min(2 ** (attempt - 1), 32)
            if attempt < config.max_retries:
                print(f"[Attempt {attempt}/{config.max_retries}] Network error, retrying in {backoff_seconds}s: {exc}")
                time.sleep(backoff_seconds)
            continue
        except (json.JSONDecodeError, LLMJsonError, ValueError) as exc:
            latest_error = str(exc)
            trace_payload["status"] = "error"
            trace_payload["error"] = latest_error
            write_json(
                trace_path.with_name(f"{trace_path.stem}_attempt_{attempt}.json"),
                trace_payload,
            )

        current_prompt = (
            f"{user_prompt}\n\n"
            f"The previous response was invalid because: {latest_error}\n"
            "Return one JSON object only. Do not include code fences or extra commentary."
        )
        time.sleep(1)

    raise LLMJsonError(f"Failed to obtain valid JSON after retries: {latest_error}")
