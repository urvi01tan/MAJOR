

from __future__ import annotations

import json
import logging
import os
from typing import Any

import requests
import yaml

from src.ai.clinical_engine import LocalClinicalEngine

log = logging.getLogger(__name__)


def _load_cfg(cfg_path: str = "config/settings.yaml") -> dict:
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}


def _load_env_file() -> None:
    """Load key-value pairs from .env or .env.example into os.environ if not running under pytest."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    for fn in (".env", ".env.example"):
        if os.path.exists(fn):
            try:
                with open(fn, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if k and not is_placeholder_value(v) and k not in os.environ:
                            os.environ[k] = v
            except Exception:
                pass


def is_placeholder_value(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    upper = text.upper()
    if upper in {"PLACEHOLDER", "CHANGEME", "XXX", "TODO"}:
        return True
    if "YOUR_" in upper or upper.endswith("_HERE"):
        return True
    return "<" in text and ">" in text


class LLMClient:
    """Multi-provider LLM client with local clinical reasoning fallback."""

    def __init__(self, cfg_path: str = "config/settings.yaml"):
        _load_env_file()
        self.cfg_path = cfg_path
        cfg = _load_cfg(cfg_path)
        self.cfg = cfg.get("ai") or {}
        env_prov = os.environ.get("SEPSISGUARD_LLM_PROVIDER")
        self.provider = str(env_prov or self.cfg.get("provider") or "openai_compatible").lower()
        self.timeout = float(self.cfg.get("timeout_seconds") or 30)
        self.max_tokens = int(self.cfg.get("max_tokens") or 800)
        self.temperature = float(self.cfg.get("temperature") or 0.2)
        self.offline_mode = bool(self.cfg.get("offline_mode", False))
        self.clinical_engine = LocalClinicalEngine()
        self._resolve_credentials()

    def _env_or(self, env_names: list[str], fallback: Any) -> Any:
        for name in env_names:
            val = os.environ.get(name)
            if val and not is_placeholder_value(val):
                return val
        return fallback

    def _resolve_credentials(self) -> None:
        key_env_var = self.cfg.get("api_key_env") or "SEPSISGUARD_LLM_API_KEY"
        possible_keys = [
            key_env_var,
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "AZURE_OPENAI_API_KEY",
        ]
        provider_block = self.cfg.get(self.provider) or {}
        self.api_key = self._env_or(
            possible_keys,
            provider_block.get("api_key") or self.cfg.get("api_key"),
        )

        # Smart provider auto-detection based on API key prefix
        if self.api_key and not is_placeholder_value(self.api_key):
            key_str = str(self.api_key).strip()
            if key_str.startswith("AIzaSy"):
                self.provider = "gemini"
            elif key_str.startswith("sk-ant-"):
                self.provider = "anthropic"
            elif key_str.startswith("sk-") and self.provider == "openai_compatible":
                self.provider = "openai"

        provider_block = self.cfg.get(self.provider) or {}

        # 2. Base URL resolution
        url_env_var = self.cfg.get("base_url_env") or "SEPSISGUARD_LLM_BASE_URL"
        default_base_url = (
            "https://generativelanguage.googleapis.com/v1beta/openai" if self.provider == "gemini"
            else "https://api.openai.com/v1" if self.provider == "openai"
            else "https://api.anthropic.com" if self.provider == "anthropic"
            else "http://localhost:11434/v1" if self.provider in ("ollama", "local")
            else provider_block.get("base_url")
            or provider_block.get("endpoint")
            or self.cfg.get("base_url")
        )
        self.base_url = self._env_or([url_env_var], default_base_url)

        # 3. Model resolution
        model_env_var = self.cfg.get("model_env") or "SEPSISGUARD_LLM_MODEL"
        default_model = (
            "gemini-2.5-flash" if self.provider == "gemini"
            else "gpt-4o-mini" if self.provider == "openai"
            else "claude-3-5-sonnet-20241022" if self.provider == "anthropic"
            else "llama3" if self.provider in ("ollama", "local")
            else provider_block.get("model")
            or provider_block.get("deployment")
            or self.cfg.get("model")
        )
        resolved_model = self._env_or([model_env_var], default_model)
        # Ensure deprecated gemini-1.5-flash is safely upgraded to gemini-2.5-flash
        if self.provider == "gemini" and (is_placeholder_value(resolved_model) or resolved_model == "gemini-1.5-flash"):
            resolved_model = "gemini-2.5-flash"
        self.model = resolved_model

        # 4. Azure specifics
        self.api_version = provider_block.get("api_version") or self.cfg.get(
            "api_version", "2024-02-15-preview"
        )
        self.chat_path = self.cfg.get("chat_completions_path") or "/chat/completions"

    def credentials_ready(self) -> bool:
        if self.offline_mode:
            return False
        if self.provider in ("ollama", "local"):
            return not is_placeholder_value(self.base_url)
        return not (
            is_placeholder_value(self.api_key)
            or is_placeholder_value(self.base_url)
            or is_placeholder_value(self.model)
        )

    def status(self) -> dict[str, Any]:
        ready = self.credentials_ready()
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "credentials_ready": ready,
            "using_placeholders": not ready,
            "offline_mode": self.offline_mode,
            "mode_display": "External LLM (Live)" if ready else "Local Clinical Intelligence Engine (Offline / Standalone)",
        }

    def _chat_url(self) -> str:
        base = str(self.base_url).rstrip("/")
        if self.provider == "azure_openai":
            deployment = self.model
            return (
                f"{base}/openai/deployments/{deployment}/chat/completions"
                f"?api-version={self.api_version}"
            )
        if self.provider == "anthropic":
            return f"{base}/v1/messages"
        # Standard OpenAI-compatible (including Gemini OpenAI compatibility, Ollama, vLLM)
        if not self.chat_path.startswith("/"):
            return f"{base}/{self.chat_path}"
        return f"{base}{self.chat_path}"

    def _headers(self) -> dict[str, str]:
        if self.provider == "azure_openai":
            return {
                "Content-Type": "application/json",
                "api-key": str(self.api_key),
            }
        if self.provider == "anthropic":
            return {
                "Content-Type": "application/json",
                "x-api-key": str(self.api_key),
                "anthropic-version": "2023-06-01",
            }
        headers = {"Content-Type": "application/json"}
        if self.api_key and not is_placeholder_value(self.api_key):
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        context: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        payload_messages = list(messages)
        if system:
            payload_messages = [{"role": "system", "content": system}, *payload_messages]
        if not self.credentials_ready():
            return self._placeholder_response(payload_messages, context=context)

        temp = self.temperature if temperature is None else temperature
        tokens = self.max_tokens if max_tokens is None else max_tokens

        if self.provider == "anthropic":
            body = {
                "model": self.model,
                "max_tokens": tokens,
                "temperature": temp,
                "system": system or "",
                "messages": [m for m in payload_messages if m.get("role") != "system"],
            }
        else:
            body = {
                "model": self.model,
                "messages": payload_messages,
                "temperature": temp,
                "max_tokens": tokens,
            }

        try:
            response = requests.post(
                self._chat_url(),
                headers=self._headers(),
                data=json.dumps(body),
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            text = self._extract_text(data)
            return {
                "ok": True,
                "placeholder": False,
                "text": text,
                "raw": data,
                "provider": self.provider,
                "model": self.model,
            }
        except Exception as exc:
            log.warning("[LLMClient] external request failed (%s); falling back to LocalClinicalEngine", exc)
            fallback = self._placeholder_response(payload_messages, context=context)
            fallback["warning"] = f"Live LLM request failed ({exc}). Served by Local Clinical Engine."
            return fallback

    def _extract_text(self, data: dict) -> str:
        if isinstance(data, dict) and data.get("choices"):
            choice = data["choices"][0]
            message = choice.get("message") or {}
            return str(message.get("content") or choice.get("text") or "").strip()
        if isinstance(data, dict) and data.get("content"):
            chunks = data["content"]
            if isinstance(chunks, list):
                return "".join(
                    str(part.get("text", "")) for part in chunks if isinstance(part, dict)
                ).strip()
            return str(chunks)
        return json.dumps(data)

    def _placeholder_response(
        self, messages: list[dict[str, str]], context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Local clinical engine response preserving placeholder mode indicator."""
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = str(msg.get("content") or "")
                break

        # Generate medically intelligent clinical reasoning
        clinical_answer = self.clinical_engine.answer_chat_query(
            last_user, context=context, history=messages
        )

        text = clinical_answer

        return {
            "ok": True,
            "placeholder": True,
            "text": text,
            "provider": self.provider,
            "model": self.model,
            "engine": "local_clinical_intelligence",
        # pyrefly: ignore[parse-error]
        }