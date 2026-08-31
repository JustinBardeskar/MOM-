from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
import hashlib
import json
import logging
import re
import time
from typing import Any, ClassVar
from uuid import UUID

import httpx
from pydantic import SecretStr

from app.ai_brain.models import (
    AIBrainSettings,
    AgentName,
    CostSummary,
    LLMProvider,
    LLMProviderError,
    LLMProviderName,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    M2ToM3Contract,
    ModelInvocation,
    ModelProfile,
    NoModelAvailableError,
)
from app.domain import TranscriptChunk

logger = logging.getLogger("ai_brain.providers")


# ==========================================
# 0. Adaptive Token-Bucket Rate Limiter
# ==========================================

class AsyncTokenBucketLimiter:
    """Adaptive Token-Bucket Rate Limiter to dynamically pace LLM requests and prevent 429 TPM overages."""

    _INSTANCE: ClassVar[AsyncTokenBucketLimiter | None] = None

    @classmethod
    def get_instance(cls, tpm_limit: int = 50000, rpm_limit: int = 60) -> AsyncTokenBucketLimiter:
        if cls._INSTANCE is None:
            cls._INSTANCE = cls(tpm_limit, rpm_limit)
        return cls._INSTANCE

    def __init__(self, tpm_limit: int = 50000, rpm_limit: int = 60) -> None:
        self._tpm_limit = tpm_limit
        self._rpm_limit = rpm_limit
        self._tokens_remaining = float(tpm_limit)
        self._requests_remaining = float(rpm_limit)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int = 1500) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._last_refill = now

            self._tokens_remaining = min(float(self._tpm_limit), self._tokens_remaining + elapsed * (self._tpm_limit / 60.0))
            self._requests_remaining = min(float(self._rpm_limit), self._requests_remaining + elapsed * (self._rpm_limit / 60.0))

            if self._tokens_remaining < estimated_tokens or self._requests_remaining < 1.0:
                wait_sec = max(
                    0.0,
                    (estimated_tokens - self._tokens_remaining) / (self._tpm_limit / 60.0) if self._tokens_remaining < estimated_tokens else 0.0,
                    (1.0 - self._requests_remaining) / (self._rpm_limit / 60.0) if self._requests_remaining < 1.0 else 0.0,
                )
                if wait_sec > 0.05:
                    logger.info("Adaptive TokenBucket pacing request (pacing %.2fs for capacity)", min(wait_sec, 2.5))
                    await asyncio.sleep(min(wait_sec, 2.5))
                self._tokens_remaining = max(0.0, self._tokens_remaining - estimated_tokens)
                self._requests_remaining = max(0.0, self._requests_remaining - 1.0)
            else:
                self._tokens_remaining -= estimated_tokens
                self._requests_remaining -= 1.0


# ==========================================
# 1. LLM Response Cache Manager
# ==========================================

@dataclass
class CacheEntry:
    response: LLMResponse
    expires_at: float


class CacheManager:
    """In-memory LLM response cache keyed by cryptographic hash of prompt payload."""

    def __init__(self, ttl_seconds: int) -> None:
        self._ttl_seconds = ttl_seconds
        self._entries: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def key(agent: AgentName, profile: ModelProfile, request: LLMRequest) -> str:
        payload = json.dumps(
            {
                "agent": agent,
                "provider": profile.provider,
                "model": profile.model,
                "system": request.system_prompt,
                "user": request.user_prompt,
                "max_output_tokens": request.max_output_tokens,
                "temperature": request.temperature,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def get(self, key: str) -> LLMResponse | None:
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= time.monotonic():
                del self._entries[key]
                return None
            return entry.response.model_copy(deep=True)

    async def set(self, key: str, response: LLMResponse) -> None:
        if self._ttl_seconds == 0:
            return
        async with self._lock:
            self._entries[key] = CacheEntry(
                response=response.model_copy(deep=True),
                expires_at=time.monotonic() + self._ttl_seconds,
            )


# ==========================================
# 2. Context Window & Chunk Selection Manager
# ==========================================

class ContextManager:
    """Manages token bounding and sliding context windows across the 10 specialist agents."""

    def __init__(self, max_tokens: int) -> None:
        self._max_tokens = max_tokens
        from app.ai_brain.context import ContextWindowManager
        self._engine = ContextWindowManager(max_tokens=max_tokens)

    def select(self, contract: M2ToM3Contract, agent: AgentName) -> str:
        return self._engine.select_context(contract, agent)

    def telemetry(self, contract: M2ToM3Contract) -> Any:
        return self._engine.inspect_telemetry(contract)


# ==========================================
# 3. Cost Optimizer & Token Tracker
# ==========================================

class CostOptimizer:
    """Routes toward the cheapest capable model and records token/cost usage."""

    def __init__(self) -> None:
        self._runs: dict[UUID, CostSummary] = {}
        self._lock = asyncio.Lock()

    async def start_run(self, job_id: UUID) -> None:
        async with self._lock:
            self._runs[job_id] = CostSummary(
                input_tokens=0,
                output_tokens=0,
                estimated_cost=0,
            )

    def rank(
        self,
        profiles: list[ModelProfile],
        required_quality: int,
        estimated_input_tokens: int,
    ) -> list[ModelProfile]:
        capable_profiles = [
            profile
            for profile in profiles
            if profile.supports_json
            and profile.quality_rank >= required_quality
            and profile.max_context_tokens >= estimated_input_tokens
        ]
        if not capable_profiles and profiles:
            capable_profiles = list(profiles)
        return sorted(
            capable_profiles,
            key=lambda profile: (
                profile.cost_rank,
                -profile.quality_rank,
                profile.provider.value,
            ),
        )

    async def record(
        self,
        job_id: UUID,
        profile: ModelProfile,
        usage: LLMUsage,
    ) -> float:
        cost = (
            usage.input_tokens * profile.input_cost_per_million
            + usage.output_tokens * profile.output_cost_per_million
        ) / 1_000_000
        async with self._lock:
            current = self._runs.get(
                job_id,
                CostSummary(input_tokens=0, output_tokens=0, estimated_cost=0),
            )
            new_input = current.input_tokens + usage.input_tokens
            new_output = current.output_tokens + usage.output_tokens
            new_cost = round(current.estimated_cost + cost, 8)
            self._runs[job_id] = CostSummary(
                input_tokens=new_input,
                output_tokens=new_output,
                estimated_cost=new_cost,
            )
            logger.info(
                "Token Usage Recorded [Job %s]: +%d in, +%d out. Cumulative: %d in + %d out = %d tokens ($%.6f USD)",
                job_id,
                usage.input_tokens,
                usage.output_tokens,
                new_input,
                new_output,
                new_input + new_output,
                new_cost,
            )
        return cost

    async def summary(self, job_id: UUID) -> CostSummary:
        async with self._lock:
            current = self._runs.get(
                job_id,
                CostSummary(input_tokens=0, output_tokens=0, estimated_cost=0),
            )
            return current.model_copy(deep=True)


# ==========================================
# 4. Observability & AI Brain Monitor
# ==========================================

class AIBrainMonitor:
    """Tracks LLM invocations, latencies, and execution errors per pipeline job."""

    def __init__(self) -> None:
        self._invocations: dict[UUID, list[ModelInvocation]] = {}
        self._errors: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def start_run(self, job_id: UUID) -> None:
        async with self._lock:
            self._invocations[job_id] = []

    async def record_invocation(
        self,
        job_id: UUID,
        invocation: ModelInvocation,
    ) -> None:
        async with self._lock:
            self._invocations.setdefault(job_id, []).append(invocation)
        logger.info(
            "llm_invocation_completed",
            extra={
                "agent": invocation.agent.value,
                "provider": invocation.provider.value,
                "model": invocation.model,
                "cache_hit": invocation.cached,
                "attempts": invocation.attempts,
                "duration_ms": round(invocation.latency_ms, 3),
                "input_tokens": invocation.usage.input_tokens,
                "output_tokens": invocation.usage.output_tokens,
            },
        )

    async def record_error(self, code: str, agent: str) -> None:
        async with self._lock:
            self._errors[code] = self._errors.get(code, 0) + 1
        logger.warning(
            "llm_invocation_failed",
            extra={"agent": agent, "error_code": code},
        )

    async def invocations(self, job_id: UUID) -> list[ModelInvocation]:
        async with self._lock:
            return [
                invocation.model_copy(deep=True) for invocation in self._invocations.get(job_id, [])
            ]


# ==========================================
# 5. Model Router
# ==========================================

class ModelRouter:
    """Selects the most cost-effective and capable provider for a given agent task."""

    _QUALITY_REQUIREMENTS: ClassVar[dict[AgentName, int]] = {
        AgentName.MEETING_UNDERSTANDING: 4,
        AgentName.SUMMARY: 4,
        AgentName.ACTION: 4,
        AgentName.DECISION: 4,
        AgentName.REQUIREMENT: 4,
        AgentName.RISK: 4,
        AgentName.SENTIMENT: 3,
        AgentName.TOPIC: 3,
        AgentName.DEADLINE: 4,
        AgentName.QUESTION: 3,
        AgentName.FOLLOW_UP: 3,
    }

    def __init__(
        self,
        providers: list[LLMProvider],
        cost_optimizer: CostOptimizer,
    ) -> None:
        self._providers = providers
        self._cost_optimizer = cost_optimizer

    def select(
        self,
        agent: AgentName,
        estimated_input_tokens: int,
        route_attempt: int = 1,
    ) -> LLMProvider:
        candidates = [provider for provider in self._providers if provider.available]
        ranked_profiles = self._cost_optimizer.rank(
            [provider.profile for provider in candidates],
            self._QUALITY_REQUIREMENTS[agent],
            estimated_input_tokens,
        )
        if not ranked_profiles:
            raise NoModelAvailableError(
                "no_llm_model_available",
                f"No configured model can execute the {agent.value} agent",
            )
        selected = ranked_profiles[(route_attempt - 1) % len(ranked_profiles)]
        return next(provider for provider in candidates if provider.profile == selected)


# ==========================================
# 6. HTTP LLM Provider Base
# ==========================================

class HttpLLMProvider(ABC):
    def __init__(
        self,
        profile: ModelProfile,
        api_key: SecretStr,
        base_url: str,
        timeout_seconds: float,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._profile = profile
        self._api_key = api_key
        self._extra_headers = extra_headers or {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
            verify=False,
        )

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    @property
    def available(self) -> bool:
        return bool(self._api_key.get_secret_value())

    async def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        est_tokens = max(500, len(request.system_prompt + request.user_prompt) // 4 + request.max_output_tokens)
        await AsyncTokenBucketLimiter.get_instance().acquire(est_tokens)

        logger.info(
            "Calling %s API [model=%s, temp=%.2f, max_tokens=%d]",
            self._profile.provider.value.upper(),
            self._profile.model,
            request.temperature,
            request.max_output_tokens,
            extra={"provider": self._profile.provider.value, "model": self._profile.model},
        )
        max_attempts = 4
        response = None
        duration_ms = 0.0
        for attempt_idx in range(max_attempts):
            started = time.monotonic()
            try:
                response = await self._client.post(
                    self._endpoint(),
                    headers=self._build_headers(),
                    json=self._build_payload(request),
                )
                duration_ms = (time.monotonic() - started) * 1000.0
            except httpx.TimeoutException as exc:
                if attempt_idx == max_attempts - 1:
                    logger.error("%s API Timeout", self._profile.provider.value.upper())
                    raise LLMProviderError("llm_timeout", f"{self._profile.provider.value} timed out") from exc
                await asyncio.sleep(1.5 * (attempt_idx + 1))
                continue
            except httpx.RequestError as exc:
                if attempt_idx == max_attempts - 1:
                    logger.error("%s API Network Error: %s", self._profile.provider.value.upper(), exc)
                    raise LLMProviderError("llm_network_error", f"{self._profile.provider.value} network error") from exc
                await asyncio.sleep(1.5 * (attempt_idx + 1))
                continue

            if response.status_code == 413 and attempt_idx < max_attempts - 1:
                logger.warning("%s payload too large (413). Trimming user prompt and retrying...", self._profile.provider.value.upper())
                orig_prompt = request.user_prompt
                trim_len = int(len(orig_prompt) * 0.60)
                request = LLMRequest(
                    system_prompt=request.system_prompt,
                    user_prompt=orig_prompt[:trim_len],
                    temperature=request.temperature,
                    max_output_tokens=min(request.max_output_tokens, 800),
                )
                await asyncio.sleep(1.0)
                continue

            if response.status_code == 429 and attempt_idx < 1:
                wait_time = 1.0
                logger.warning("%s rate limited (429). Retrying once in %.1fs...", self._profile.provider.value.upper(), wait_time)
                await asyncio.sleep(wait_time)
                continue
            break

        if response is None or response.status_code != 200:
            status_code = response.status_code if response is not None else 500
            resp_text = response.text[:200] if response is not None else "No response"
            logger.error(
                "%s API HTTP Error: status=%d in %.1fms: %s",
                self._profile.provider.value.upper(),
                status_code,
                duration_ms,
                resp_text,
                extra={"status": status_code, "duration_ms": duration_ms, "provider": self._profile.provider.value, "model": self._profile.model},
            )
            raise LLMProviderError(
                "llm_http_error",
                f"{self._profile.provider.value} returned HTTP {status_code}",
            )

        logger.info(
            "%s API Success: status=%d in %.1fms",
            self._profile.provider.value.upper(),
            response.status_code,
            duration_ms,
            extra={"status": response.status_code, "duration_ms": duration_ms, "provider": self._profile.provider.value, "model": self._profile.model},
        )
        return self._parse_response(response.json(), response.headers)

    async def close(self) -> None:
        await self._client.aclose()

    @abstractmethod
    def _endpoint(self) -> str: ...

    @abstractmethod
    def _build_headers(self) -> dict[str, str]: ...

    @abstractmethod
    def _build_payload(self, request: LLMRequest) -> dict[str, Any]: ...

    @abstractmethod
    def _parse_response(
        self,
        payload: dict[str, Any],
        headers: httpx.Headers,
    ) -> LLMResponse: ...


# ==========================================
# 7. Concrete LLM Provider Drivers
# ==========================================

class OpenAILikeProvider(HttpLLMProvider):
    def _endpoint(self) -> str:
        return "/chat/completions"

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        headers.update(self._extra_headers)
        return headers

    def _build_payload(self, request: LLMRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._profile.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        return payload

    def _parse_response(
        self,
        payload: dict[str, Any],
        headers: httpx.Headers,
    ) -> LLMResponse:
        try:
            choices = payload.get("choices") or []
            if not choices:
                raise KeyError("empty choices array")
            choice = choices[0]
            message = choice.get("message") or {}
            content = message.get("content") or ""
            usage_dict = payload.get("usage") or {}
            usage = LLMUsage(
                input_tokens=int(usage_dict.get("prompt_tokens") or 0),
                output_tokens=int(usage_dict.get("completion_tokens") or 0),
            )
            return LLMResponse(
                text=content,
                usage=usage,
                provider_request_id=payload.get("id") or headers.get("x-request-id"),
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError(
                "llm_malformed_response",
                f"Malformed payload from {self._profile.provider.value}",
            ) from exc


class GroqProvider(OpenAILikeProvider):
    def _build_payload(self, request: LLMRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._profile.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        if "gpt-oss" in self._profile.model.lower():
            payload["reasoning_effort"] = "low"
        elif "deepseek" in self._profile.model.lower():
            payload["reasoning_format"] = "hidden"
        return payload

    def _parse_response(
        self,
        payload: dict[str, Any],
        headers: httpx.Headers,
    ) -> LLMResponse:
        resp = super()._parse_response(payload, headers)
        raw_text = resp.text
        # Strip reasoning tags e.g. <think>...</think>
        cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        
        # Strip markdown json code block fences if present
        if "```json" in cleaned:
            cleaned = re.sub(r"```json\s*(.*?)\s*```", r"\1", cleaned, flags=re.DOTALL).strip()
        elif "```" in cleaned:
            cleaned = re.sub(r"```\s*(.*?)\s*```", r"\1", cleaned, flags=re.DOTALL).strip()

        # Find JSON object boundaries
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            cand = cleaned[start:end+1]
            try:
                json.loads(cand)
                cleaned = cand
            except Exception:
                # Attempt trailing comma repair
                fixed = re.sub(r",\s*([\}\]])", r"\1", cand)
                try:
                    json.loads(fixed)
                    cleaned = fixed
                except Exception:
                    cleaned = cand
        return LLMResponse(
            text=cleaned,
            usage=resp.usage,
            provider_request_id=resp.provider_request_id,
        )


class OpenAIProvider(OpenAILikeProvider):
    pass


class OpenRouterProvider(OpenAILikeProvider):
    # Free models on OpenRouter that do NOT support response_format: json_object
    _NO_JSON_FORMAT_MODELS = ("gemma", "minimax", "nemotron", "r1", "deepseek", "llama", "qwen", "mistral")

    def _build_payload(self, request: LLMRequest) -> dict[str, Any]:
        payload = super()._build_payload(request)
        model_lower = self._profile.model.lower()

        # Remove response_format for models that don't support it (causes HTTP 400)
        if any(m in model_lower for m in self._NO_JSON_FORMAT_MODELS):
            payload.pop("response_format", None)

        # Enable reasoning for models that support extended reasoning
        # nemotron, gemma, r1, deepseek all support this on OpenRouter
        if any(m in model_lower for m in ("nemotron", "gemma", "r1", "deepseek")):
            payload["reasoning"] = {"enabled": True}

        return payload

    def _parse_response(
        self,
        payload: dict[str, Any],
        headers: httpx.Headers,
    ) -> LLMResponse:
        resp = super()._parse_response(payload, headers)
        raw_text = resp.text
        # Strip reasoning tags e.g. <think>...</think>
        cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        
        # Strip markdown json code block fences if present
        if "```json" in cleaned:
            cleaned = re.sub(r"```json\s*(.*?)\s*```", r"\1", cleaned, flags=re.DOTALL).strip()
        elif "```" in cleaned:
            cleaned = re.sub(r"```\s*(.*?)\s*```", r"\1", cleaned, flags=re.DOTALL).strip()

        # Find JSON object boundaries
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            cand = cleaned[start:end+1]
            try:
                json.loads(cand)
                cleaned = cand
            except Exception:
                fixed = re.sub(r",\s*([\}\]])", r"\1", cand)
                try:
                    json.loads(fixed)
                    cleaned = fixed
                except Exception:
                    cleaned = cand
        return LLMResponse(
            text=cleaned,
            usage=resp.usage,
            provider_request_id=resp.provider_request_id,
        )


class AnthropicProvider(HttpLLMProvider):
    def _endpoint(self) -> str:
        return "/v1/messages"

    def _build_headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key.get_secret_value(),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _build_payload(self, request: LLMRequest) -> dict[str, Any]:
        return {
            "model": self._profile.model,
            "system": request.system_prompt,
            "messages": [{"role": "user", "content": request.user_prompt}],
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
        }

    def _parse_response(
        self,
        payload: dict[str, Any],
        headers: httpx.Headers,
    ) -> LLMResponse:
        try:
            content = "".join(item["text"] for item in payload["content"] if item["type"] == "text")
            usage_dict = payload.get("usage", {})
            return LLMResponse(
                text=content,
                usage=LLMUsage(
                    input_tokens=usage_dict.get("input_tokens", 0),
                    output_tokens=usage_dict.get("output_tokens", 0),
                ),
                provider_request_id=payload.get("id"),
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError(
                "llm_malformed_response",
                "Malformed Anthropic payload",
            ) from exc


class GeminiProvider(HttpLLMProvider):
    def _endpoint(self) -> str:
        return f"/models/{self._profile.model}:generateContent?key={self._api_key.get_secret_value()}"

    def _build_headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def _build_payload(self, request: LLMRequest) -> dict[str, Any]:
        return {
            "systemInstruction": {"parts": [{"text": request.system_prompt}]},
            "contents": [{"parts": [{"text": request.user_prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": request.temperature,
                "maxOutputTokens": request.max_output_tokens,
            },
        }

    def _parse_response(
        self,
        payload: dict[str, Any],
        headers: httpx.Headers,
    ) -> LLMResponse:
        try:
            parts = payload["candidates"][0]["content"]["parts"]
            content = "".join(part["text"] for part in parts if "text" in part)
            metadata = payload.get("usageMetadata", {})
            return LLMResponse(
                text=content,
                usage=LLMUsage(
                    input_tokens=metadata.get("promptTokenCount", 0),
                    output_tokens=metadata.get("candidatesTokenCount", 0),
                ),
                provider_request_id=headers.get("x-goog-request-id"),
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError(
                "llm_malformed_response",
                "Malformed Gemini payload",
            ) from exc


# ==========================================
# 8. Provider Factory
# ==========================================

def build_providers(settings: AIBrainSettings) -> list[LLMProvider]:
    providers: list[LLMProvider] = []
    for name in settings.provider_priority_list:
        profile = settings.profile_for(name)
        api_key = getattr(settings, f"{name.value}_api_key")
        if api_key is None or not api_key.get_secret_value().strip():
            continue
        base_url = str(getattr(settings, f"{name.value}_base_url"))
        if name == LLMProviderName.GROQ:
            providers.append(
                GroqProvider(
                    profile=profile,
                    api_key=api_key,
                    base_url=base_url,
                    timeout_seconds=settings.request_timeout_seconds,
                )
            )
        elif name == LLMProviderName.OPENAI:
            providers.append(
                OpenAIProvider(
                    profile=profile,
                    api_key=api_key,
                    base_url=base_url,
                    timeout_seconds=settings.request_timeout_seconds,
                )
            )
        elif name == LLMProviderName.ANTHROPIC:
            providers.append(
                AnthropicProvider(
                    profile=profile,
                    api_key=api_key,
                    base_url=base_url,
                    timeout_seconds=settings.request_timeout_seconds,
                )
            )
        elif name == LLMProviderName.GEMINI:
            providers.append(
                GeminiProvider(
                    profile=profile,
                    api_key=api_key,
                    base_url=base_url,
                    timeout_seconds=settings.request_timeout_seconds,
                )
            )
        elif name == LLMProviderName.OPENROUTER:
            extra_headers: dict[str, str] = {}
            if settings.openrouter_http_referer:
                extra_headers["HTTP-Referer"] = settings.openrouter_http_referer
            if settings.openrouter_app_name:
                extra_headers["X-Title"] = settings.openrouter_app_name
            providers.append(
                OpenRouterProvider(
                    profile=profile,
                    api_key=api_key,
                    base_url=base_url,
                    timeout_seconds=settings.request_timeout_seconds,
                    extra_headers=extra_headers,
                )
            )
    return providers
