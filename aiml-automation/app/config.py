"""
app.config
==========
Global Configuration, Structured JSON Logging, and TLS Security Setup.

This module provides:
- Settings: Pydantic-based environment settings with validation for dev/staging/prod.
- configure_logging: Configures structured JSON stream logging with ISO timestamps and telemetry metadata.
- configure_system_trust_store: Injects native operating system TLS roots for outbound requests.
"""

from datetime import UTC, datetime
from functools import lru_cache
import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ==========================================
# Application Configuration Settings
# ==========================================

class Settings(BaseSettings):
    """
    Central application configuration loaded from environment variables and .env file.
    Prefix: AUTOMATION_ (e.g. AUTOMATION_PORT, AUTOMATION_ENVIRONMENT).
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AUTOMATION_",
        case_sensitive=False,
        extra="ignore",
    )

    # Server & Environment
    environment: Literal["development", "testing", "staging", "production"] = "development"
    service_name: str = "meeting-intelligence-automation"
    host: str = "0.0.0.0"
    port: int = 8100
    log_level: str = "INFO"
    docs_enabled: bool = True
    api_key: SecretStr | None = None
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    callback_allowed_hosts: str = ""
    asset_allowed_hosts: str = "graph.microsoft.com,sharepoint.com,microsoftstream.com,microsoft.com,1drv.ms,teams.microsoft.com"

    # Worker & Execution
    worker_enabled: bool = True
    ai_brain_enabled: bool = True
    worker_concurrency: int = Field(default=2, ge=1, le=16)
    work_directory: Path = Path("runtime/jobs")
    keep_work_files: bool = False

    # Ingestion & Media Processing
    max_download_bytes: int = Field(default=10_737_418_240, ge=1)  # 10 GB (covers 3-4+ hour HD recordings)
    download_timeout_seconds: float = Field(default=1800.0, gt=0)   # 30 minutes
    ffmpeg_binary: str | None = None
    ffmpeg_timeout_seconds: float = Field(default=1800.0, gt=0)     # 30 minutes
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    # Preprocessing Parameters
    preprocessing_version: str = "1.0.0"
    filler_words: str = "um,uh,erm,hmm,mm-hmm"
    token_encoding: str = "cl100k_base"
    chunk_target_tokens: int = Field(default=800, ge=100)
    chunk_max_tokens: int = Field(default=1000, ge=100)
    chunk_overlap_tokens: int = Field(default=120, ge=0)
    context_neighbor_tokens: int = Field(default=120, ge=0)

    @property
    def cors_origin_list(self) -> list[str]:
        """Parsed list of allowed CORS origins."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def callback_host_list(self) -> list[str]:
        """Parsed list of allowed callback hostnames."""
        return [host.strip().lower() for host in self.callback_allowed_hosts.split(",") if host.strip()]

    @property
    def asset_host_list(self) -> list[str]:
        """Parsed list of allowed asset download hostnames."""
        return [host.strip().lower() for host in self.asset_allowed_hosts.split(",") if host.strip()]

    @property
    def filler_word_list(self) -> list[str]:
        """Parsed list of speech filler words."""
        return [word.strip().lower() for word in self.filler_words.split(",") if word.strip()]

    @model_validator(mode="after")
    def validate_production_security(self) -> "Settings":
        """Enforces security rules for staging and production environments."""
        if self.chunk_target_tokens > self.chunk_max_tokens:
            raise ValueError("chunk_target_tokens cannot exceed chunk_max_tokens")
        if self.chunk_overlap_tokens >= self.chunk_target_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_target_tokens")
        if self.environment in {"staging", "production"} and self.api_key is None:
            raise ValueError("AUTOMATION_API_KEY is required in staging and production")
        if self.environment == "production" and self.docs_enabled:
            raise ValueError("AUTOMATION_DOCS_ENABLED must be false in production")
        if self.environment in {"staging", "production"} and not self.asset_host_list:
            raise ValueError("AUTOMATION_ASSET_ALLOWED_HOSTS is required in staging and production")
        return self


@lru_cache
def get_settings() -> Settings:
    """Cached singleton instance of application settings."""
    return Settings()


# ==========================================
# Structured Logging & TLS Configuration
# ==========================================

_tls_configured = False


def configure_system_trust_store() -> None:
    """Use operating-system certificate roots for outbound HTTPS clients."""
    global _tls_configured
    if _tls_configured:
        return
    import truststore
    truststore.inject_into_ssl()
    _tls_configured = True


class JsonFormatter(logging.Formatter):
    """Formats log records as structured JSON lines with execution telemetry."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Include optional contextual metadata if present on the log record
        for field in (
            "request_id",
            "method",
            "endpoint",
            "status",
            "duration_ms",
            "agent",
            "provider",
            "model",
            "cache_hit",
            "attempts",
            "input_tokens",
            "output_tokens",
            "error_code",
        ):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str) -> None:
    """Configures the root logger with JSON output formatting and level."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(level.upper())
