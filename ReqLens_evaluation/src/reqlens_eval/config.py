"""Central configuration for the evaluation pipeline."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EvalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Azure OpenAI ────────────────────────────────────────────────────────
    azure_openai_api_key: str = Field(default="")
    azure_openai_endpoint: str = Field(default="")
    azure_openai_base_url: str = Field(default="")
    azure_openai_chat_deployment: str = Field(default="gpt-4.1-mini")
    azure_openai_judge_deployment: str = Field(default="gpt-4.1-mini")
    azure_openai_embedding_deployment: str = Field(default="text-embedding-3-large")
    # Dedicated deployment for the reqinone_v1 adapter — should point to a
    # gpt-4o-mini deployment to match the original notebook (temperature=0.5).
    # Falls back to azure_openai_chat_deployment when not set.
    reqinone_v1_deployment: str = Field(default="")

    # ── Paths ───────────────────────────────────────────────────────────────
    benchmark_output_dir: str = Field(default="../reqlens_benchmark_builder/outputs")
    reqlens_src_path: str = Field(default="../ReqLens/src")
    eval_output_dir: str = Field(default="outputs")

    # ── Runtime ─────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    # judge_temperature: float = Field(default=0.0)
    # max_judge_tokens: int = Field(default=512)  # OLD: too tight for complex duplicate/contradiction reasoning
    max_judge_tokens: int = Field(default=2048)

    @property
    def benchmark_path(self) -> Path:
        return Path(self.benchmark_output_dir)

    @property
    def eval_output_path(self) -> Path:
        return Path(self.eval_output_dir)


@lru_cache(maxsize=1)
def get_settings() -> EvalSettings:
    return EvalSettings()
