"""Central configuration – loaded from environment / .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Azure OpenAI ────────────────────────────────────────────────────────────
    azure_openai_api_key: str = Field(default="")
    azure_openai_endpoint: str = Field(default="")
    azure_openai_base_url: str = Field(default="")
    # Primary chat deployment — used for source bundle generation (quality-critical)
    azure_openai_chat_deployment: str = Field(default="gpt-4.1-mini")
    # Reasoning / validation deployment — used for coverage checking and merging
    azure_openai_reasoning_deployment: str = Field(default="gpt-4.1-mini")
    # Extraction deployment — used for per-chunk req extraction (cost-sensitive)
    azure_openai_extraction_deployment: str = Field(default="gpt-4.1-mini")
    azure_openai_embedding_deployment: str = Field(default="text-embedding-3-large")

    # ── Runtime ─────────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    output_dir: str = "outputs_temp"

    # ── PROMISE ─────────────────────────────────────────────────────────────────
    promise_input: str = "data/promise/promise.csv"
    promise_has_header: bool = True
    promise_project_col: str = "ProjectID"
    promise_text_col: str = "RequirementText"
    promise_label_col: str = "class1"
    promise_max_projects: int = 5
    promise_max_reqs_per_project: int = 40

    # ── PURE ────────────────────────────────────────────────────────────────────
    pure_input_dir: str = "data/pure"
    pure_max_docs: int = 5
    # Chunk sizes in characters (≈ 4 chars per token)
    pure_raw_chunk_chars: int = 12_000
    pure_raw_chunk_overlap: int = 1_200
    # How many leading chars to extract for global context
    pure_profile_summary_chars: int = 50_000
    # Hard cap on LLM calls per extraction path (avoids runaway cost on huge docs)
    pure_max_section_chunks: int = 60
    pure_max_raw_chunks: int = 50
    # Minimum requirement-signal count for a chunk to be sent for extraction
    pure_req_signal_min: int = 2

    # ── Generation ──────────────────────────────────────────────────────────────
    artifact_count: int = 3
    max_repair_rounds: int = 1
    min_coverage_rate: float = 0.90
    max_unsupported_count: int = 2
    temp_generation: float = 0.4
    temp_validation: float = 0.0
    # Token budgets per call type
    max_tokens_generation: int = 16_000   # source bundle can be very long
    max_tokens_validation: int = 4_096    # per-batch validator (10 reqs max)
    max_tokens_extraction: int = 4_096    # per-chunk extraction
    max_tokens_merge: int = 8_192         # requirement merge pass

    # ── Derived paths ────────────────────────────────────────────────────────────
    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def pure_input_path(self) -> Path:
        return Path(self.pure_input_dir)

    @property
    def promise_input_path(self) -> Path:
        return Path(self.promise_input)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor – call ``get_settings()`` from every module."""
    return Settings()  # type: ignore[call-arg]
