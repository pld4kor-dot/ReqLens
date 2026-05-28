"""ReqInOne 2.0 – Application settings loaded from environment."""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    dev = "dev"
    staging = "staging"
    production = "production"


class GraphBackend(str, Enum):
    networkx = "networkx"
    neo4j = "neo4j"


class VectorBackend(str, Enum):
    pgvector = "pgvector"
    qdrant = "qdrant"


class Settings(BaseSettings):
    """Central configuration – values come from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Azure OpenAI ────────────────────────────────────────────────
    azure_openai_api_key: str = Field(description="Azure OpenAI API key")
    azure_openai_endpoint: str = Field(
        description="Azure OpenAI resource endpoint (https://…openai.azure.com)"
    )
    azure_openai_base_url: str = Field(
        description="Azure OpenAI v1 base URL (https://…/openai/v1/)"
    )
    azure_openai_chat_deployment: str = Field(
        default="gpt-4.1",
        description="Deployment name for chat / structured-output calls",
    )
    azure_openai_reasoning_deployment: str = Field(
        default="o3",
        description="Deployment name for reasoning-heavy tasks",
    )
    azure_openai_embedding_deployment: str = Field(
        default="text-embedding-3-large",
        description="Deployment name for embeddings",
    )

    # ── Database ────────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+psycopg://reqlens:reqlens@localhost:5432/reqlens",
    )

    # ── Vector DB ───────────────────────────────────────────────────
    vector_backend: VectorBackend = VectorBackend.pgvector

    # ── Graph DB ────────────────────────────────────────────────────
    graph_backend: GraphBackend = GraphBackend.networkx
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "reqinone2"

    # ── Runtime ─────────────────────────────────────────────────────
    environment: Environment = Environment.dev
    log_level: str = "INFO"
    enable_llm_cache: bool = True
    enable_human_review_gate: bool = True

    # ── Token / cost limits ─────────────────────────────────────────
    max_input_tokens_per_call: int = 120_000
    max_output_tokens_per_call: int = 16_000
    embedding_batch_size: int = 64

    @property
    def is_dev(self) -> bool:
        return self.environment == Environment.dev


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor – import and call ``get_settings()``."""
    return Settings()  # type: ignore[call-arg]
