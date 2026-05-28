"""Agent base class and shared context / result models."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class AgentContext(BaseModel):
    """Immutable context passed to every agent invocation."""

    project_id: str
    document_ids: list[str] = Field(default_factory=list)
    requirement_ids: list[str] = Field(default_factory=list)
    run_id: str = ""


class AgentResult(BaseModel):
    """Standard result returned by every agent."""

    agent_name: str
    status: str = "completed"  # completed | partial | failed
    created_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class BaseAgent(ABC):
    """Abstract base for all ReqInOne agents.

    Every agent:
      1. Receives an ``AgentContext``.
      2. Reads data from repositories / stores.
      3. Calls the LLM gateway if needed.
      4. Writes results back to repositories / stores.
      5. Returns an ``AgentResult``.
    """

    name: str = "base"

    @abstractmethod
    async def run(self, context: AgentContext) -> AgentResult:
        """Execute the agent's task. Must be overridden."""
        raise NotImplementedError
