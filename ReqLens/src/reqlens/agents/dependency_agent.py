"""Dependency Agent – detect typed edges between requirements.

Stage 1: Heuristic + LLM dependency prediction
Stage 2: GNN-based edge prediction (future)
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import structlog

from reqlens.agents.base import AgentContext, AgentResult, BaseAgent
from reqlens.domain.ids import generate_id
from reqlens.domain.models import GraphEdge, Requirement
from reqlens.llm.azure_client import AzureOpenAIClient
from reqlens.llm.prompts import DEPENDENCY_SYSTEM_V1
from reqlens.llm.schemas import DependencyAnalysisOutput
from reqlens.storage.vector_store import VectorStore

logger = structlog.get_logger(__name__)


class DependencyAgent(BaseAgent):
    name = "dependency"

    def __init__(
        self,
        llm: AzureOpenAIClient,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.llm = llm
        self.vector_store = vector_store

    async def run(
        self,
        context: AgentContext,
        requirements: list[Requirement],
    ) -> AgentResult:
        edges = self.detect_dependencies(context, requirements)
        return AgentResult(
            agent_name=self.name,
            status="completed",
            created_ids=[e.id for e in edges],
        )

    def detect_dependencies(
        self,
        context: AgentContext,
        requirements: list[Requirement],
        *,
        top_k_pairs: int = 50,
        similarity_threshold: float = 0.3,
    ) -> list[GraphEdge]:
        """Detect dependency edges between requirements.

        Strategy:
          1. Generate candidate pairs using embedding similarity.
          2. Fall back to all-pairs for small sets.
          3. Ask LLM to classify only the top candidate pairs.
        """
        if len(requirements) < 2:
            return []

        pairs = self._generate_candidate_pairs(
            requirements,
            top_k=top_k_pairs,
            threshold=similarity_threshold,
        )

        if not pairs:
            return []

        all_edges: list[GraphEdge] = []

        # Process pairs in batches
        batch_size = 10
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            pair_block = "\n\n".join(
                f"Pair:\n  A [{a.id}]: {a.text}\n  B [{b.id}]: {b.text}"
                for a, b in batch
            )
            user_prompt = (
                f"Requirement pairs to analyse:\n\n{pair_block}\n\n"
                "For each pair, determine if a dependency edge exists."
            )

            try:
                output: DependencyAnalysisOutput = self.llm.structured_chat(
                    system_prompt=DEPENDENCY_SYSTEM_V1,
                    user_prompt=user_prompt,
                    response_model=DependencyAnalysisOutput,
                    project_id=context.project_id,
                    agent_name=self.name,
                )
            except Exception as exc:
                logger.warning(
                    "dependency.llm_error",
                    batch_index=i,
                    batch_size=len(batch),
                    pair_ids=[(a.id, b.id) for a, b in batch],
                    error=str(exc),
                )
                continue

            for edge_llm in output.edges:
                edge = GraphEdge(
                    id=generate_id("GE"),
                    project_id=context.project_id,
                    source_node_id=edge_llm.source_requirement_id,
                    target_node_id=edge_llm.target_requirement_id,
                    edge_type=edge_llm.edge_type,
                    confidence=edge_llm.confidence,
                    created_by=self.name,
                    explanation=edge_llm.explanation,
                )
                all_edges.append(edge)

        logger.info("dependency.done", pairs_analysed=len(pairs), edges_found=len(all_edges))
        return all_edges

    def _generate_candidate_pairs(
        self,
        requirements: list[Requirement],
        *,
        top_k: int = 50,
        threshold: float = 0.3,
    ) -> list[tuple[Requirement, Requirement]]:
        """Generate likely requirement pairs using embedding similarity.

        For small sets (< 20), use all pairs. For larger sets, use
        vector similarity to prune.
        """
        if len(requirements) <= 20:
            return list(combinations(requirements, 2))

        # Use embedding similarity to find top-k pairs
        if self.vector_store is None:
            # Fallback: take first top_k pairs from all combinations
            all_pairs = list(combinations(requirements, 2))
            return all_pairs[:top_k]

        pairs: list[tuple[Requirement, Requirement, float]] = []
        req_map = {r.id: r for r in requirements}

        for req in requirements:
            emb = self.vector_store.get(req.id)
            if emb is None:
                continue
            results = self.vector_store.search(
                emb,
                top_k=10,
                exclude_ids={req.id},
            )
            for other_id, score in results:
                if score >= threshold and other_id in req_map:
                    pairs.append((req, req_map[other_id], score))

        # Deduplicate and sort by similarity
        seen: set[tuple[str, str]] = set()
        unique: list[tuple[Requirement, Requirement]] = []
        pairs.sort(key=lambda x: x[2], reverse=True)
        for a, b, _ in pairs:
            key = tuple(sorted([a.id, b.id]))
            if key not in seen:
                seen.add(key)
                unique.append((a, b))
            if len(unique) >= top_k:
                break

        return unique
