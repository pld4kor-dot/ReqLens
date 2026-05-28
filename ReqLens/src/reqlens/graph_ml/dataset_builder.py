"""Dataset builder – convert reviewed graph to PyTorch Geometric HeteroData."""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from reqlens.domain.models import GraphEdge, Requirement, SourceSpan

logger = structlog.get_logger(__name__)


def build_hetero_data(
    requirements: list[Requirement],
    spans: list[SourceSpan],
    edges: list[GraphEdge],
    embedding_dim: int = 3072,
) -> Any:
    """Build a PyTorch Geometric HeteroData object.

    Returns None if torch_geometric is not installed.
    """
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError:
        logger.warning("graph_ml.torch_geometric_not_installed")
        return None

    data = HeteroData()

    # ── Requirement nodes ───────────────────────────────────────────
    req_ids = [r.id for r in requirements]
    req_id_to_idx = {rid: i for i, rid in enumerate(req_ids)}

    req_features = []
    for req in requirements:
        if req.embedding:
            req_features.append(req.embedding)
        else:
            req_features.append([0.0] * embedding_dim)

    if req_features:
        data["requirement"].x = torch.tensor(np.array(req_features), dtype=torch.float32)
        data["requirement"].node_ids = req_ids

    # ── Source span nodes ───────────────────────────────────────────
    span_ids = [s.id for s in spans]
    span_id_to_idx = {sid: i for i, sid in enumerate(span_ids)}

    span_features = []
    for span in spans:
        if span.embedding:
            span_features.append(span.embedding)
        else:
            span_features.append([0.0] * embedding_dim)

    if span_features:
        data["source_span"].x = torch.tensor(np.array(span_features), dtype=torch.float32)
        data["source_span"].node_ids = span_ids

    # ── Edges ───────────────────────────────────────────────────────
    edge_index_by_type: dict[str, list[list[int]]] = {}

    for edge in edges:
        src_idx = req_id_to_idx.get(edge.source_node_id)
        tgt_idx = req_id_to_idx.get(edge.target_node_id)

        if src_idx is None or tgt_idx is None:
            # Try span nodes for derived_from edges
            if edge.edge_type.value == "derived_from":
                src_idx = req_id_to_idx.get(edge.source_node_id)
                tgt_idx = span_id_to_idx.get(edge.target_node_id)
                if src_idx is not None and tgt_idx is not None:
                    key = ("requirement", "derived_from", "source_span")
                    edge_index_by_type.setdefault(str(key), [[], []])
                    edge_index_by_type[str(key)][0].append(src_idx)
                    edge_index_by_type[str(key)][1].append(tgt_idx)
            continue

        key = ("requirement", edge.edge_type.value, "requirement")
        edge_index_by_type.setdefault(str(key), [[], []])
        edge_index_by_type[str(key)][0].append(src_idx)
        edge_index_by_type[str(key)][1].append(tgt_idx)

    for key_str, indices in edge_index_by_type.items():
        # Parse the key tuple back
        key = eval(key_str)  # noqa: S307 – safe, we built the string
        data[key].edge_index = torch.tensor(indices, dtype=torch.long)

    logger.info(
        "dataset_builder.done",
        req_nodes=len(req_ids),
        span_nodes=len(span_ids),
        edge_types=len(edge_index_by_type),
    )
    return data
