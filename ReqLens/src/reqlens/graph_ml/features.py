"""Feature engineering for GNN node and edge attributes."""

from __future__ import annotations

import numpy as np

from reqlens.domain.enums import RequirementKind, NFRSubtype, ReviewStatus
from reqlens.domain.models import GraphEdge, Requirement


def requirement_metadata_features(req: Requirement) -> list[float]:
    """Build a metadata feature vector for a requirement node.

    Features:
      - One-hot for requirement kind (5 dims)
      - One-hot for NFR subtype (12 dims)
      - Quality score (1 dim, 0 if None)
      - Review status encoding (1 dim)
    Total: 19 dims
    """
    # Kind one-hot
    kinds = list(RequirementKind)
    kind_vec = [1.0 if req.kind == k else 0.0 for k in kinds]

    # NFR subtype one-hot
    subtypes = list(NFRSubtype)
    subtype_vec = [1.0 if req.nfr_subtype == s else 0.0 for s in subtypes]

    # Quality score
    quality = [req.quality_score if req.quality_score is not None else 0.0]

    # Review status as numeric
    status_map = {
        ReviewStatus.pending: 0.0,
        ReviewStatus.accepted: 1.0,
        ReviewStatus.rejected: -1.0,
        ReviewStatus.needs_revision: 0.5,
        ReviewStatus.deferred: 0.25,
    }
    status_val = [status_map.get(req.review_status, 0.0)]

    return kind_vec + subtype_vec + quality + status_val


def edge_features(edge: GraphEdge, embedding_similarity: float = 0.0) -> list[float]:
    """Build a feature vector for a graph edge.

    Features:
      - Confidence (1 dim)
      - Embedding similarity (1 dim)
      - Review status encoding (1 dim)
    Total: 3 dims
    """
    status_map = {
        "pending": 0.0,
        "accepted": 1.0,
        "rejected": -1.0,
    }
    return [
        edge.confidence,
        embedding_similarity,
        status_map.get(edge.review_status.value if hasattr(edge.review_status, "value") else str(edge.review_status), 0.0),
    ]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))
