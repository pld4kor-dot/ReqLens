"""GNN training script (placeholder for R-GCN / HGT training)."""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


def train_edge_predictor(hetero_data, *, epochs: int = 50, lr: float = 0.01):
    """Train an edge-type prediction GNN.

    Requires torch_geometric. This is a placeholder that will be
    filled in Milestone K.
    """
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.nn import SAGEConv
    except ImportError:
        logger.warning("train_gnn.torch_geometric_not_installed")
        return None

    logger.info("train_gnn.starting", epochs=epochs)

    # TODO: Implement R-GCN or HGT training loop
    # Steps:
    #   1. Create train/val/test edge splits
    #   2. Build model (see models/rgcn.py or models/hgt.py)
    #   3. Train with link prediction loss
    #   4. Evaluate on held-out edges
    #   5. Return trained model and metrics

    logger.info("train_gnn.placeholder", message="GNN training not yet implemented")
    return None
