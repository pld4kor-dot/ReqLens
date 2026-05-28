"""R-GCN model for typed edge prediction on the requirement graph.

Placeholder – to be implemented in Milestone K when enough
reviewed edge labels have been collected.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch_geometric.nn import RGCNConv

    class RequirementRGCN(nn.Module):
        """Relational Graph Convolutional Network for edge prediction."""

        def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            num_relations: int,
        ) -> None:
            super().__init__()
            self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations)
            self.conv2 = RGCNConv(hidden_channels, out_channels, num_relations)

        def forward(self, x, edge_index, edge_type):
            x = self.conv1(x, edge_index, edge_type).relu()
            x = self.conv2(x, edge_index, edge_type)
            return x

except ImportError:
    logger.debug("rgcn.torch_geometric_not_available")
    RequirementRGCN = None  # type: ignore[assignment, misc]
