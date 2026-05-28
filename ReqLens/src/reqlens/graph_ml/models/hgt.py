"""Heterogeneous Graph Transformer for the full multi-type graph.

Placeholder – to be implemented in Milestone K.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch_geometric.nn import HGTConv, Linear

    class RequirementHGT(nn.Module):
        """Heterogeneous Graph Transformer for requirement graphs."""

        def __init__(
            self,
            hidden_channels: int,
            out_channels: int,
            num_heads: int,
            num_layers: int,
            metadata: tuple,  # (node_types, edge_types)
        ) -> None:
            super().__init__()
            self.lin_dict = nn.ModuleDict()
            for node_type in metadata[0]:
                self.lin_dict[node_type] = Linear(-1, hidden_channels)

            self.convs = nn.ModuleList()
            for _ in range(num_layers):
                conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
                self.convs.append(conv)

            self.lin_out = Linear(hidden_channels, out_channels)

        def forward(self, x_dict, edge_index_dict):
            x_dict = {
                key: self.lin_dict[key](x).relu()
                for key, x in x_dict.items()
            }
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)
            # Return requirement node embeddings
            return {key: self.lin_out(x) for key, x in x_dict.items()}

except ImportError:
    logger.debug("hgt.torch_geometric_not_available")
    RequirementHGT = None  # type: ignore[assignment, misc]
