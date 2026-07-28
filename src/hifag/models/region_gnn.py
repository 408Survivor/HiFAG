"""RegionGNN: GAT encoder for HiFAG's coarse 9-region facial graph.

The coarse graph has 9 region nodes per frame (node = t * 9 + region_id,
frame-major) and two kinds of edges:
  - spatial edges within each frame: anatomical adjacency (default) or full
    connectivity (config switch, ablation A6);
  - temporal edges: bidirectional chain per region across adjacent frames.

Edges are identical across samples, so the base edge_index is precomputed as
a buffer and simply offset per graph in the batch — the same pattern AFGNN's
AudioGNN uses for its chain edges.

The encoder reuses AFGNN's WeightedGATConv + AttentionalAggregation, mirroring
FaceGNN's structure (LayerNorm, ELU, dropout) at a smaller budget.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import AttentionalAggregation

from models.weighted_gat_conv import WeightedGATConv

from hifag.data.region_features import FEATURE_DIM, NUM_REGIONS

# Anatomical adjacency between regions (undirected pairs, region ids):
#   0 face_contour, 1 left_eyebrow, 2 right_eyebrow, 3 nose_bridge,
#   4 nose_bottom, 5 left_eye, 6 right_eye, 7 outer_mouth, 8 inner_mouth
# Per DESIGN.md: 眉-眼、眼-鼻、鼻-嘴、轮廓-各区域、外唇-内唇.
ANATOMICAL_PAIRS: List[Tuple[int, int]] = [
    # contour connects to every other region
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
    # brow-eye
    (1, 5), (2, 6),
    # eye-nose
    (5, 3), (6, 3),
    # nose bridge-bottom, nose-mouth
    (3, 4), (4, 7),
    # outer-inner mouth
    (7, 8),
]

EDGE_MODES = ("anatomical", "full")


def build_region_base_edges(
    num_frames: int,
    edge_mode: str = "anatomical",
    add_self_loops: bool = True,
) -> torch.Tensor:
    """Build the edge_index of a single coarse graph (9 * num_frames nodes).

    Args:
        num_frames: number of frames T.
        edge_mode: "anatomical" or "full" spatial connectivity per frame.
        add_self_loops: whether to add self-loop edges for every node.

    Returns:
        edge_index: (2, E) long tensor, frame-major node indexing.
    """
    if edge_mode not in EDGE_MODES:
        raise ValueError(f"Unsupported edge_mode: {edge_mode}. Choose from {EDGE_MODES}")

    if edge_mode == "full":
        pairs = [(a, b) for a in range(NUM_REGIONS) for b in range(NUM_REGIONS) if a != b]
    else:
        pairs = []
        for a, b in ANATOMICAL_PAIRS:
            pairs.append((a, b))
            pairs.append((b, a))

    edges = []
    # Spatial edges within each frame.
    for t in range(num_frames):
        offset = t * NUM_REGIONS
        for a, b in pairs:
            edges.append((offset + a, offset + b))

    # Temporal chain edges per region (bidirectional).
    for r in range(NUM_REGIONS):
        for t in range(num_frames - 1):
            cur = t * NUM_REGIONS + r
            nxt = (t + 1) * NUM_REGIONS + r
            edges.append((cur, nxt))
            edges.append((nxt, cur))

    if add_self_loops:
        for n in range(num_frames * NUM_REGIONS):
            edges.append((n, n))

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


class RegionGNN(nn.Module):
    """GAT encoder over the coarse 9-region spatio-temporal graph.

    Args:
        in_channels: node feature dim (FEATURE_DIM = 10).
        hidden_channels: hidden dimension.
        out_channels: output graph-level embedding dimension.
        num_layers: number of GAT layers.
        heads: number of attention heads.
        dropout: dropout probability.
        num_frames: frames per graph T (must match the dataset).
        edge_mode: "anatomical" or "full" spatial connectivity.
        add_self_loops: whether to add self-loop edges.
    """

    def __init__(
        self,
        in_channels: int = FEATURE_DIM,
        hidden_channels: int = 64,
        out_channels: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.5,
        num_frames: int = 32,
        edge_mode: str = "anatomical",
        add_self_loops: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.num_frames = num_frames
        self.edge_mode = edge_mode
        self.add_self_loops = add_self_loops
        self.num_nodes_per_graph = NUM_REGIONS * num_frames

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First GAT layer
        self.convs.append(
            WeightedGATConv(
                in_channels=in_channels,
                out_channels=hidden_channels // heads,
                heads=heads,
                concat=True,
                dropout=dropout,
            )
        )
        self.norms.append(nn.LayerNorm(hidden_channels))

        # Intermediate GAT layers
        for _ in range(num_layers - 2):
            self.convs.append(
                WeightedGATConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_channels))

        # Last GAT layer
        self.convs.append(
            WeightedGATConv(
                in_channels=hidden_channels,
                out_channels=out_channels,
                heads=1,
                dropout=dropout,
                concat=False,
            )
        )

        # AttentionalAggregation readout (same gate pattern as FaceGNN)
        self.readout_gate = nn.Sequential(
            nn.Linear(out_channels, out_channels // 2),
            nn.ReLU(),
            nn.Linear(out_channels // 2, 1),
        )
        self.readout = AttentionalAggregation(gate_nn=self.readout_gate)

        # Base edges are fixed given (num_frames, edge_mode); offset per graph
        # in the batch at forward time. Not a parameter, not in state_dict.
        self.register_buffer(
            "base_edge_index",
            build_region_base_edges(num_frames, edge_mode, add_self_loops),
            persistent=False,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def build_batched_edges(self, num_graphs: int) -> torch.Tensor:
        """Replicate base edges for each graph with node-index offsets."""
        base = self.base_edge_index.unsqueeze(0).repeat(num_graphs, 1, 1)
        offsets = (
            torch.arange(num_graphs, device=self.base_edge_index.device)
            .view(-1, 1, 1)
            * self.num_nodes_per_graph
        )
        batched = base + offsets
        return batched.permute(1, 0, 2).reshape(2, -1)

    def forward(
        self, x: torch.Tensor, batch: torch.Tensor, return_nodes: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: coarse node features, (num_graphs * 9 * T, in_channels).
            batch: batch vector assigning coarse nodes to graphs.
            return_nodes: if True, also return the per-node embeddings from
                before the readout (used by the coarse->fine FiLM modulation).

        Returns:
            graph_embedding: (num_graphs, out_channels).
            node_embeddings (only if return_nodes): (num_nodes, out_channels).
        """
        # Feature contract: fail loudly instead of silently truncating
        # (SFAF region one-hot lesson).
        assert x.size(-1) == self.in_channels, (
            f"Coarse node feature dim mismatch: expected {self.in_channels}, "
            f"got {x.size(-1)}"
        )
        num_graphs = int(batch.max().item()) + 1
        assert x.size(0) == num_graphs * self.num_nodes_per_graph, (
            f"Expected {num_graphs * self.num_nodes_per_graph} coarse nodes "
            f"({num_graphs} graphs x {self.num_nodes_per_graph}), got {x.size(0)}"
        )

        edge_index = self.build_batched_edges(num_graphs)

        for i, conv in enumerate(self.convs):
            if i < self.num_layers - 1:
                x = conv(x, edge_index)
                x = self.norms[i](x)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = conv(x, edge_index)

        graph_emb = self.readout(x, batch)  # (num_graphs, out_channels)
        if return_nodes:
            return graph_emb, x
        return graph_emb
