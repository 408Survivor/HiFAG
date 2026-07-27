"""HiFAG: Hierarchical Facial-Audio Graph Network.

Combines up to three branches with per-branch config switches:
  - fine:   AFGNN FaceGNN over the 68-landmark spatio-temporal graph (reused);
  - coarse: RegionGNN over the 9-region graph with hand-crafted descriptors;
  - audio:  AFGNN AudioGNN over the temporal acoustic chain (reused).

Stage-1 fusion is plain concat -> MLP (hierarchical fine<->coarse message
passing is stage 2 and intentionally not implemented here).

Follows the AFGNN model interface: forward(data) -> (batch_size, 1) logit, so
AFGNN's trainer (train_model / evaluate) works unchanged.
"""

import torch
import torch.nn as nn

from models.audio_gnn import AudioGNN
from models.face_gnn import FaceGNN

from hifag.data.region_features import FEATURE_DIM, NUM_REGIONS
from hifag.models.region_gnn import RegionGNN


class HiFAG(nn.Module):
    """Hierarchical facial(-audio) graph model for depression recognition.

    Args:
        use_fine: whether to use the 68-landmark fine face branch.
        use_coarse: whether to use the 9-region coarse face branch.
        use_audio: whether to use the audio branch.
        face_*: fine branch (FaceGNN) hyperparameters.
        coarse_*: coarse branch (RegionGNN) hyperparameters.
        audio_*: audio branch (AudioGNN) hyperparameters.
        mlp_hidden: hidden dim of the classification MLP.
        dropout: dropout probability (shared by all branches and the head).
        num_edge_types / edge_emb_dim: optional edge type embedding for the
            fine branch (passed through to FaceGNN).
    """

    def __init__(
        self,
        use_fine: bool = True,
        use_coarse: bool = True,
        use_audio: bool = False,
        # Fine branch (AFGNN FaceGNN)
        face_in_channels: int = 13,
        face_hidden_channels: int = 128,
        face_out_channels: int = 128,
        face_num_layers: int = 3,
        face_heads: int = 4,
        # Coarse branch (RegionGNN)
        coarse_in_channels: int = FEATURE_DIM,
        coarse_hidden_channels: int = 64,
        coarse_out_channels: int = 64,
        coarse_num_layers: int = 2,
        coarse_heads: int = 4,
        coarse_num_frames: int = 32,
        coarse_edge_mode: str = "anatomical",
        coarse_add_self_loops: bool = True,
        # Audio branch (AFGNN AudioGNN)
        audio_in_channels: int = 25,
        audio_hidden_channels: int = 64,
        audio_out_channels: int = 64,
        audio_num_layers: int = 2,
        audio_heads: int = 4,
        # Head
        mlp_hidden: int = 128,
        dropout: float = 0.5,
        num_edge_types: int = None,
        edge_emb_dim: int = 1,
    ):
        super().__init__()
        if not (use_fine or use_coarse or use_audio):
            raise ValueError("At least one of use_fine/use_coarse/use_audio must be True.")

        self.use_fine = use_fine
        self.use_coarse = use_coarse
        self.use_audio = use_audio
        self.coarse_num_frames = coarse_num_frames
        self.coarse_nodes_per_graph = NUM_REGIONS * coarse_num_frames

        if use_fine:
            self.face_gnn = FaceGNN(
                in_channels=face_in_channels,
                hidden_channels=face_hidden_channels,
                out_channels=face_out_channels,
                num_layers=face_num_layers,
                heads=face_heads,
                dropout=dropout,
                num_edge_types=num_edge_types,
                edge_emb_dim=edge_emb_dim,
            )
            self.face_in_channels = face_in_channels
        else:
            self.face_gnn = None

        if use_coarse:
            self.region_gnn = RegionGNN(
                in_channels=coarse_in_channels,
                hidden_channels=coarse_hidden_channels,
                out_channels=coarse_out_channels,
                num_layers=coarse_num_layers,
                heads=coarse_heads,
                dropout=dropout,
                num_frames=coarse_num_frames,
                edge_mode=coarse_edge_mode,
                add_self_loops=coarse_add_self_loops,
            )
            self.coarse_in_channels = coarse_in_channels
        else:
            self.region_gnn = None

        if use_audio:
            self.audio_gnn = AudioGNN(
                in_channels=audio_in_channels,
                hidden_channels=audio_hidden_channels,
                out_channels=audio_out_channels,
                num_layers=audio_num_layers,
                heads=audio_heads,
                dropout=dropout,
            )
        else:
            self.audio_gnn = None

        classifier_in = 0
        if use_fine:
            classifier_in += face_out_channels
        if use_coarse:
            classifier_in += coarse_out_channels
        if use_audio:
            classifier_in += audio_out_channels

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_hidden, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _coarse_batch(self, coarse_x: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Derive the coarse batch vector (equal nodes per graph by construction)."""
        num_graphs = coarse_x.size(0) // self.coarse_nodes_per_graph
        return (
            torch.arange(num_graphs, device=device, dtype=torch.long)
            .repeat_interleave(self.coarse_nodes_per_graph)
        )

    def forward(self, data):
        """
        Args:
            data: torch_geometric Batch with fine graph attributes
                  (x, edge_index, batch, optional edge_weight / edge_type),
                  optional coarse_x and optional audio_x.

        Returns:
            logit: shape (batch_size, 1).
        """
        embeddings = []

        if self.use_fine:
            x = data.x
            # Feature contract: fail loudly on dim mismatch (SFAF lesson).
            assert x.size(-1) == self.face_in_channels, (
                f"Fine node feature dim mismatch: expected {self.face_in_channels}, "
                f"got {x.size(-1)}"
            )
            h_fine = self.face_gnn(
                x,
                data.edge_index,
                data.batch,
                edge_weight=getattr(data, "edge_weight", None),
                edge_type=getattr(data, "edge_type", None),
            )
            embeddings.append(h_fine)

        if self.use_coarse:
            coarse_x = getattr(data, "coarse_x", None)
            if coarse_x is None:
                raise ValueError(
                    "use_coarse=True but input Data has no coarse_x. "
                    "Make sure the dataset is HiFAGFaceDataset."
                )
            assert coarse_x.size(-1) == self.coarse_in_channels, (
                f"Coarse node feature dim mismatch: expected "
                f"{self.coarse_in_channels}, got {coarse_x.size(-1)}"
            )
            coarse_batch = self._coarse_batch(coarse_x, coarse_x.device)
            h_coarse = self.region_gnn(coarse_x, coarse_batch)
            embeddings.append(h_coarse)

        if self.use_audio:
            audio_x = getattr(data, "audio_x", None)
            if audio_x is None:
                if self.training:
                    raise ValueError(
                        "use_audio=True but input Data has no audio_x. "
                        "Make sure the dataset returns audio graphs."
                    )
                batch_size = data.y.size(0)
                h_audio = torch.zeros(
                    batch_size,
                    self.audio_gnn.out_channels,
                    device=data.y.device,
                    dtype=data.y.dtype,
                )
            else:
                audio_batch = getattr(data, "audio_batch", None)
                if audio_batch is None:
                    num_audio_nodes = audio_x.size(0)
                    num_graphs = int(data.batch.max().item()) + 1
                    num_frames = num_audio_nodes // num_graphs
                    audio_batch = torch.arange(
                        num_graphs, device=audio_x.device, dtype=torch.long
                    ).repeat_interleave(num_frames)
                    audio_batch = audio_batch[:num_audio_nodes]
                h_audio = self.audio_gnn(audio_x, batch=audio_batch)
            embeddings.append(h_audio)

        h = torch.cat(embeddings, dim=-1)
        return self.classifier(h)
