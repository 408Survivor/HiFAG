"""HiFAG: Hierarchical Facial-Audio Graph Network.

Combines up to three branches with per-branch config switches:
  - fine:   AFGNN FaceGNN over the 68-landmark spatio-temporal graph (reused);
  - coarse: RegionGNN over the 9-region graph with hand-crafted descriptors;
  - audio:  AFGNN AudioGNN over the temporal acoustic chain (reused).

Stage-1 fusion is plain concat -> MLP by default; setting
`fusion_type="cross_attention"` fuses the face side (fine+coarse concat) with
the audio embedding via AFGNN's CrossModalAttentionFusion instead.

Stage-2 hierarchical interaction (DESIGN.md 3.5): with
`hierarchical="coarse_to_fine_film"`, the coarse branch runs first and its
per-node embeddings FiLM-modulate the fine branch's geometric input features
([x, y, dx, dy]; the region one-hot is left untouched). The FiLM layers are
zero-initialized so training starts from the unmodulated model.

Follows the AFGNN model interface: forward(data) -> (batch_size, 1) logit, so
AFGNN's trainer (train_model / evaluate) works unchanged.
"""

import torch
import torch.nn as nn

from models.audio_gnn import AudioGNN
from models.face_gnn import FaceGNN
from models.fusion import CrossModalAttentionFusion

from hifag.data.region_features import FEATURE_DIM, NUM_REGIONS, REGION_GROUPS
from hifag.models.region_gnn import RegionGNN

NUM_LANDMARKS = 68
HIERARCHICAL_MODES = ("none", "coarse_to_fine_film")


def build_landmark_region_index() -> torch.Tensor:
    """(68,) long tensor mapping each landmark to its region id (0..8)."""
    index = torch.empty(NUM_LANDMARKS, dtype=torch.long)
    for region_id, indices in enumerate(REGION_GROUPS.values()):
        index[torch.tensor(indices, dtype=torch.long)] = region_id
    return index


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
        fusion_type: "concat" (default) or "cross_attention" (face side vs
            audio, requires use_audio and at least one face branch).
        fusion_hidden_dim: attention hidden dim for cross_attention fusion.
        hierarchical: "none" (default) or "coarse_to_fine_film" (stage 2:
            coarse per-node embeddings FiLM-modulate the fine geometric
            input; requires use_fine and use_coarse).
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
        # Fusion
        fusion_type: str = "concat",
        fusion_hidden_dim: int = 64,
        # Hierarchical interaction (stage 2)
        hierarchical: str = "none",
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

        # Cross-modal attention between the face side (fine+coarse concat) and
        # audio. Output dims match concat, so classifier_in is unchanged.
        face_dim = classifier_in - (audio_out_channels if use_audio else 0)
        if fusion_type == "cross_attention":
            if not (use_audio and face_dim > 0):
                raise ValueError(
                    "fusion_type='cross_attention' requires use_audio=True and "
                    "at least one of use_fine/use_coarse."
                )
            self.fusion = CrossModalAttentionFusion(
                face_dim, audio_out_channels, fusion_hidden_dim
            )
        elif fusion_type == "concat":
            self.fusion = None
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
        self.fusion_type = fusion_type

        # Stage-2 hierarchical interaction: coarse per-node embeddings
        # FiLM-modulate the fine branch's geometric input [x, y, dx, dy].
        # Zero-initialized -> training starts from the unmodulated model.
        if hierarchical not in HIERARCHICAL_MODES:
            raise ValueError(
                f"Unsupported hierarchical mode: {hierarchical}. "
                f"Choose from {HIERARCHICAL_MODES}"
            )
        if hierarchical == "coarse_to_fine_film":
            if not (use_fine and use_coarse):
                raise ValueError(
                    "hierarchical='coarse_to_fine_film' requires "
                    "use_fine=True and use_coarse=True."
                )
            self.film_gamma = nn.Linear(coarse_out_channels, 4)
            self.film_beta = nn.Linear(coarse_out_channels, 4)
            nn.init.zeros_(self.film_gamma.weight)
            nn.init.zeros_(self.film_gamma.bias)
            nn.init.zeros_(self.film_beta.weight)
            nn.init.zeros_(self.film_beta.bias)
            self.register_buffer(
                "landmark_region",
                build_landmark_region_index(),
                persistent=False,
            )
        else:
            self.film_gamma = None
            self.film_beta = None
        self.hierarchical = hierarchical

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

    def _film_modulate(
        self, x: torch.Tensor, batch: torch.Tensor, coarse_nodes: torch.Tensor
    ) -> torch.Tensor:
        """FiLM-modulate the fine geometric input [x, y, dx, dy] with the
        coarse node embedding of the node's own region at the same frame.

        Fine nodes are frame-major (t * 68 + landmark); coarse nodes are
        frame-major (t * 9 + region). The landmark->region map is a fixed
        buffer; the modulation only touches the first 4 feature dims.
        """
        counts = torch.bincount(batch)
        ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        within = torch.arange(batch.size(0), device=x.device) - ptr[batch]
        t = within // NUM_LANDMARKS
        landmark = within % NUM_LANDMARKS
        region = self.landmark_region[landmark]
        coarse_index = (
            batch * self.coarse_nodes_per_graph + t * NUM_REGIONS + region
        )
        h_c = coarse_nodes[coarse_index]  # (N, coarse_out_channels)
        gamma = self.film_gamma(h_c)
        beta = self.film_beta(h_c)
        # No in-place writes: autograd-safe modulation of the geometric dims.
        geo = x[:, :4] * (1.0 + gamma) + beta
        return torch.cat([geo, x[:, 4:]], dim=-1)

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
        # Branch outputs are stored by name and concatenated in the fixed
        # order [fine, coarse, audio], regardless of execution order (the
        # coarse branch must run first when FiLM modulation is active).
        branch_embs = {}

        # Coarse branch runs first: with coarse_to_fine_film its per-node
        # embeddings are needed to modulate the fine input.
        coarse_nodes = None
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
            if self.hierarchical == "coarse_to_fine_film":
                h_coarse, coarse_nodes = self.region_gnn(
                    coarse_x, coarse_batch, return_nodes=True
                )
            else:
                h_coarse = self.region_gnn(coarse_x, coarse_batch)
            branch_embs["coarse"] = h_coarse

        if self.use_fine:
            x = data.x
            # Feature contract: fail loudly on dim mismatch (SFAF lesson).
            assert x.size(-1) == self.face_in_channels, (
                f"Fine node feature dim mismatch: expected {self.face_in_channels}, "
                f"got {x.size(-1)}"
            )
            if coarse_nodes is not None:
                x = self._film_modulate(x, data.batch, coarse_nodes)
            h_fine = self.face_gnn(
                x,
                data.edge_index,
                data.batch,
                edge_weight=getattr(data, "edge_weight", None),
                edge_type=getattr(data, "edge_type", None),
            )
            branch_embs["fine"] = h_fine

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

        # Fixed concat order: [fine, coarse, audio].
        for name in ("fine", "coarse"):
            if name in branch_embs:
                embeddings.append(branch_embs[name])

        if self.fusion is not None:
            # Face side = fine+coarse concat; attend face<->audio, then concat.
            h_face = torch.cat(embeddings, dim=-1)
            h = self.fusion(h_face, h_audio)
        else:
            if self.use_audio:
                embeddings.append(h_audio)
            h = torch.cat(embeddings, dim=-1)
        return self.classifier(h)
