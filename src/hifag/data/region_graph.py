"""Coarse region-graph data pipeline for HiFAG.

Wraps AFGNN's ``DVlogFaceDataset`` so that every sample additionally carries
``coarse_x``: the (9 * T, FEATURE_DIM) hand-crafted region descriptors computed
from the *same* sampled (and possibly augmented) frames that the fine face
graph was built from.

Design note: the coarse graph's edges are NOT stored per sample. They are
identical for all samples given (num_frames, edge_mode), so ``RegionGNN``
builds the batched edge_index internally from the batch vector — the same
pattern AFGNN's ``AudioGNN`` uses for its chain edges. PyG's ``Batch``
collates ``coarse_x`` by plain concatenation along dim 0, which is exactly
what we need (nodes are frame-major: node index = t * 9 + region_id).
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch_geometric.loader import DataLoader

# AFGNN imports (available via hifag/paths.py).
from data.dvlog_face_dataset import DVlogFaceDataset
from models.graph_utils import compute_audio_norm_stats, sample_frames

from hifag.data.region_features import (
    NUM_REGIONS,
    compute_region_features,
    feature_dim,
    flat_to_coords,
)


def compute_coarse_norm_stats(
    visual: np.ndarray, num_frames: int, drop_groups=()
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute train-set mean/std for coarse region descriptors.

    Args:
        visual: (N, T_full, 136) raw landmark sequences (train split).
        num_frames: frames sampled per graph (must match the dataset).
        drop_groups: feature groups to drop (must match the dataset).

    Returns:
        mean: (feature_dim(drop_groups),) averaged over samples, frames, regions.
        std: (feature_dim(drop_groups),) with a small floor to avoid division by zero.
    """
    feats = []
    for seq in visual:
        sampled = sample_frames(seq, num_frames)
        # OpenFace layout is [x_0..x_67, y_0..y_67]; do NOT reshape(68, 2).
        coords = flat_to_coords(sampled)
        feats.append(compute_region_features(coords, drop_groups=drop_groups))
    feats = np.stack(feats, axis=0)  # (N, T, NUM_REGIONS, FEATURE_DIM)
    mean = feats.mean(axis=(0, 1, 2))
    std = feats.std(axis=(0, 1, 2))
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


class HiFAGFaceDataset(DVlogFaceDataset):
    """AFGNN face dataset that also attaches coarse region descriptors.

    Adds one attribute to each returned ``Data`` object:
        coarse_x: (num_frames * NUM_REGIONS, FEATURE_DIM) float tensor,
                  frame-major (node = t * NUM_REGIONS + region_id).

    Extra args:
        coarse_normalize: standardize descriptors with train-set stats.
        coarse_norm_mean / coarse_norm_std: stats; required
            when coarse_normalize is True.
        coarse_drop_groups: feature groups to drop (ablations A4/A5);
            must match the stats and the model's coarse_in_channels.
        fix_coordinate_layout: re-pair node features from the mis-read
            interleaved layout to true (x, y) pairs (see __getitem__).
            Default True; only disable to reproduce pre-fix experiments.
    """

    def __init__(
        self,
        *args,
        coarse_normalize: bool = True,
        coarse_norm_mean: np.ndarray = None,
        coarse_norm_std: np.ndarray = None,
        coarse_drop_groups=(),
        fix_coordinate_layout: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if coarse_normalize and (coarse_norm_mean is None or coarse_norm_std is None):
            raise ValueError(
                "coarse_normalize=True requires coarse_norm_mean/coarse_norm_std"
            )
        self.coarse_normalize = coarse_normalize
        self.coarse_norm_mean = coarse_norm_mean
        self.coarse_norm_std = coarse_norm_std
        self.coarse_drop_groups = tuple(coarse_drop_groups)
        self.coarse_dim = feature_dim(self.coarse_drop_groups)
        self.fix_coordinate_layout = fix_coordinate_layout

    def _repair_node_coords(self, x: torch.Tensor) -> torch.Tensor:
        """Re-pair mis-read node coordinates into true (x, y) pairs.

        D-Vlog official features store each frame as [x_0..x_67, y_0..y_67]
        (OpenFace convention), but AFGNN's build_face_graph reads them with
        reshape(T, 68, 2), i.e. as interleaved pairs. As a result every
        fine node holds flat[2i] and flat[2i+1] — two x's or two y's of
        adjacent storage slots, not a true (x, y) coordinate (bug found
        2026-07-28). The full flat vector is recoverable from x[:, :2] by
        re-interleaving; true coords are then (flat[:68], flat[68:]).

        Velocity columns (dx, dy) are recomputed from the true coords; any
        remaining columns (region one-hot) are passed through untouched.
        NOTE: AFGNN's augmentation (when enabled) was applied to the
        mis-paired coords upstream and cannot be undone here — all HiFAG
        configs currently disable augmentation.
        """
        T = self.num_frames
        a = x[:, 0].reshape(T, 68).numpy()
        b = x[:, 1].reshape(T, 68).numpy()
        flat = np.empty((T, 136), dtype=np.float64)
        flat[:, 0::2] = a
        flat[:, 1::2] = b
        coords = np.stack([flat[:, :68], flat[:, 68:]], axis=-1)  # (T, 68, 2)

        cols = [coords]
        n_skip = 2
        if x.size(1) >= 4:  # [x, y, dx, dy, ...] -> recompute velocity
            velocity = np.zeros_like(coords)
            if T > 1:
                velocity[1:] = coords[1:] - coords[:-1]
            cols.append(velocity)
            n_skip = 4
        if x.size(1) > n_skip:  # region one-hot etc.
            cols.append(x[:, n_skip:].numpy().reshape(T, 68, -1))
        fixed = np.concatenate(cols, axis=-1).reshape(T * 68, -1)
        return torch.from_numpy(fixed).float()

    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        # data.x layout: [x, y, (dx, dy), (region one-hot...)] per landmark,
        # frame-major (node = t * 68 + landmark_id) — but with the mis-paired
        # coordinates described in _repair_node_coords.
        num_nodes = self.num_frames * 68
        assert data.x.size(0) == num_nodes, (
            f"Expected {num_nodes} fine nodes, got {data.x.size(0)}"
        )
        if self.fix_coordinate_layout:
            data.x = self._repair_node_coords(data.x)

        coords = data.x[:, 0:2].numpy().reshape(self.num_frames, 68, 2)

        feats = compute_region_features(coords, drop_groups=self.coarse_drop_groups)
        if self.coarse_normalize:
            feats = (feats - self.coarse_norm_mean) / self.coarse_norm_std

        data.coarse_x = torch.from_numpy(
            feats.reshape(self.num_frames * NUM_REGIONS, self.coarse_dim)
        ).float()
        return data


def get_hifag_loaders(
    data_dir: str,
    num_frames: int = 32,
    batch_size: int = 32,
    num_workers: int = 4,
    audio_num_frames: int = None,
    add_static_edges: bool = True,
    add_temporal_edges: bool = True,
    add_dynamic_edges: bool = False,
    dynamic_k: int = 3,
    dynamic_metric: str = "cosine",
    dynamic_edge_weight: float = 1.0,
    dynamic_feature: str = "full",
    dynamic_region_restricted: bool = False,
    use_edge_type: bool = False,
    add_region_onehot: bool = False,
    add_self_loops: bool = False,
    use_velocity: bool = True,
    region_ablation: Optional[int] = None,
    random_static_edges: bool = False,
    random_edge_seed: int = 0,
    augment: bool = False,
    rotation_range: Tuple[float, float] = (-10.0, 10.0),
    scale_range: Tuple[float, float] = (0.95, 1.05),
    translate_range: Tuple[float, float] = (-0.05, 0.05),
    noise_std: float = 0.02,
    temporal_mask_prob: float = 0.2,
    temporal_mask_max_ratio: float = 0.15,
    audio_use_delta: bool = False,
    audio_self_loops: bool = False,
    audio_skip: int = 0,
    use_audio: bool = True,
    coarse_normalize: bool = True,
    coarse_drop_groups=(),
    fix_coordinate_layout: bool = True,
    worker_init_fn=None,
):
    """Build train/valid/test DataLoaders with coarse region graphs attached.

    Mirrors AFGNN's ``get_dvlog_face_loaders`` but instantiates
    ``HiFAGFaceDataset`` and computes coarse descriptor normalization stats
    from the train split (analogous to the audio norm stats).
    """
    # Audio normalization stats from the training split (AFGNN convention).
    audio_mean, audio_std = None, None
    train_acoustic_path = os.path.join(data_dir, "train_acoustic.npy")
    if use_audio and os.path.exists(train_acoustic_path):
        train_acoustic = np.load(train_acoustic_path).astype(np.float32)
        audio_mean, audio_std = compute_audio_norm_stats(train_acoustic)

    # Coarse descriptor normalization stats from the training split.
    coarse_mean, coarse_std = None, None
    if coarse_normalize:
        train_visual_path = os.path.join(data_dir, "train_visual.npy")
        train_visual = np.load(train_visual_path).astype(np.float32)
        coarse_mean, coarse_std = compute_coarse_norm_stats(
            train_visual, num_frames, drop_groups=coarse_drop_groups
        )

    loaders = {}
    for split in ["train", "valid", "test"]:
        visual_path = os.path.join(data_dir, f"{split}_visual.npy")
        labels_path = os.path.join(data_dir, f"{split}_labels.npy")
        acoustic_path = os.path.join(data_dir, f"{split}_acoustic.npy")

        is_train = split == "train"

        dataset = HiFAGFaceDataset(
            visual_path=visual_path,
            labels_path=labels_path,
            acoustic_path=acoustic_path if use_audio else None,
            num_frames=num_frames,
            audio_num_frames=audio_num_frames,
            add_static_edges=add_static_edges,
            add_temporal_edges=add_temporal_edges,
            add_dynamic_edges=add_dynamic_edges,
            dynamic_k=dynamic_k,
            dynamic_metric=dynamic_metric,
            dynamic_edge_weight=dynamic_edge_weight,
            dynamic_feature=dynamic_feature,
            dynamic_region_restricted=dynamic_region_restricted,
            use_edge_type=use_edge_type,
            add_region_onehot=add_region_onehot,
            add_self_loops=add_self_loops,
            use_velocity=use_velocity,
            region_ablation=region_ablation,
            random_static_edges=random_static_edges,
            random_edge_seed=random_edge_seed,
            augment=(augment and is_train),
            rotation_range=rotation_range,
            scale_range=scale_range,
            translate_range=translate_range,
            noise_std=noise_std,
            temporal_mask_prob=temporal_mask_prob,
            temporal_mask_max_ratio=temporal_mask_max_ratio,
            audio_use_delta=audio_use_delta,
            audio_self_loops=audio_self_loops,
            audio_skip=audio_skip,
            audio_norm_mean=audio_mean,
            audio_norm_std=audio_std,
            coarse_normalize=coarse_normalize,
            coarse_norm_mean=coarse_mean,
            coarse_norm_std=coarse_std,
            coarse_drop_groups=coarse_drop_groups,
            fix_coordinate_layout=fix_coordinate_layout,
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=is_train,
            worker_init_fn=worker_init_fn,
        )

    return loaders
