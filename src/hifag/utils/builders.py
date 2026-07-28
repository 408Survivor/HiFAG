"""Central factory helpers for building HiFAG models and D-Vlog data loaders.

The data loader factory lives in hifag.data.region_graph (it attaches the
coarse region graph on top of AFGNN's dataset); this module adds the model
builder and config loading, mirroring SFAF's builders.py.
"""

import yaml

from hifag.models.hifag import HiFAG

# Re-export so train.py / test.py have a single import point, as in SFAF.
from hifag.data.region_graph import get_hifag_loaders


def load_config(config_path: str) -> dict:
    """Load a YAML config file into a plain dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, device=None):
    """Construct a HiFAG model from a config dict."""
    m = cfg["model"]
    d = cfg.get("data", {})

    model = HiFAG(
        use_fine=m.get("use_fine", True),
        use_coarse=m.get("use_coarse", True),
        use_audio=m.get("use_audio", False),
        # Fine branch (AFGNN FaceGNN)
        face_in_channels=m.get("face_in_channels", 13),
        face_hidden_channels=m.get("face_hidden_channels", 128),
        face_out_channels=m.get("face_out_channels", 128),
        face_num_layers=m.get("face_num_layers", 3),
        face_heads=m.get("face_heads", 4),
        # Coarse branch (RegionGNN)
        coarse_in_channels=m.get("coarse_in_channels", 10),
        coarse_hidden_channels=m.get("coarse_hidden_channels", 64),
        coarse_out_channels=m.get("coarse_out_channels", 64),
        coarse_num_layers=m.get("coarse_num_layers", 2),
        coarse_heads=m.get("coarse_heads", 4),
        coarse_num_frames=m.get("coarse_num_frames", d.get("num_frames", 32)),
        coarse_edge_mode=m.get("coarse_edge_mode", "anatomical"),
        coarse_add_self_loops=m.get("coarse_add_self_loops", True),
        # Audio branch (AFGNN AudioGNN)
        audio_in_channels=m.get("audio_in_channels", 25),
        audio_hidden_channels=m.get("audio_hidden_channels", 64),
        audio_out_channels=m.get("audio_out_channels", 64),
        audio_num_layers=m.get("audio_num_layers", 2),
        audio_heads=m.get("audio_heads", 4),
        # Head
        mlp_hidden=m.get("mlp_hidden", 128),
        dropout=m.get("dropout", 0.5),
        num_edge_types=m.get("num_edge_types", None),
        edge_emb_dim=m.get("edge_emb_dim", 1),
        # Fusion
        fusion_type=m.get("fusion_type", "concat"),
        fusion_hidden_dim=m.get("fusion_hidden_dim", 64),
        # Hierarchical interaction (stage 2)
        hierarchical=m.get("hierarchical", "none"),
    )

    if device is not None:
        model = model.to(device)
    return model


def build_loaders(
    cfg: dict,
    *,
    data_dir: str = None,
    batch_size: int = None,
    num_workers: int = None,
    augment: bool = False,
    worker_init_fn=None,
):
    """Build train/valid/test DataLoaders with coarse region graphs attached."""
    d = cfg["data"]
    m = cfg["model"]
    aug = cfg.get("augmentation", {})

    if data_dir is None:
        data_dir = d["processed_dir"]
    if batch_size is None:
        batch_size = cfg["training"]["batch_size"]
    if num_workers is None:
        num_workers = cfg["training"]["num_workers"]

    return get_hifag_loaders(
        data_dir=data_dir,
        num_frames=d.get("num_frames", 32),
        audio_num_frames=d.get("audio_num_frames", d.get("num_frames", 32)),
        batch_size=batch_size,
        num_workers=num_workers,
        add_static_edges=m.get("add_static_edges", True),
        add_temporal_edges=m.get("add_temporal_edges", True),
        add_dynamic_edges=m.get("add_dynamic_edges", False),
        dynamic_k=m.get("dynamic_k", 3),
        dynamic_metric=m.get("dynamic_metric", "cosine"),
        dynamic_edge_weight=m.get("dynamic_edge_weight", 1.0),
        dynamic_feature=m.get("dynamic_feature", "full"),
        dynamic_region_restricted=m.get("dynamic_region_restricted", False),
        use_edge_type=m.get("use_edge_type", False),
        add_region_onehot=m.get("add_region_onehot", False),
        add_self_loops=m.get("add_self_loops", False),
        use_velocity=m.get("use_velocity", True),
        region_ablation=m.get("region_ablation", None),
        random_static_edges=m.get("random_static_edges", False),
        random_edge_seed=m.get("random_edge_seed", 0),
        augment=augment,
        rotation_range=aug.get("rotation_range", (-10.0, 10.0)),
        scale_range=aug.get("scale_range", (0.95, 1.05)),
        translate_range=aug.get("translate_range", (-0.05, 0.05)),
        noise_std=aug.get("noise_std", 0.02),
        temporal_mask_prob=aug.get("temporal_mask_prob", 0.2),
        temporal_mask_max_ratio=aug.get("temporal_mask_max_ratio", 0.15),
        audio_use_delta=m.get("audio_use_delta", False),
        audio_self_loops=m.get("audio_self_loops", False),
        audio_skip=m.get("audio_skip", 0),
        use_audio=m.get("use_audio", False),
        coarse_normalize=m.get("coarse_normalize", True),
        worker_init_fn=worker_init_fn,
    )
