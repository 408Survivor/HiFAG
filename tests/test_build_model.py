"""Smoke tests for HiFAG: data pipeline + model forward + feature contract.

Runs fully on synthetic data (no real D-Vlog files needed). Must pass before
any real training run (project engineering principle 6).
"""

import os
import sys

import numpy as np
import pytest
import torch

# Make `hifag` importable and expose AFGNN's src/ (same as entry scripts).
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)
import hifag.paths  # noqa: F401, E402

from hifag.data.region_features import FEATURE_DIM, NUM_REGIONS, compute_region_features
from hifag.utils.builders import build_loaders, build_model

NUM_FRAMES = 8
NUM_LANDMARKS = 68
AUDIO_DIM = 25
BATCH_SIZE = 4


def _make_split(data_dir, split, n_samples, rng):
    visual = rng.standard_normal((n_samples, 600, NUM_LANDMARKS * 2)).astype(np.float32)
    labels = (rng.random(n_samples) > 0.5).astype(np.int64)
    acoustic = rng.standard_normal((n_samples, 600, AUDIO_DIM)).astype(np.float32)
    np.save(os.path.join(data_dir, f"{split}_visual.npy"), visual)
    np.save(os.path.join(data_dir, f"{split}_labels.npy"), labels)
    np.save(os.path.join(data_dir, f"{split}_acoustic.npy"), acoustic)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("dvlog_synth")
    rng = np.random.default_rng(0)
    _make_split(str(d), "train", 8, rng)
    _make_split(str(d), "valid", 4, rng)
    _make_split(str(d), "test", 4, rng)
    return str(d)


def _base_cfg(data_dir, **model_overrides):
    model = {
        "use_fine": True,
        "use_coarse": True,
        "use_audio": False,
        "face_in_channels": 13,
        "face_hidden_channels": 32,
        "face_out_channels": 32,
        "face_num_layers": 2,
        "face_heads": 4,
        "coarse_in_channels": FEATURE_DIM,
        "coarse_hidden_channels": 32,
        "coarse_out_channels": 32,
        "coarse_num_layers": 2,
        "coarse_heads": 4,
        "coarse_edge_mode": "anatomical",
        "coarse_normalize": True,
        "audio_in_channels": AUDIO_DIM,
        "audio_hidden_channels": 32,
        "audio_out_channels": 32,
        "audio_num_layers": 2,
        "audio_heads": 4,
        "mlp_hidden": 32,
        "dropout": 0.5,
        "add_static_edges": True,
        "add_temporal_edges": True,
        "add_region_onehot": True,
        "add_self_loops": True,
    }
    model.update(model_overrides)
    return {
        "data": {"processed_dir": data_dir, "num_frames": NUM_FRAMES},
        "model": model,
        "training": {"batch_size": BATCH_SIZE, "num_workers": 0},
    }


def test_region_features_shape():
    rng = np.random.default_rng(1)
    coords = rng.standard_normal((NUM_FRAMES, NUM_LANDMARKS, 2))
    feats = compute_region_features(coords)
    assert feats.shape == (NUM_FRAMES, NUM_REGIONS, FEATURE_DIM)
    assert np.isfinite(feats).all()


def test_region_features_drop_groups():
    """A4/A5 ablation switch: dropping groups removes dims entirely."""
    rng = np.random.default_rng(2)
    coords = rng.standard_normal((NUM_FRAMES, NUM_LANDMARKS, 2))

    full = compute_region_features(coords)
    nosym = compute_region_features(coords, drop_groups=["symmetry"])
    geom = compute_region_features(coords, drop_groups=["motion", "symmetry"])

    assert nosym.shape == (NUM_FRAMES, NUM_REGIONS, 8)
    assert geom.shape == (NUM_FRAMES, NUM_REGIONS, 4)
    # Dims are removed, not zeroed: kept dims identical to the full descriptor.
    assert np.array_equal(nosym, full[:, :, :8])
    assert np.array_equal(geom, full[:, :, :4])

    with pytest.raises(ValueError, match="Unknown feature groups"):
        compute_region_features(coords, drop_groups=["bogus"])


def test_loaders_attach_coarse(data_dir):
    cfg = _base_cfg(data_dir)
    loaders = build_loaders(cfg, augment=False)
    batch = next(iter(loaders["train"]))

    expected_fine_dim = 4 + 9  # [x, y, dx, dy] + region one-hot
    assert batch.x.shape == (BATCH_SIZE * NUM_FRAMES * 68, expected_fine_dim)
    assert batch.coarse_x.shape == (
        BATCH_SIZE * NUM_FRAMES * NUM_REGIONS,
        FEATURE_DIM,
    )
    assert torch.isfinite(batch.coarse_x).all()
    assert batch.y.shape == (BATCH_SIZE,)


@pytest.mark.parametrize(
    "overrides",
    [
        {"use_fine": False, "use_coarse": True, "use_audio": False},   # A2
        {"use_fine": True, "use_coarse": True, "use_audio": False},    # A3
        {"use_fine": True, "use_coarse": True, "use_audio": True},     # A3 + audio
        {"use_fine": True, "use_coarse": False, "use_audio": False},   # fine only
    ],
)
def test_model_forward(data_dir, overrides):
    cfg = _base_cfg(data_dir, **overrides)
    loaders = build_loaders(cfg, augment=False)
    model = build_model(cfg)
    model.eval()

    batch = next(iter(loaders["valid"]))
    with torch.no_grad():
        logit = model(batch)
    assert logit.shape == (BATCH_SIZE, 1)
    assert torch.isfinite(logit).all()


def test_model_forward_full_edges(data_dir):
    """Ablation A6 switch: fully-connected coarse spatial edges."""
    cfg = _base_cfg(data_dir, coarse_edge_mode="full")
    loaders = build_loaders(cfg, augment=False)
    model = build_model(cfg)
    model.eval()

    batch = next(iter(loaders["valid"]))
    with torch.no_grad():
        logit = model(batch)
    assert logit.shape == (BATCH_SIZE, 1)


def test_model_forward_drop_groups(data_dir):
    """A4/A5: dataset, norm stats and model agree on the reduced coarse dim."""
    for drop, dim in [(["symmetry"], 8), (["motion", "symmetry"], 4)]:
        cfg = _base_cfg(
            data_dir,
            use_fine=False,
            use_coarse=True,
            use_audio=False,
            coarse_in_channels=dim,
            coarse_drop_groups=drop,
        )
        loaders = build_loaders(cfg, augment=False)
        model = build_model(cfg)
        model.eval()

        batch = next(iter(loaders["valid"]))
        assert batch.coarse_x.shape == (BATCH_SIZE * NUM_FRAMES * NUM_REGIONS, dim)
        with torch.no_grad():
            logit = model(batch)
        assert logit.shape == (BATCH_SIZE, 1)
        assert torch.isfinite(logit).all()


def test_model_forward_cross_attention(data_dir):
    """A8: cross-attention fusion between face side (fine+coarse) and audio."""
    cfg = _base_cfg(
        data_dir, use_audio=True, fusion_type="cross_attention"
    )
    loaders = build_loaders(cfg, augment=False)
    model = build_model(cfg)
    model.eval()

    batch = next(iter(loaders["valid"]))
    with torch.no_grad():
        logit = model(batch)
    assert logit.shape == (BATCH_SIZE, 1)
    assert torch.isfinite(logit).all()


def test_cross_attention_requires_audio_and_face(data_dir):
    """cross_attention without audio or without a face branch must raise."""
    cfg = _base_cfg(data_dir, use_audio=False, fusion_type="cross_attention")
    with pytest.raises(ValueError, match="cross_attention"):
        build_model(cfg)

    cfg = _base_cfg(
        data_dir,
        use_fine=False,
        use_coarse=False,
        use_audio=True,
        fusion_type="cross_attention",
    )
    with pytest.raises(ValueError, match="cross_attention"):
        build_model(cfg)


def test_model_forward_coarse_to_fine_film(data_dir):
    """A10: coarse per-node embeddings FiLM-modulate the fine input."""
    cfg = _base_cfg(
        data_dir,
        use_audio=True,
        fusion_type="cross_attention",
        hierarchical="coarse_to_fine_film",
    )
    loaders = build_loaders(cfg, augment=False)
    model = build_model(cfg)
    model.eval()

    batch = next(iter(loaders["valid"]))
    with torch.no_grad():
        logit = model(batch)
    assert logit.shape == (BATCH_SIZE, 1)
    assert torch.isfinite(logit).all()


def test_film_zero_init_matches_unmodulated(data_dir):
    """Zero-init FiLM must start exactly at the unmodulated model's output."""
    cfg_none = _base_cfg(data_dir, hierarchical="none")
    cfg_film = _base_cfg(data_dir, hierarchical="coarse_to_fine_film")
    loaders = build_loaders(cfg_none, augment=False)

    model_none = build_model(cfg_none)
    model_film = build_model(cfg_film)
    # Align shared weights; film layers stay zero-initialized.
    model_film.load_state_dict(model_none.state_dict(), strict=False)
    model_none.eval()
    model_film.eval()

    batch = next(iter(loaders["valid"]))
    with torch.no_grad():
        out_none = model_none(batch)
        out_film = model_film(batch)
    assert torch.allclose(out_none, out_film, atol=1e-6)


def test_film_requires_fine_and_coarse(data_dir):
    """coarse_to_fine_film without both face branches must raise."""
    cfg = _base_cfg(
        data_dir, use_coarse=False, hierarchical="coarse_to_fine_film"
    )
    with pytest.raises(ValueError, match="coarse_to_fine_film"):
        build_model(cfg)

    cfg = _base_cfg(
        data_dir, use_fine=False, hierarchical="coarse_to_fine_film"
    )
    with pytest.raises(ValueError, match="coarse_to_fine_film"):
        build_model(cfg)


def test_feature_contract_coarse_dim(data_dir):
    """Wrong coarse feature dim must raise, never silently truncate."""
    cfg = _base_cfg(data_dir, use_fine=False, use_coarse=True)
    loaders = build_loaders(cfg, augment=False)
    model = build_model(cfg)
    model.eval()

    batch = next(iter(loaders["valid"]))
    batch.coarse_x = batch.coarse_x[:, : FEATURE_DIM - 1]  # break the contract
    with pytest.raises(AssertionError, match="Coarse node feature dim mismatch"):
        with torch.no_grad():
            model(batch)


def test_training_step_runs(data_dir):
    """One optimizer step on synthetic data: loss must be finite."""
    cfg = _base_cfg(data_dir)
    loaders = build_loaders(cfg, augment=False)
    model = build_model(cfg)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    batch = next(iter(loaders["train"]))
    optimizer.zero_grad()
    logit = model(batch)
    loss = criterion(logit.squeeze(-1), batch.y.float())
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss).all()
