"""Experiment directory management for HiFAG.

Provides helpers to create incremental experiment folders (exp_1, exp_2, ...)
and persist run metadata (config, model summary) so each run is self-contained.

Ported from SFAF src/sfaf/utils/experiment.py.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


_EXP_DIR_PATTERN = re.compile(r"^exp_(\d+)$")


def get_next_exp_dir(base_dir: str = "experiments") -> Path:
    """Return the next available experiment directory, e.g. experiments/exp_3.

    The index is determined by scanning existing ``exp_N`` directories under
    ``base_dir`` and picking max(N) + 1. The directory is created if it does
    not exist.
    """
    base = Path(base_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)

    max_idx = 0
    for entry in base.iterdir():
        if entry.is_dir():
            match = _EXP_DIR_PATTERN.match(entry.name)
            if match:
                max_idx = max(max_idx, int(match.group(1)))

    next_idx = max_idx + 1
    exp_dir = base / f"exp_{next_idx}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def get_latest_exp_dir(base_dir: str = "experiments") -> Optional[Path]:
    """Return the most recent experiment directory, or None if none exists."""
    base = Path(base_dir).resolve()
    if not base.exists():
        return None

    max_idx = 0
    latest = None
    for entry in base.iterdir():
        if entry.is_dir():
            match = _EXP_DIR_PATTERN.match(entry.name)
            if match:
                idx = int(match.group(1))
                if idx > max_idx:
                    max_idx = idx
                    latest = entry
    return latest


def infer_exp_dir_from_checkpoint(checkpoint_path: str) -> Optional[Path]:
    """Infer the experiment directory from a checkpoint path.

    If the checkpoint resides under an ``exp_N`` folder, that folder is
    returned. Otherwise None.
    """
    path = Path(checkpoint_path).resolve()
    for parent in path.parents:
        if _EXP_DIR_PATTERN.match(parent.name):
            return parent
    return None


def save_config(exp_dir: Path, cfg: Dict[str, Any], filename: str = "config.yaml") -> Path:
    """Save the run config as YAML inside the experiment directory."""
    out = exp_dir / filename
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return out


def save_model_summary(
    exp_dir: Path,
    model,
    filename: str = "model_summary.txt",
) -> Path:
    """Save model architecture string and parameter count."""
    out = exp_dir / filename
    lines = [
        "HiFAG Model Summary",
        "=" * 60,
        f"Created at: {datetime.now().isoformat()}",
        "",
        str(model),
        "",
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
        "",
    ]
    with open(out, "w") as f:
        f.write("\n".join(lines))
    return out


def save_run_info(
    exp_dir: Path,
    cfg: Dict[str, Any],
    model,
    command: Optional[str] = None,
) -> None:
    """Persist config, model summary, and an optional command line string."""
    save_config(exp_dir, cfg)
    save_model_summary(exp_dir, model)

    info_path = exp_dir / "run_info.txt"
    with open(info_path, "w") as f:
        f.write(f"Start time: {datetime.now().isoformat()}\n")
        if command:
            f.write(f"Command: {command}\n")
        f.write(f"Config file: {cfg.get('_config_path', 'N/A')}\n")
