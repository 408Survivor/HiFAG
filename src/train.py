#!/usr/bin/env python3
"""
Training script for HiFAG.

Usage:
    conda activate DVlog
    cd /home/ltq/DepressionCode/DepGNN/HiFAG
    python src/train.py --config experiments/configs/hifag_a3_fine_coarse.yaml
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Path setup: makes `hifag` importable and exposes AFGNN's src/ for reuse.
# (src/ is already on sys.path when running `python src/train.py`; importing
# hifag.paths appends AFGNN's src/.)
# ---------------------------------------------------------------------------
from hifag import paths  # noqa: F401

import numpy as np
import torch

from hifag.utils.builders import build_loaders, build_model, load_config
from hifag.utils.experiment import get_next_exp_dir, save_run_info

# Reuse AFGNN training utilities.
from utils.losses import FocalLoss, compute_class_weights
from utils.run_context import start_run
from utils.trainer import train_model


def build_criterion(cfg, train_labels_path):
    """Build loss criterion based on config."""
    loss_type = cfg["training"].get("loss", "bce")

    if loss_type == "focal":
        alpha = cfg["training"].get("focal_alpha", 0.25)
        gamma = cfg["training"].get("focal_gamma", 2.0)
        print(f"[Loss] FocalLoss(alpha={alpha}, gamma={gamma})")
        return FocalLoss(alpha=alpha, gamma=gamma)

    elif loss_type == "weighted_bce":
        labels = np.load(train_labels_path)
        weights = compute_class_weights(torch.from_numpy(labels))
        pos_weight = weights[1] / weights[0]
        print(f"[Loss] Weighted BCE (pos_weight={pos_weight:.4f})")
        return torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float)
        )

    else:
        print("[Loss] Standard BCEWithLogitsLoss")
        return torch.nn.BCEWithLogitsLoss()


def build_scheduler(optimizer, cfg):
    """Build optional learning rate scheduler."""
    scheduler_cfg = cfg["training"].get("scheduler", None)
    if scheduler_cfg is None:
        return None

    sched_type = scheduler_cfg.get("type", "plateau")
    if sched_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=scheduler_cfg.get("factor", 0.5),
            patience=scheduler_cfg.get("patience", 5),
            verbose=True,
        )
        print(
            f"[Scheduler] ReduceLROnPlateau(factor={scheduler_cfg.get('factor', 0.5)}, "
            f"patience={scheduler_cfg.get('patience', 5)})"
        )
        return scheduler
    elif sched_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_cfg.get("step_size", 20),
            gamma=scheduler_cfg.get("gamma", 0.5),
        )
        print(
            f"[Scheduler] StepLR(step_size={scheduler_cfg.get('step_size', 20)}, "
            f"gamma={scheduler_cfg.get('gamma', 0.5)})"
        )
        return scheduler
    elif sched_type == "cosine":
        T_max = scheduler_cfg.get("T_max", cfg["training"]["epochs"])
        eta_min = scheduler_cfg.get("eta_min", 0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min,
        )
        print(f"[Scheduler] CosineAnnealingLR(T_max={T_max}, eta_min={eta_min})")
        return scheduler
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_type}")


def main():
    parser = argparse.ArgumentParser(description="Train HiFAG")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/hifag_a3_fine_coarse.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save training history (defaults to the experiment folder)",
    )
    parser.add_argument(
        "--exp_base_dir",
        type=str,
        default="experiments",
        help="Base directory for auto-incremented experiment folders (exp_1, exp_2, ...)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible training",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["_config_path"] = args.config

    exp_dir = get_next_exp_dir(args.exp_base_dir)
    exp_name = exp_dir.name
    print("=" * 60)
    print(f"[Experiment] Start {exp_name}")
    print(f"  Directory: {exp_dir}")
    print("=" * 60)

    # Override checkpoint and output paths so everything lives in exp_dir.
    cfg["training"]["checkpoint_path"] = str(exp_dir / "best.pt")
    args.output_dir = args.output_dir if args.output_dir is not None else str(exp_dir)

    run = start_run(cfg, "train", seed=args.seed)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"[Seed] Using random seed: {args.seed}")

    def worker_init_fn(worker_id):
        if args.seed is not None:
            np.random.seed(args.seed + worker_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = cfg["data"]["processed_dir"]
    loaders = build_loaders(
        cfg,
        augment=cfg.get("augmentation", {}).get("enabled", False),
        worker_init_fn=worker_init_fn,
    )

    model = build_model(cfg, device)

    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    # Persist config and model summary in the experiment folder.
    save_run_info(exp_dir, cfg, model, command=" ".join(sys.argv))
    print(f"[Experiment] Saved config and model summary to: {exp_dir}")

    optimizer_type = cfg["training"].get("optimizer", "adam").lower()
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
        )
        print(f"[Optimizer] Adam(lr={cfg['training']['lr']}, weight_decay={cfg['training']['weight_decay']})")
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
        )
        print(f"[Optimizer] AdamW(lr={cfg['training']['lr']}, weight_decay={cfg['training']['weight_decay']})")
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    train_labels_path = os.path.join(data_dir, "train_labels.npy")
    criterion = build_criterion(cfg, train_labels_path).to(device)

    scheduler = build_scheduler(optimizer, cfg)

    grad_clip = cfg["training"].get("grad_clip", None)
    if grad_clip is not None:
        print(f"[Training] Gradient clipping (max_norm={grad_clip})")

    save_path = cfg["training"]["checkpoint_path"]
    if args.seed is not None:
        base, ext = os.path.splitext(save_path)
        save_path = f"{base}_seed{args.seed}{ext}"
        print(f"[Checkpoint] Seed-specific checkpoint: {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    history = train_model(
        model=model,
        train_loader=loaders["train"],
        valid_loader=loaders["valid"],
        optimizer=optimizer,
        device=device,
        criterion=criterion,
        num_epochs=cfg["training"]["epochs"],
        patience=cfg["training"]["patience"],
        save_path=save_path,
        scheduler=scheduler,
        grad_clip=grad_clip,
        early_stopping_metric=cfg["training"].get("early_stopping_metric", "auc"),
        config=cfg,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    history_path = os.path.join(args.output_dir, "training_history.json")
    if args.seed is not None:
        base, ext = os.path.splitext(history_path)
        history_path = f"{base}_seed{args.seed}{ext}"
    history_record = {
        "config": args.config,
        "checkpoint": save_path,
        "timestamp": datetime.now().isoformat(),
        "history": [
            {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in row.items()}
            for row in history
        ],
    }
    with open(history_path, "w") as f:
        json.dump(history_record, f, indent=2)
    print(f"Training history saved to: {history_path}")

    print("\n" + "=" * 60)
    print(f"[Experiment] End {exp_name}")
    print("=" * 60)
    print(f"Best checkpoint saved to: {save_path}")
    print(f"Experiment folder: {exp_dir}")
    print(f"To evaluate, run:")
    print(f"  python src/test.py --config {args.config} --checkpoint {save_path} --split test")

    run.finalize(checkpoint_path=save_path, history=history, extra={"exp_dir": str(exp_dir)})


if __name__ == "__main__":
    main()
