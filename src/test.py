#!/usr/bin/env python3
"""
Evaluation script for HiFAG.

Usage:
    conda activate DVlog
    cd /home/ltq/DepressionCode/DepGNN/HiFAG
    python src/test.py --config experiments/configs/hifag_a3_fine_coarse.yaml \
        --checkpoint experiments/exp_1/best.pt --split test
"""

import argparse
import json
import os
import sys
from datetime import datetime

# Path setup: makes `hifag` importable and exposes AFGNN's src/ for reuse.
from hifag import paths  # noqa: F401

import torch

from hifag.utils.builders import build_loaders, build_model, load_config
from hifag.utils.experiment import (
    get_latest_exp_dir,
    infer_exp_dir_from_checkpoint,
)

# Reuse AFGNN training utilities.
from utils.run_context import start_run
from utils.trainer import evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate HiFAG")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/hifag_a3_fine_coarse.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (defaults to config's training.checkpoint_path)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="Which split to evaluate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save test results (defaults to the experiment folder)",
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=None,
        help="Experiment directory (e.g. experiments/exp_3). If omitted, inferred from checkpoint or latest experiment.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.checkpoint is None:
        args.checkpoint = cfg["training"]["checkpoint_path"]

    # Resolve experiment directory for saving results.
    if args.exp_dir is not None:
        exp_dir = args.exp_dir
    else:
        inferred = infer_exp_dir_from_checkpoint(args.checkpoint)
        if inferred is not None:
            exp_dir = str(inferred)
        else:
            latest = get_latest_exp_dir("experiments")
            exp_dir = str(latest) if latest is not None else "experiments/results"
    args.output_dir = args.output_dir if args.output_dir is not None else exp_dir
    print(f"[Experiment] Saving test results to: {args.output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=True
    )
    embedded_cfg = checkpoint.get("config")
    if embedded_cfg is not None:
        run_cfg = embedded_cfg
        print(f"[Config] Using config embedded in checkpoint: {args.checkpoint}")
    else:
        run_cfg = cfg
        print("[Config] Checkpoint has no embedded config; falling back to --config")

    run = start_run(run_cfg, "eval")

    loaders = build_loaders(run_cfg, augment=False)
    model = build_model(run_cfg, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from: {args.checkpoint}")
    metric_key = checkpoint.get("early_stopping_metric", "auc")
    best_score = checkpoint.get("best_score", checkpoint.get("best_f1", "N/A"))
    best_threshold = checkpoint.get("best_threshold", 0.5)
    print(f"Best val {metric_key.upper()} (from training): {best_score}")
    print(f"Best validation threshold (F1-tuned): {best_threshold:.4f}")

    metrics = evaluate(
        model, loaders[args.split], device, threshold=best_threshold
    )
    print(f"\n{args.split.upper()} Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(args.output_dir, f"test_{args.split}_results.json")

    result_record = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "timestamp": datetime.now().isoformat(),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }

    with open(result_path, "w") as f:
        json.dump(result_record, f, indent=2)

    print(f"\nResults saved to: {result_path}")

    run.finalize(
        checkpoint_path=args.checkpoint,
        metrics=metrics,
        extra={"split": args.split},
    )


if __name__ == "__main__":
    main()
