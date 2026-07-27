"""Sanity check: do hand-crafted region descriptors carry depression signal?

Loads D-Vlog landmark sequences, computes per-region descriptors
(hifag.data.region_features), aggregates them over time per sample
(temporal mean + std), and compares depressed vs non-depressed groups with
univariate statistics (Cohen's d, Welch t with normal-approx p-value,
rank-based AUC).

No model is trained here — this answers "does the signal exist" before we
build the coarse graph branch. See DESIGN.md section 5.

Usage:
    python src/scripts/sanity_region_features.py \
        --data_dir /data/ltq/DVlog/processed_official_features --num_frames 32
"""

import argparse
import json
import math
import os
import sys

import numpy as np

# Make `hifag` importable regardless of the current working directory.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import hifag.paths  # noqa: F401,E402  (sys.path setup, incl. AFGNN/src)
from hifag.data.region_features import (
    FEATURE_NAMES,
    REGION_NAMES,
    compute_region_features,
)


def sample_frames(visual_seq: np.ndarray, num_frames: int) -> np.ndarray:
    """Uniform frame sampling, identical to AFGNN's sample_frames."""
    T = visual_seq.shape[0]
    indices = np.linspace(0, T - 1, num_frames, dtype=int)
    return visual_seq[indices]


def load_split(data_dir: str, split: str):
    visual = np.load(os.path.join(data_dir, f"{split}_visual.npy"))
    labels = np.load(os.path.join(data_dir, f"{split}_labels.npy")).astype(np.int64)
    return visual, labels


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d of group a (pos) vs group b (neg), pooled std."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if var <= 0:
        return 0.0
    return float((a.mean() - b.mean()) / math.sqrt(var))


def welch_t_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided p-value of Welch's t-test via normal approximation."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 1.0
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    if se <= 0:
        return 1.0
    t = abs(a.mean() - b.mean()) / se
    # normal approximation: p = 2 * (1 - Phi(t))
    return float(math.erfc(t / math.sqrt(2.0)))


def rank_auc(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) via average ranks (ties handled). a=pos, b=neg."""
    values = np.concatenate([a, b])
    order = values.argsort(kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    # average ranks for ties
    sorted_vals = values[order]
    sorted_ranks = ranks[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            sorted_ranks[i : j + 1] = sorted_ranks[i : j + 1].mean()
        i = j + 1
    ranks[order] = sorted_ranks
    na, nb = len(a), len(b)
    sum_ranks_pos = ranks[:na].sum()
    return float((sum_ranks_pos - na * (na + 1) / 2.0) / (na * nb))


def aggregate_sample(feats: np.ndarray) -> np.ndarray:
    """(T, R, F) -> flat (2*R*F,) vector of temporal mean and std."""
    mean = feats.mean(axis=0)   # (R, F)
    std = feats.std(axis=0)     # (R, F)
    return np.concatenate([mean.reshape(-1), std.reshape(-1)]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir", default="/data/ltq/DVlog/processed_official_features"
    )
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument(
        "--out",
        default="experiments/results/sanity_region_features.json",
        help="output JSON path (relative to HiFAG root if not absolute)",
    )
    args = parser.parse_args()

    # Load all splits (no training happens here, so combining is fine and
    # maximizes statistical power).
    visual_all, labels_all = [], []
    for split in ["train", "valid", "test"]:
        v, l = load_split(args.data_dir, split)
        visual_all.append(v)
        labels_all.append(l)
        print(f"[load] {split}: {v.shape}, pos_ratio={l.mean():.3f}")
    visual = np.concatenate(visual_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)
    n_pos = int(labels.sum())
    print(f"[load] total: {visual.shape}, pos={n_pos}, neg={len(labels) - n_pos}")

    # Per-sample aggregated region features.
    agg_names = (
        [f"mean::{r}.{f}" for r in REGION_NAMES for f in FEATURE_NAMES]
        + [f"std::{r}.{f}" for r in REGION_NAMES for f in FEATURE_NAMES]
    )
    X = np.zeros((len(visual), len(agg_names)), dtype=np.float32)
    for i, seq in enumerate(visual):
        sampled = sample_frames(seq, args.num_frames)
        coords = sampled.reshape(args.num_frames, 68, 2)
        feats = compute_region_features(coords)  # (T, 9, 10)
        X[i] = aggregate_sample(feats)
        if (i + 1) % 200 == 0:
            print(f"[features] {i + 1}/{len(visual)}")

    pos = labels == 1
    neg = ~pos

    results = []
    for j, name in enumerate(agg_names):
        a = X[pos, j].astype(np.float64)
        b = X[neg, j].astype(np.float64)
        results.append(
            {
                "feature": name,
                "mean_pos": float(a.mean()),
                "mean_neg": float(b.mean()),
                "cohens_d": cohens_d(a, b),
                "p_value": welch_t_pvalue(a, b),
                "auc": rank_auc(a, b),
            }
        )

    results.sort(key=lambda r: abs(r["auc"] - 0.5), reverse=True)

    # Per-region summary: best |AUC-0.5| among that region's features.
    region_summary = {}
    for r in REGION_NAMES:
        region_feats = [res for res in results if f"::{r}." in res["feature"]]
        best = max(region_feats, key=lambda res: abs(res["auc"] - 0.5))
        region_summary[r] = {
            "best_feature": best["feature"],
            "best_auc": best["auc"],
            "best_cohens_d": best["cohens_d"],
            "best_p_value": best["p_value"],
        }

    report = {
        "data_dir": args.data_dir,
        "num_frames": args.num_frames,
        "n_samples": int(len(labels)),
        "n_pos": n_pos,
        "n_neg": int(len(labels) - n_pos),
        "region_summary": region_summary,
        "top_features": results[: args.top_k],
        "all_features": results,
    }

    out_path = args.out
    if not os.path.isabs(out_path):
        import hifag.paths as _paths

        out_path = os.path.join(_paths.HIFAG_ROOT, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Console summary.
    print("\n=== Per-region best features ===")
    for r, s in region_summary.items():
        print(
            f"  {r:15s} {s['best_feature']:32s} "
            f"AUC={s['best_auc']:.3f} d={s['best_cohens_d']:+.3f} p={s['best_p_value']:.2e}"
        )
    print(f"\n=== Top {args.top_k} features by |AUC-0.5| ===")
    for res in results[: args.top_k]:
        sig = "*" if res["p_value"] < 0.05 else " "
        print(
            f" {sig} {res['feature']:36s} AUC={res['auc']:.3f} "
            f"d={res['cohens_d']:+.3f} p={res['p_value']:.2e}"
        )
    n_sig = sum(1 for res in results if res["p_value"] < 0.05)
    print(f"\n{len(results)} features total, {n_sig} with p<0.05 (uncorrected).")
    print(f"Report saved to {out_path}")


if __name__ == "__main__":
    main()
