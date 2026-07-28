"""Plot region-level depression effects of the hand-crafted descriptors.

Two-panel figure supporting the coarse graph's interpretability claim:
  A) Heatmap: Cohen's d (depressed vs non-depressed) of the temporal-std
     aggregates, 9 regions x 10 descriptors. Negative (blue) = depressed
     group lower = narrower facial dynamics.
  B) Face map: each of the 9 regions colored by its strongest temporal-std
     effect (per-region best Cohen's d), drawn on the mean landmark layout.

Statistics are read from experiments/results/sanity_region_features.json
(produced by sanity_region_features.py); no recomputation, no model.

Usage:
    python src/scripts/plot_region_effects.py \
        --data_dir /data/ltq/DVlog/processed_official_features
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import hifag.paths  # noqa: F401,E402
from hifag.data.region_features import (
    FEATURE_NAMES,
    REGION_GROUPS,
    REGION_NAMES,
    flat_to_coords,
)


def load_stats(path: str):
    with open(path) as f:
        report = json.load(f)
    d_std = np.zeros((len(REGION_NAMES), len(FEATURE_NAMES)))
    p_std = np.ones_like(d_std)
    for rec in report["all_features"]:
        agg, rest = rec["feature"].split("::")
        region, feat = rest.split(".")
        if agg != "std":
            continue
        i, j = REGION_NAMES.index(region), FEATURE_NAMES.index(feat)
        d_std[i, j] = rec["cohens_d"]
        p_std[i, j] = rec["p_value"]
    region_best = np.array(
        [report["region_summary"][r]["best_cohens_d"] for r in REGION_NAMES]
    )
    return d_std, p_std, region_best


def mean_landmarks(data_dir: str, n: int = 200) -> np.ndarray:
    """Mean (68, 2) landmark layout over a subset of train samples."""
    visual = np.load(os.path.join(data_dir, "train_visual.npy"))[:n]
    # OpenFace layout is [x_0..x_67, y_0..y_67]; do NOT reshape(68, 2).
    coords = np.stack([flat_to_coords(seq) for seq in visual], axis=0)
    return coords.mean(axis=(0, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir", default="/data/ltq/DVlog/processed_official_features"
    )
    parser.add_argument(
        "--stats", default="experiments/results/sanity_region_features.json"
    )
    parser.add_argument(
        "--out", default="experiments/results/region_depression_effect.png"
    )
    args = parser.parse_args()

    d_std, p_std, region_best = load_stats(args.stats)
    landmarks = mean_landmarks(args.data_dir)

    fig, (ax_hm, ax_face) = plt.subplots(
        1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.35, 1]}
    )

    # ---- Panel A: heatmap of temporal-std Cohen's d ----
    vmax = np.abs(d_std).max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax_hm.imshow(d_std, cmap="RdBu_r", norm=norm, aspect="auto")
    ax_hm.set_xticks(range(len(FEATURE_NAMES)))
    ax_hm.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right", fontsize=9)
    ax_hm.set_yticks(range(len(REGION_NAMES)))
    ax_hm.set_yticklabels(REGION_NAMES, fontsize=9)
    for i in range(len(REGION_NAMES)):
        for j in range(len(FEATURE_NAMES)):
            star = "*" if p_std[i, j] < 0.05 else ""
            color = "white" if abs(d_std[i, j]) > 0.6 * vmax else "black"
            ax_hm.text(
                j, i, f"{d_std[i, j]:+.2f}{star}",
                ha="center", va="center", fontsize=7, color=color,
            )
    ax_hm.set_title(
        "A. Temporal-variability effect per region x descriptor\n"
        "(Cohen's d, depressed vs non-depressed; * p<0.05)",
        fontsize=11,
    )
    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
    cbar.set_label("Cohen's d (blue = depressed lower)", fontsize=9)

    # ---- Panel B: face map colored by per-region best effect ----
    SHORT_NAMES = {
        "face_contour": "jaw contour",
        "left_eyebrow": "L brow",
        "right_eyebrow": "R brow",
        "nose_bridge": "nose bridge",
        "nose_bottom": "nose bottom",
        "left_eye": "L eye",
        "right_eye": "R eye",
        "outer_mouth": "outer mouth",
        "inner_mouth": "inner mouth",
    }
    # Fixed per-region label offsets (data coords) to avoid collisions;
    # tuned on the mean landmark layout.
    LABEL_OFFSETS = {
        "face_contour": (0.0, 0.30),
        "left_eyebrow": (0.0, -0.16),
        "right_eyebrow": (0.0, -0.16),
        "nose_bridge": (0.34, -0.04),
        "nose_bottom": (0.36, 0.10),
        "left_eye": (0.0, -0.14),
        "right_eye": (0.0, -0.14),
        "outer_mouth": (0.45, 0.22),
        "inner_mouth": (-0.45, 0.22),
    }
    norm_face = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")
    for r, name in enumerate(REGION_NAMES):
        pts = landmarks[REGION_GROUPS[name]]
        color = cmap(norm_face(region_best[r]))
        if name == "face_contour":
            # Jaw points are an open chain: a filled hull would cover the
            # whole face. Draw it as a thick polyline instead.
            ax_face.plot(
                pts[:, 0], pts[:, 1], color=color, linewidth=6,
                solid_capstyle="round", zorder=2,
            )
            anchor = landmarks[8]  # chin
        else:
            hull = ConvexHull(pts)
            poly = pts[hull.vertices]
            ax_face.fill(
                poly[:, 0], poly[:, 1],
                color=color, alpha=0.8,
                edgecolor="white", linewidth=1.0, zorder=2,
            )
            anchor = pts.mean(axis=0)
        lx, ly = anchor + np.array(LABEL_OFFSETS[name])
        ax_face.text(
            lx, ly, f"{SHORT_NAMES[name]}\n{region_best[r]:+.2f}",
            ha="center", va="center", fontsize=7.5, fontweight="bold",
            zorder=4,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )
    ax_face.scatter(
        landmarks[:, 0], landmarks[:, 1], s=4, c="black", zorder=3
    )
    ax_face.set_aspect("equal")
    ax_face.invert_yaxis()  # image coordinates -> anatomical orientation
    ax_face.axis("off")
    ax_face.set_title(
        "B. Strongest temporal-variability effect per region\n"
        "(best Cohen's d among that region's std features)",
        fontsize=10,
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_face)
    cbar2 = fig.colorbar(sm, ax=ax_face, fraction=0.046, pad=0.04)
    cbar2.set_label("Cohen's d (blue = depressed lower)", fontsize=9)

    fig.suptitle(
        "Macro-level facial regions vs depression: reduced dynamic range "
        "in the depressed group (D-Vlog, n=961)",
        fontsize=12.5, y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
