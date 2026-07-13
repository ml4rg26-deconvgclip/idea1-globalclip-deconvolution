#!/usr/bin/env python3
"""
Regenerate Idea 1 reconstruction report figures using the plotting style
from the comparison notebook.

This script only reads existing CSV result files. It does NOT rerun PARNET,
does NOT retrain the reconstruction head, and does NOT modify .pt/checkpoint files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RESULT_DIR = Path("results/reconstruction_head/interphase_50")


def apply_notebook_plot_style() -> None:
    """Apply simple matplotlib style matching the project comparison notebook."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "lines.linewidth": 1.5,
            "lines.markersize": 3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for suffix in ["png", "pdf"]:
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        written.append(path)

    plt.close(fig)
    return written


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    print(f"Reading: {path}")
    return pd.read_csv(path)


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_to_original = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def numeric_col(df: pd.DataFrame, candidates: list[str]) -> str:
    col = find_col(df, candidates)
    if col is not None:
        return col

    numeric_cols = []
    for c in df.columns:
        values = pd.to_numeric(df[c], errors="coerce")
        if values.notna().sum() > 0:
            numeric_cols.append(c)

    if not numeric_cols:
        raise ValueError(f"No numeric columns found in: {list(df.columns)}")

    return numeric_cols[0]


def plot_training_curve(training: pd.DataFrame, output_dir: Path) -> list[Path]:
    df = training.copy()

    epoch_col = find_col(df, ["epoch", "Epoch", "step", "iteration"])
    if epoch_col is None:
        df["_epoch"] = np.arange(len(df))
        epoch_col = "_epoch"

    train_col = find_col(df, ["train_loss", "training_loss", "loss_train"])
    valid_col = find_col(df, ["valid_loss", "val_loss", "validation_loss", "loss_valid"])

    # Robust fallback: choose columns containing both loss and train/valid.
    if train_col is None:
        for c in df.columns:
            if "train" in c.lower() and "loss" in c.lower():
                train_col = c
                break
    if valid_col is None:
        for c in df.columns:
            if ("valid" in c.lower() or "val" in c.lower()) and "loss" in c.lower():
                valid_col = c
                break

    fig, ax = plt.subplots(figsize=(13, 4.5))

    if train_col is not None:
        ax.plot(
            df[epoch_col],
            pd.to_numeric(df[train_col], errors="coerce"),
            marker="o",
            markersize=3,
            color="tab:blue",
            label="train",
        )

    if valid_col is not None:
        ax.plot(
            df[epoch_col],
            pd.to_numeric(df[valid_col], errors="coerce"),
            marker="o",
            markersize=3,
            color="tab:orange",
            label="valid",
        )

    if train_col is None and valid_col is None:
        # Last-resort fallback: plot every numeric column except epoch.
        for idx, c in enumerate(df.columns):
            if c == epoch_col:
                continue
            values = pd.to_numeric(df[c], errors="coerce")
            if values.notna().sum() > 0:
                ax.plot(df[epoch_col], values, marker="o", markersize=3, label=c)

    ax.set_title("Training and validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Profile cross-entropy / NLL")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    return save_figure(fig, output_dir, "training_curve")


def plot_baseline_comparison(evaluation: pd.DataFrame, output_dir: Path) -> list[Path]:
    df = evaluation.copy()

    split_col = find_col(df, ["split"])
    model_col = find_col(df, ["model"])
    loss_col = numeric_col(df, ["loss", "nll", "cross_entropy"])

    if split_col is not None:
        test_df = df[df[split_col].astype(str).str.lower() == "test"].copy()
        if len(test_df) == 0:
            test_df = df.copy()
    else:
        test_df = df.copy()

    if model_col is None:
        model_col = "model"
        test_df[model_col] = np.arange(len(test_df)).astype(str)

    test_df[loss_col] = pd.to_numeric(test_df[loss_col], errors="coerce")
    test_df = test_df.dropna(subset=[loss_col])
    test_df = test_df.sort_values(loss_col, ascending=True)

    labels = test_df[model_col].astype(str).tolist()
    losses = test_df[loss_col].astype(float).tolist()

    fig, ax = plt.subplots(figsize=(13, 4.5))
    colors = ["tab:blue" if "reconstruction" in label.lower() else "tab:orange" for label in labels]
    bars = ax.bar(labels, losses, color=colors, alpha=0.85)

    ax.set_title("Test loss comparison (lower is better)")
    ax.set_ylabel("Profile cross-entropy / NLL")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")

    for bar, value in zip(bars, losses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    return save_figure(fig, output_dir, "baseline_comparison_test")


def make_track_label(df: pd.DataFrame) -> pd.Series:
    # Prefer the pre-computed readable label from the annotated weights table.
    rbp_ct_col = find_col(df, ["rbp_ct", "track_label", "track_name"])
    if rbp_ct_col is not None:
        return df[rbp_ct_col].astype(str)

    rbp_col = find_col(df, ["rbp", "RBP", "rbp_name", "target"])
    cell_col = find_col(df, ["ct", "cell_line", "cellline", "cell"])

    if rbp_col is not None and cell_col is not None:
        return df[rbp_col].astype(str) + "_" + df[cell_col].astype(str)

    label_col = find_col(df, ["track", "name", "experiment", "feature"])
    if label_col is not None:
        return df[label_col].astype(str)

    idx_col = find_col(df, ["track_index", "index"])
    if idx_col is not None:
        return "track_" + df[idx_col].astype(str)

    return pd.Series([f"track_{i}" for i in range(len(df))], index=df.index)


def plot_top_track_weights(track_df: pd.DataFrame, output_dir: Path, top_n: int) -> list[Path]:
    df = track_df.copy()
    weight_col = numeric_col(df, ["weight", "softmax_weight", "reconstruction_weight"])

    df["_weight"] = pd.to_numeric(df[weight_col], errors="coerce")
    df["_label"] = make_track_label(df)
    df = df.dropna(subset=["_weight"])
    df = df.sort_values("_weight", ascending=False).head(top_n)
    df = df.sort_values("_weight", ascending=True)

    fig_height = max(5, min(14, 0.32 * len(df) + 1.5))
    fig, ax = plt.subplots(figsize=(13, fig_height))

    ax.barh(df["_label"], df["_weight"], color="tab:blue", alpha=0.85)
    ax.set_title("Top track-level reconstruction weights")
    ax.set_xlabel("Softmax weight")
    ax.set_ylabel("")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    fig.tight_layout()
    return save_figure(fig, output_dir, "top30_track_weights")


def plot_top_rbp_weights(rbp_df: pd.DataFrame, output_dir: Path, top_n: int) -> list[Path]:
    df = rbp_df.copy()

    rbp_col = find_col(df, ["rbp", "RBP", "rbp_name", "target", "gene", "name"])
    if rbp_col is None:
        rbp_col = df.columns[0]

    weight_col = numeric_col(df, ["weight", "softmax_weight", "aggregated_weight", "total_weight"])

    df["_weight"] = pd.to_numeric(df[weight_col], errors="coerce")
    df["_label"] = df[rbp_col].astype(str)
    df = df.dropna(subset=["_weight"])
    df = df.sort_values("_weight", ascending=False).head(top_n)
    df = df.sort_values("_weight", ascending=True)

    fig_height = max(5, min(14, 0.32 * len(df) + 1.5))
    fig, ax = plt.subplots(figsize=(13, fig_height))

    ax.barh(df["_label"], df["_weight"], color="tab:blue", alpha=0.85)
    ax.set_title("Top RBP-level reconstruction weights")
    ax.set_xlabel("Aggregated softmax weight")
    ax.set_ylabel("")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    fig.tight_layout()
    return save_figure(fig, output_dir, "top_rbp_weights")


def generate(result_dir: Path, output_dir: Path | None, top_n: int) -> list[Path]:
    apply_notebook_plot_style()

    result_dir = result_dir.resolve()
    output_dir = (output_dir or result_dir / "report_assets_notebook_style").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    training_path = result_dir / "training_metrics.csv"
    evaluation_path = result_dir / "evaluation_metrics.csv"
    top_tracks_path = result_dir / "top30_track_weights_annotated.csv"
    if not top_tracks_path.exists():
        top_tracks_path = result_dir / "top30_track_weights.csv"
    rbp_weights_path = result_dir / "reconstruction_weights_rbp_level.csv"

    training = read_csv(training_path)
    evaluation = read_csv(evaluation_path)
    top_tracks = read_csv(top_tracks_path)
    rbp_weights = read_csv(rbp_weights_path)

    written: list[Path] = []
    written.extend(plot_training_curve(training, output_dir))
    written.extend(plot_baseline_comparison(evaluation, output_dir))
    written.extend(plot_top_track_weights(top_tracks, output_dir, top_n=top_n))
    written.extend(plot_top_rbp_weights(rbp_weights, output_dir, top_n=top_n))

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Directory containing reconstruction result CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <result-dir>/report_assets_notebook_style",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top rows to show in weight plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = generate(args.result_dir, args.output_dir, args.top_n)

    print("\nWrote report assets:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
