from pathlib import Path
import argparse
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})


def pick_col(columns, candidates, name):
    for c in candidates:
        if c in columns:
            return c
    raise ValueError(f"Could not find {name}. Tried {candidates}. Available columns: {list(columns)}")


def make_track_label(df):
    if "rbp_ct" in df.columns:
        return df["rbp_ct"].astype(str)
    if "track_label" in df.columns:
        return df["track_label"].astype(str)
    if "rbp" in df.columns and "ct" in df.columns:
        return df["rbp"].astype(str) + "_" + df["ct"].astype(str)
    if "rbp_x" in df.columns and "ct_x" in df.columns:
        return df["rbp_x"].astype(str) + "_" + df["ct_x"].astype(str)
    return "track_" + df.index.astype(str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-n", type=int, default=8)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # (a) Training and validation loss
    # ----------------------------
    train_path = run_dir / "training_metrics.csv"
    train_df = pd.read_csv(train_path)

    epoch_col = pick_col(train_df.columns, ["epoch", "Epoch"], "epoch column")
    train_col = pick_col(
        train_df.columns,
        ["train_loss", "loss_train", "train", "training_loss"],
        "training loss column",
    )
    valid_col = pick_col(
        train_df.columns,
        ["valid_loss", "val_loss", "validation_loss", "valid", "val"],
        "validation loss column",
    )

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.plot(
        train_df[epoch_col],
        train_df[train_col],
        marker="o",
        linewidth=2.0,
        markersize=4,
        label="train",
    )
    ax.plot(
        train_df[epoch_col],
        train_df[valid_col],
        marker="o",
        linewidth=2.0,
        markersize=4,
        label="valid",
    )

    ax.set_title("Training and validation loss", pad=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Profile cross-entropy / NLL")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    fig.savefig(out_dir / "training_validation_loss_big.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "training_validation_loss_big.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ----------------------------
    # (b) Top track-level reconstruction weights (TOP 8 only)
    # ----------------------------
    weights_path = run_dir / "reconstruction_weights_track_level_annotated.csv"
    weights_df = pd.read_csv(weights_path)

    weight_col = pick_col(
        weights_df.columns,
        ["reconstruction_weight", "weight"],
        "reconstruction weight column",
    )

    weights_df = weights_df.copy()
    weights_df["track_label"] = make_track_label(weights_df)
    weights_df["plot_weight"] = weights_df[weight_col].astype(float)

    top_df = (
        weights_df.sort_values("plot_weight", ascending=False)
        .head(args.top_n)
        .copy()
    )

    # reverse for barh so the largest is at the top
    top_df = top_df.iloc[::-1].copy()

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    bars = ax.barh(top_df["track_label"], top_df["plot_weight"])

    ax.set_title(f"Top {args.top_n} track-level reconstruction weights", pad=10)
    ax.set_xlabel("Softmax weight")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)

    xmax = float(top_df["plot_weight"].max()) * 1.22
    ax.set_xlim(0, xmax)

    for bar, val in zip(bars, top_df["plot_weight"]):
        ax.text(
            min(val + xmax * 0.012, xmax * 0.98),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=11,
        )

    fig.tight_layout()
    fig.savefig(out_dir / "top8_track_weights_big.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "top8_track_weights_big.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved files:")
    for p in sorted(out_dir.iterdir()):
        print(" ", p.name)


if __name__ == "__main__":
    main()
