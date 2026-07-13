#!/usr/bin/env python3

from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 19,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 19,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})


def find_column(df, candidates):
    lower_to_real = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_to_real:
            return lower_to_real[c.lower()]
    return None


def pretty_model_name(x):
    mapping = {
        "reconstruction_head": "Reconstruction\nhead",
        "parnet_unweighted_average": "Parnet\naverage",
        "uniform_valid_profile": "Uniform\nprofile",
        "train_mean_observed": "Train mean\nprofile",
    }
    return mapping.get(str(x), str(x).replace("_", "\n"))


def plot_loss_comparison(run_dir, out_dir):
    path = run_dir / "evaluation_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}")

    df = pd.read_csv(path)
    model_col = find_column(df, ["model", "baseline", "name"])
    loss_col = find_column(df, ["loss", "test_loss", "cross_entropy", "ce_loss"])

    if model_col is None or loss_col is None:
        raise ValueError(f"Cannot infer model/loss columns from {list(df.columns)}")

    if "split" in df.columns:
        test_df = df[df["split"].astype(str).str.lower().eq("test")].copy()
        if len(test_df) == 0:
            test_df = df.copy()
    else:
        test_df = df.copy()

    order = [
        "reconstruction_head",
        "parnet_unweighted_average",
        "uniform_valid_profile",
        "train_mean_observed",
    ]
    test_df["_order"] = test_df[model_col].apply(
        lambda x: order.index(x) if x in order else len(order)
    )
    test_df = test_df.sort_values("_order")

    labels = [pretty_model_name(x) for x in test_df[model_col]]
    losses = test_df[loss_col].astype(float).values

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    bars = ax.bar(labels, losses,width=0.45)

    ax.set_title("Idea 1 test loss comparison", fontsize=19, pad=14)
    ax.set_ylabel("Test cross-entropy loss\n(lower is better)", fontsize=16)
    ax.set_xlabel("Model / baseline", fontsize=16)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    upper = max(losses) * 1.12
    ax.set_ylim(0, upper)

    for bar, val in zip(bars, losses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(out_dir / f"baseline_comparison_test.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def choose_weight_file(run_dir):
    candidates = [
        "reconstruction_weights_rbp_level.csv",
        "top30_track_weights_annotated.csv",
        "reconstruction_weights_track_level_annotated.csv",
        "top30_track_weights.csv",
        "reconstruction_weights_track_level.csv",
    ]
    for name in candidates:
        path = run_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        "Cannot find a weights CSV. Expected one of: "
        + ", ".join(candidates)
    )


def build_label(row, columns):
    label_col = None
    for c in ["label", "track_label", "rbp_label", "name"]:
        if c in columns:
            label_col = c
            break

    if label_col is not None:
        return str(row[label_col])

    rbp_col = None
    for c in ["RBP", "rbp", "RBP_name", "rbp_name", "protein", "gene_name"]:
        if c in columns:
            rbp_col = c
            break

    cell_col = None
    for c in ["cell_line", "cellline", "cell", "Cell line", "Cell_line"]:
        if c in columns:
            cell_col = c
            break

    if rbp_col is not None and cell_col is not None:
        return f"{row[rbp_col]} ({row[cell_col]})"
    if rbp_col is not None:
        return str(row[rbp_col])

    idx_col = None
    for c in ["track_index", "track", "index"]:
        if c in columns:
            idx_col = c
            break

    if idx_col is not None:
        return f"track {row[idx_col]}"

    return str(row.name)


def plot_rbp_weights(run_dir, out_dir, top_n):
    path = choose_weight_file(run_dir)
    print(f"Using weights file: {path}")

    df = pd.read_csv(path)
    weight_col = find_column(
        df,
        ["weight", "softmax_weight", "reconstruction_weight", "rbp_weight", "value"]
    )

    if weight_col is None:
        raise ValueError(f"Cannot infer weight column from {list(df.columns)}")

    df = df.copy()
    df[weight_col] = df[weight_col].astype(float)
    df = df.sort_values(weight_col, ascending=False).head(top_n)
    df["plot_label"] = df.apply(lambda row: build_label(row, df.columns), axis=1)

    fig_height = max(7.0, 0.38 * len(df))
    fig, ax = plt.subplots(figsize=(10.5, fig_height))

    ax.barh(df["plot_label"], df[weight_col])
    ax.invert_yaxis()

    title = "Top RBP-level reconstruction weights"
    if "track" in path.name.lower():
        title = "Top RBP / cell-line reconstruction weights"

    ax.set_title(title, fontsize=19, pad=14)
    ax.set_xlabel("Softmax reconstruction weight", fontsize=16)
    ax.set_ylabel("RBP / cell-line track", fontsize=16)

    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    fig.tight_layout()
    fig.subplots_adjust(left=0.36, right=0.98, top=0.93, bottom=0.10)

    for ext in ["pdf", "png"]:
        fig.savefig(out_dir / f"top_rbp_weights.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_loss_comparison(run_dir, out_dir)
    plot_rbp_weights(run_dir, out_dir, args.top_n)

    print("Done. Wrote:")
    print(out_dir / "baseline_comparison_test.pdf")
    print(out_dir / "baseline_comparison_test.png")
    print(out_dir / "top_rbp_weights.pdf")
    print(out_dir / "top_rbp_weights.png")


if __name__ == "__main__":
    main()
