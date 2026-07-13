from pathlib import Path
import argparse
import math

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def sparse_to_dense(track):
    if isinstance(track, dict) and {"indices", "values", "size"}.issubset(track.keys()):
        return torch.sparse_coo_tensor(
            track["indices"],
            track["values"],
            track["size"]
        ).to_dense().float()

    if torch.is_tensor(track):
        return track.float()

    raise TypeError(f"Unsupported track type: {type(track)}")

def get_eclip_signal(outputs):
    """
    Find the 223-track eCLIP signal tensor.
    The eCLIP data may be stored as [223, 1, 600], [1, 223, 600],
    [223, 600], or sometimes [600, 223].
    We convert it to [223, 600].
    """
    candidate_keys = [
        "eCLIP",
        "eclip",
        "signal",
        "total",
        "target",
        "profiles",
        "profile",
    ]

    def normalize_shape(x, key):
        x = x.float()

        # Remove singleton dimensions, e.g. [223, 1, 600] -> [223, 600]
        x = x.squeeze()

        if x.ndim != 2:
            raise ValueError(f"Key {key} has unsupported shape after squeeze: {tuple(x.shape)}")

        # Expected shape: [223, 600]
        if x.shape == (223, 600):
            return x

        # Sometimes stored as [600, 223]
        if x.shape == (600, 223):
            return x.T

        raise ValueError(f"Key {key} has unsupported 2D shape: {tuple(x.shape)}")

    for key in candidate_keys:
        if key in outputs:
            x = sparse_to_dense(outputs[key])
            try:
                return normalize_shape(x, key), key
            except ValueError as e:
                print("Tried key", key, "but shape did not match:", e)

    # Try all output keys as fallback
    for key, value in outputs.items():
        try:
            x = sparse_to_dense(value)
            return normalize_shape(x, key), key
        except Exception as e:
            print("Tried fallback key", key, "but failed:", e)

    raise ValueError(f"Could not identify eCLIP signal key. Available keys: {list(outputs.keys())}")



def pearson_1_vs_many(y, X, eps=1e-8):
    """
    y: [600]
    X: [223, 600]
    return: [223]
    """
    y = y.float()
    X = X.float()

    y_centered = y - y.mean()
    X_centered = X - X.mean(dim=1, keepdim=True)

    numerator = (X_centered * y_centered[None, :]).sum(dim=1)
    denominator = torch.sqrt((X_centered ** 2).sum(dim=1) * (y_centered ** 2).sum() + eps)

    corr = numerator / denominator
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr


def load_weights(weights_dir):
    candidates = [
        weights_dir / "reconstruction_weights_track_level_annotated.csv",
        weights_dir / "reconstruction_weights_track_level.csv",
        weights_dir / "top30_track_weights_annotated.csv",
        weights_dir / "top30_track_weights.csv",
    ]

    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            print(f"Using weights file: {p}")
            return df

    raise FileNotFoundError("Could not find reconstruction weights CSV in " + str(weights_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gclip-path",
        default="/home/twang/storage_ml4rg26-deconvgclip/provided_data/600nt_globalCLIP_synchronized_datasets/globalclip_lysate_noNHS_600bp_signalfiltered.pt",
    )
    parser.add_argument(
        "--eclip-path",
        default="/home/twang/storage_ml4rg26-deconvgclip/provided_data/600nt_globalCLIP_synchronized_datasets/eCLIP_600bp_signalfiltered.pt",
    )
    parser.add_argument(
        "--weights-dir",
        default="results/reconstruction_head/interphase_50",
    )
    parser.add_argument(
        "--out-dir",
        default="results/validation/correlation_rank",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-n-tiles", type=int, default=1000)
    parser.add_argument("--top-n-weights", type=int, default=8)

    args = parser.parse_args()

    gclip_path = Path(args.gclip_path)
    eclip_path = Path(args.eclip_path)
    weights_dir = Path(args.weights_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading gCLIP:", gclip_path)
    gclip = torch.load(gclip_path, map_location="cpu", weights_only=False)

    print("Loading eCLIP:", eclip_path)
    eclip = torch.load(eclip_path, map_location="cpu", weights_only=False)

    g_split = gclip[args.split]
    e_split = eclip[args.split]

    assert len(g_split) == len(e_split), "gCLIP and eCLIP split length mismatch"

    names_g = [x["meta"]["name"] for x in g_split]
    names_e = [x["meta"]["name"] for x in e_split]
    assert names_g == names_e, "gCLIP and eCLIP intervals are not in the same order"

    print(f"{args.split} set size:", len(g_split))
    print("Intervals match: yes")

    # 1. Pick top 1000 tiles by raw gCLIP signal sum
    tile_sums = []
    for i, sample in enumerate(g_split):
        y = sparse_to_dense(sample["outputs"]["globalCLIP"]).squeeze(0)
        tile_sums.append(float(y.sum().item()))

    tile_sums = np.array(tile_sums)
    top_indices = np.argsort(-tile_sums)[: args.top_n_tiles]

    top_tiles_df = pd.DataFrame({
        "tile_rank_by_gclip_sum": np.arange(1, len(top_indices) + 1),
        "split_index": top_indices,
        "name": [names_g[i] for i in top_indices],
        "gclip_raw_sum": tile_sums[top_indices],
    })
    top_tiles_df.to_csv(out_dir / "top1000_tiles_by_gclip_signal.csv", index=False)

    print("Selected top tiles:", len(top_indices))
    print("Top gCLIP sums:", tile_sums[top_indices[:5]])

    # 2. Compute 1000 x 223 correlations
    corr_rows = []
    eclip_key_used = None

    for rank, i in enumerate(top_indices, start=1):
        y = sparse_to_dense(g_split[i]["outputs"]["globalCLIP"]).squeeze(0)
        X, key = get_eclip_signal(e_split[i]["outputs"])
        eclip_key_used = key

        if X.shape[0] != 223:
            raise ValueError(f"Expected 223 eCLIP tracks, got shape {X.shape}")

        corr = pearson_1_vs_many(y, X).numpy()

        row = {
            "tile_rank_by_gclip_sum": rank,
            "split_index": int(i),
            "name": names_g[i],
            "gclip_raw_sum": float(tile_sums[i]),
        }

        for k in range(223):
            row[f"track_{k}"] = float(corr[k])

        corr_rows.append(row)

        if rank % 100 == 0:
            print(f"Processed {rank}/{len(top_indices)} tiles")

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(out_dir / "tile_by_track_correlations.csv", index=False)

    print("eCLIP signal key used:", eclip_key_used)

    corr_mat = corr_df[[f"track_{k}" for k in range(223)]].to_numpy()

    # track summary
    mean_corr = np.nanmean(corr_mat, axis=0)
    median_corr = np.nanmedian(corr_mat, axis=0)

    # ranks per tile: rank 1 = highest corr
    ranks = np.zeros_like(corr_mat, dtype=float)
    for i in range(corr_mat.shape[0]):
        order = np.argsort(-corr_mat[i])
        ranks[i, order] = np.arange(1, corr_mat.shape[1] + 1)

    mean_rank = ranks.mean(axis=0)
    median_rank = np.median(ranks, axis=0)

    summary = pd.DataFrame({
        "track_index": np.arange(223),
        "mean_corr_top1000": mean_corr,
        "median_corr_top1000": median_corr,
        "mean_corr_rank_top1000": mean_rank,
        "median_corr_rank_top1000": median_rank,
    })

    # 3. Merge reconstruction weights
    weights = load_weights(weights_dir)

    if "track_index" not in weights.columns:
        if "track" in weights.columns:
            weights = weights.rename(columns={"track": "track_index"})
        else:
            weights["track_index"] = np.arange(len(weights))

    weight_col = None
    for c in ["reconstruction_weight", "weight", "softmax_weight"]:
        if c in weights.columns:
            weight_col = c
            break

    if weight_col is None:
        raise ValueError(f"Could not find weight column. Columns: {list(weights.columns)}")

    weights = weights.copy()
    weights["track_index"] = weights["track_index"].astype(int)
    weights["reconstruction_weight_for_merge"] = weights[weight_col]

    weights["reconstruction_weight_rank"] = (
        weights["reconstruction_weight_for_merge"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    keep_cols = ["track_index", "reconstruction_weight_for_merge", "reconstruction_weight_rank"]
    for c in ["track_label", "rbp_cell_line", "rbp", "RBP", "cell_line", "label", "track_name"]:
        if c in weights.columns:
            keep_cols.append(c)

    summary = summary.merge(weights[keep_cols], on="track_index", how="left")
    summary = summary.sort_values("mean_corr_top1000", ascending=False)
    summary.to_csv(out_dir / "track_correlation_summary.csv", index=False)

    # 4. Top-weighted tracks and their correlation ranks
    top_weighted = summary.sort_values("reconstruction_weight_for_merge", ascending=False).head(args.top_n_weights)
    top_weighted.to_csv(out_dir / "top_weighted_tracks_correlation_ranks.csv", index=False)

    print("\nTop-weighted tracks with correlation ranks:")
    print(top_weighted.to_string(index=False))

    # 5. Scatter plot
    plot_df = summary.dropna(subset=["reconstruction_weight_for_merge"])

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.scatter(
        plot_df["mean_corr_top1000"],
        plot_df["reconstruction_weight_for_merge"],
        s=14,
        alpha=0.75,
    )

    ax.set_xlabel("Mean Pearson correlation with gCLIP\n(top 1000 high-signal test tiles)", fontsize=8)
    ax.set_ylabel("Reconstruction weight", fontsize=8)
    ax.set_title("Reconstruction weight vs. eCLIP-gCLIP correlation", fontsize=9)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "reconstruction_weight_vs_mean_correlation.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "reconstruction_weight_vs_mean_correlation.png", dpi=300, bbox_inches="tight")

    # 6. Barplot for top-weighted correlation ranks
    label_col = None
    for c in ["track_label", "rbp_cell_line", "label", "track_name", "rbp", "RBP"]:
        if c in top_weighted.columns:
            label_col = c
            break

    if label_col is None:
        labels = ["track_" + str(x) for x in top_weighted["track_index"]]
    else:
        labels = top_weighted[label_col].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    y = np.arange(len(top_weighted))
    ax.barh(y, top_weighted["median_corr_rank_top1000"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Median correlation rank across top 1000 tiles\n(1 = best among 223 tracks)", fontsize=8)
    ax.set_title("Correlation ranks of top-weighted RBP tracks", fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "top_weighted_tracks_correlation_ranks.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "top_weighted_tracks_correlation_ranks.png", dpi=300, bbox_inches="tight")

    print("\nSaved outputs to:", out_dir)


if __name__ == "__main__":
    main()
