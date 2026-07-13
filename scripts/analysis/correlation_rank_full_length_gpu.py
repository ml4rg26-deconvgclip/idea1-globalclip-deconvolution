from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sparse_to_dense(track):
    return torch.sparse_coo_tensor(
        track["indices"],
        track["values"],
        track["size"]
    ).to_dense().float()


def signal_len(ex, key):
    return int(ex["outputs"][key]["size"][-1])


def signal_sum(ex, key):
    return float(ex["outputs"][key]["values"].sum().item())


def find_weight_file(weights_dir):
    candidates = [
        weights_dir / "reconstruction_weights_track_level_annotated.csv",
        weights_dir / "reconstruction_weights_track_level.csv",
        weights_dir / "top30_track_weights_annotated.csv",
        weights_dir / "top30_track_weights.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No weight file found in {weights_dir}")


def load_weights(weights_dir):
    p = find_weight_file(weights_dir)
    print("Using weights file:", p)
    df = pd.read_csv(p)

    if "track_index" not in df.columns:
        df["track_index"] = np.arange(len(df))

    if "weight" in df.columns:
        weight_col = "weight"
    elif "reconstruction_weight" in df.columns:
        weight_col = "reconstruction_weight"
    else:
        raise ValueError(f"Cannot find weight column. Columns: {list(df.columns)}")

    df = df.copy()
    df["reconstruction_weight"] = df[weight_col].astype(float)
    df["reconstruction_weight_rank"] = (
        df["reconstruction_weight"].rank(ascending=False, method="min").astype(int)
    )

    keep = ["track_index", "reconstruction_weight", "reconstruction_weight_rank"]
    for c in ["rbp_ct", "rbp", "ct", "track_label", "label"]:
        if c in df.columns and c not in keep:
            keep.append(c)

    return df[keep]


def pearson_batch(y, x, eps=1e-8):
    """
    y: [B, L]
    x: [B, K, L]
    return: [B, K]
    """
    y = y.float()
    x = x.float()

    y_c = y - y.mean(dim=1, keepdim=True)
    x_c = x - x.mean(dim=2, keepdim=True)

    numerator = (x_c * y_c[:, None, :]).sum(dim=2)
    denominator = torch.sqrt(
        (x_c ** 2).sum(dim=2) * (y_c ** 2).sum(dim=1, keepdim=True) + eps
    )

    corr = numerator / denominator
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr


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
        "--rbp-cts",
        default="/home/twang/storage_ml4rg26-deconvgclip/provided_data/600nt_globalCLIP_synchronized_datasets/eCLIP_600bp_signalfiltered.rbp_cts.tsv",
    )
    parser.add_argument(
        "--weights-dir",
        default="results/reconstruction_head/interphase_50",
    )
    parser.add_argument(
        "--out-dir",
        default="results/validation/correlation_rank_full_length",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-n-tiles", type=int, default=1000)
    parser.add_argument("--top-n-weights", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    gclip_path = Path(args.gclip_path)
    eclip_path = Path(args.eclip_path)
    rbp_cts_path = Path(args.rbp_cts)
    weights_dir = Path(args.weights_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print("Using device:", device)

    print("Loading gCLIP:", gclip_path)
    gclip = torch.load(gclip_path, map_location="cpu", weights_only=False)

    print("Loading eCLIP:", eclip_path)
    eclip = torch.load(eclip_path, map_location="cpu", weights_only=False)

    g_split = gclip[args.split]
    e_split = eclip[args.split]

    assert len(g_split) == len(e_split)

    names_g = [x["meta"]["name"] for x in g_split]
    names_e = [x["meta"]["name"] for x in e_split]
    assert names_g == names_e, "gCLIP and eCLIP intervals are not aligned"

    print("Split:", args.split)
    print("Total test windows:", len(g_split))
    print("Intervals aligned: yes")

    # Select only windows where both signals are full-length 600.
    eligible = []
    g_sums = []

    for i, ex in enumerate(g_split):
        g_len = signal_len(ex, "globalCLIP")
        e_len = signal_len(e_split[i], "eCLIP")

        if g_len == 600 and e_len == 600:
            eligible.append(i)
            g_sums.append(signal_sum(ex, "globalCLIP"))

    eligible = np.array(eligible)
    g_sums = np.array(g_sums)

    print("Full-length eligible windows:", len(eligible))

    order = np.argsort(-g_sums)
    selected = eligible[order[: args.top_n_tiles]]
    selected_sums = g_sums[order[: args.top_n_tiles]]

    print("Selected full-length high-signal tiles:", len(selected))
    print("Top gCLIP sums:", selected_sums[:10])

    pd.DataFrame({
        "tile_rank_by_gclip_sum": np.arange(1, len(selected) + 1),
        "split_index": selected,
        "name": [names_g[i] for i in selected],
        "gclip_raw_sum": selected_sums,
        "length_used": 600,
    }).to_csv(out_dir / "top1000_full_length_tiles_by_gclip_signal.csv", index=False)

    # Compute correlations on GPU in batches.
    corr_chunks = []

    for start in range(0, len(selected), args.batch_size):
        batch_idx = selected[start:start + args.batch_size]

        y_list = []
        x_list = []

        for i in batch_idx:
            y = sparse_to_dense(g_split[i]["outputs"]["globalCLIP"]).squeeze()
            x = sparse_to_dense(e_split[i]["outputs"]["eCLIP"]).squeeze()

            if y.shape != (600,):
                raise ValueError(f"gCLIP shape mismatch at index {i}: {tuple(y.shape)}")
            if x.shape != (223, 600):
                raise ValueError(f"eCLIP shape mismatch at index {i}: {tuple(x.shape)}")

            y_list.append(y)
            x_list.append(x)

        y_b = torch.stack(y_list).to(device)
        x_b = torch.stack(x_list).to(device)

        corr_b = pearson_batch(y_b, x_b).cpu().numpy()
        corr_chunks.append(corr_b)

        done = min(start + args.batch_size, len(selected))
        print(f"Processed {done}/{len(selected)} tiles")

    corr_mat = np.vstack(corr_chunks)
    np.save(out_dir / "tile_by_track_correlations.npy", corr_mat)

    corr_df = pd.DataFrame(corr_mat, columns=[f"track_{i}" for i in range(corr_mat.shape[1])])
    corr_df.insert(0, "split_index", selected)
    corr_df.insert(1, "name", [names_g[i] for i in selected])
    corr_df.insert(2, "gclip_raw_sum", selected_sums)
    corr_df.to_csv(out_dir / "tile_by_track_correlations.csv", index=False)

    # Per-tile ranks: rank 1 means highest correlation among 223 tracks.
    ranks = np.empty_like(corr_mat, dtype=float)
    for r in range(corr_mat.shape[0]):
        rank_order = np.argsort(-corr_mat[r])
        ranks[r, rank_order] = np.arange(1, corr_mat.shape[1] + 1)

    summary = pd.DataFrame({
        "track_index": np.arange(corr_mat.shape[1]),
        "mean_corr_top1000": np.mean(corr_mat, axis=0),
        "median_corr_top1000": np.median(corr_mat, axis=0),
        "mean_corr_rank_top1000": np.mean(ranks, axis=0),
        "median_corr_rank_top1000": np.median(ranks, axis=0),
        "fraction_rank_top10": (ranks <= 10).mean(axis=0),
        "fraction_rank_top20": (ranks <= 20).mean(axis=0),
    })

    if rbp_cts_path.exists():
        rbp = pd.read_csv(rbp_cts_path, sep="\t")
        rbp["track_index"] = np.arange(len(rbp))
        summary = summary.merge(rbp, on="track_index", how="left")

    weights = load_weights(weights_dir)
    summary = summary.merge(weights, on="track_index", how="left")

    summary.to_csv(out_dir / "track_correlation_summary.csv", index=False)

    top_weighted = (
        summary.sort_values("reconstruction_weight", ascending=False)
        .head(args.top_n_weights)
    )
    top_weighted.to_csv(out_dir / "top_weighted_tracks_correlation_ranks.csv", index=False)

    top_correlated = summary.sort_values("mean_corr_top1000", ascending=False).head(30)
    top_correlated.to_csv(out_dir / "top30_correlated_tracks.csv", index=False)

    print("\nTop-weighted tracks and their direct eCLIP-gCLIP correlation ranks:")
    print(top_weighted.to_string(index=False))

    # Scatter plot: reconstruction weight vs mean correlation.
    plot_df = summary.dropna(subset=["reconstruction_weight"]).copy()

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.scatter(plot_df["mean_corr_top1000"], plot_df["reconstruction_weight"], s=14, alpha=0.75)
    ax.set_xlabel("Mean Pearson correlation with gCLIP\n(top 1000 full-length high-signal test tiles)", fontsize=8)
    ax.set_ylabel("Reconstruction weight", fontsize=8)
    ax.set_title("Reconstruction weight vs. eCLIP-gCLIP correlation", fontsize=9)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "reconstruction_weight_vs_mean_correlation.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "reconstruction_weight_vs_mean_correlation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    label_col = "rbp_ct" if "rbp_ct" in top_weighted.columns else "track_index"
    labels = top_weighted[label_col].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    y_pos = np.arange(len(top_weighted))
    ax.barh(y_pos, top_weighted["median_corr_rank_top1000"])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Median correlation rank across top 1000 tiles\n(1 = best among 223 tracks)", fontsize=8)
    ax.set_title("Correlation ranks of top-weighted tracks", fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "top_weighted_tracks_correlation_ranks.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "top_weighted_tracks_correlation_ranks.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved outputs to:", out_dir)


if __name__ == "__main__":
    main()
