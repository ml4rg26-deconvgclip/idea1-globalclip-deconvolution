from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pick_first(d, keys, name):
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"Could not find {name}. Tried {keys}. Available keys: {list(d.keys())}")


def squeeze_profile(x):
    if torch.is_tensor(x):
        x = x.detach().cpu()
    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]
    return x


def to_string_list(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        x = x.tolist()
    out = []
    for v in x:
        if isinstance(v, (bytes, bytearray)):
            out.append(v.decode())
        else:
            out.append(str(v))
    return out


def masked_pearson_batch(y, p, mask, eps=1e-8):
    """
    y, p, mask: [B, L]
    """
    y = y.float()
    p = p.float()
    mask = mask.bool()

    m = mask.float()
    n = m.sum(dim=1, keepdim=True).clamp_min(1.0)

    y_mean = (y * m).sum(dim=1, keepdim=True) / n
    p_mean = (p * m).sum(dim=1, keepdim=True) / n

    yc = (y - y_mean) * m
    pc = (p - p_mean) * m

    num = (yc * pc).sum(dim=1)
    den = torch.sqrt((yc ** 2).sum(dim=1) * (pc ** 2).sum(dim=1) + eps)

    corr = num / den
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
            print("Using weights:", p)
            df = pd.read_csv(p)
            break
    else:
        raise FileNotFoundError(f"No weights file found in {weights_dir}")

    if "track_index" not in df.columns:
        df["track_index"] = np.arange(len(df))

    if "weight" in df.columns:
        weight_col = "weight"
    elif "reconstruction_weight" in df.columns:
        weight_col = "reconstruction_weight"
    else:
        raise ValueError(f"Could not find weight column. Columns: {list(df.columns)}")

    df = df.copy()
    df["weight_for_plot"] = df[weight_col].astype(float)

    return df


def get_label(row):
    for c in ["rbp_ct", "track_label", "rbp_cell_line", "label"]:
        if c in row.index:
            return str(row[c])
    if "rbp" in row.index and "ct" in row.index:
        return f"{row['rbp']}_{row['ct']}"
    if "track_index" in row.index:
        return f"track_{row['track_index']}"
    return str(row.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default="/home/twang/storage_ml4rg26-deconvgclip/twang/repos/idea1-globalclip-deconvolution/results/reconstruction_head/interphase_50",
    )
    parser.add_argument(
        "--top-tiles",
        default="/home/twang/storage_ml4rg26-deconvgclip/twang/repos/idea1-globalclip-deconvolution/results/validation/correlation_rank_full_length/top1000_full_length_tiles_by_gclip_signal.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/twang/storage_ml4rg26-deconvgclip/twang/repos/idea1-globalclip-deconvolution/results/validation/example_profile",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-n-tracks", type=int, default=5)
    parser.add_argument("--max-rank-for-example", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    top_tiles_path = Path(args.top_tiles)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print("Using device:", device)

    metadata = json.loads((run_dir / "metadata.json").read_text())
    target_path = Path(metadata["input_paths"]["target_profiles"])
    parnet_path = Path(metadata["input_paths"]["parnet_predictions"])

    print("Loading target profiles:", target_path)
    target_data = torch.load(target_path, map_location="cpu", weights_only=False)

    profiles = squeeze_profile(pick_first(target_data, ["profiles", "profile", "target_profiles"], "profiles"))
    valid_mask = squeeze_profile(pick_first(target_data, ["valid_mask", "mask"], "valid_mask")).bool()

    split_values = pick_first(target_data, ["split", "splits", "split_labels"], "split labels")
    split_values = np.array(to_string_list(split_values))

    print("Loading Parnet predictions:", parnet_path)
    parnet_data = torch.load(parnet_path, map_location="cpu", weights_only=False)

    if isinstance(parnet_data, dict):
        parnet_profiles = pick_first(
            parnet_data,
            ["parnet_total_profile", "total_profile", "profiles", "predictions"],
            "Parnet profiles",
        )
    else:
        parnet_profiles = parnet_data

    print("Target profile shape:", tuple(profiles.shape))
    print("Parnet profile shape:", tuple(parnet_profiles.shape))

    weights_df = load_weights(run_dir)
    n_tracks = int(parnet_profiles.shape[1])

    weights = np.zeros(n_tracks, dtype=np.float32)
    for _, row in weights_df.iterrows():
        idx = int(row["track_index"])
        if 0 <= idx < n_tracks:
            weights[idx] = float(row["weight_for_plot"])

    weights = weights / weights.sum()

    top_tracks = (
        weights_df.sort_values("weight_for_plot", ascending=False)
        .head(args.top_n_tracks)
        .copy()
    )
    top_track_indices = top_tracks["track_index"].astype(int).tolist()
    top_track_labels = [get_label(row) for _, row in top_tracks.iterrows()]
    top_track_weights = top_tracks["weight_for_plot"].astype(float).tolist()

    print("\nTop tracks used in plot:")
    for idx, label, w in zip(top_track_indices, top_track_labels, top_track_weights):
        print(f"  track {idx}: {label}, weight={w:.6f}")

    # Map split-local indices from top_tiles CSV to global indices in target/parnet arrays.
    split_positions = np.where(split_values == args.split)[0]
    if len(split_positions) == 0:
        raise ValueError(f"No split positions found for split={args.split}")

    top_tiles = pd.read_csv(top_tiles_path)
    top_tiles = top_tiles[top_tiles["tile_rank_by_gclip_sum"] <= args.max_rank_for_example].copy()

    if len(top_tiles) == 0:
        raise ValueError("No candidate tiles left after filtering by max rank.")

    split_indices = top_tiles["split_index"].astype(int).to_numpy()
    global_indices = split_positions[split_indices]

    # Compute reconstruction Pearson for candidate tiles.
    weights_t = torch.tensor(weights, dtype=torch.float32, device=device)

    candidate_rows = []

    for start in range(0, len(global_indices), args.batch_size):
        batch_global = global_indices[start:start + args.batch_size]

        x = parnet_profiles[batch_global].float().to(device)
        y = profiles[batch_global].float().to(device)
        m = valid_mask[batch_global].bool().to(device)

        pred = torch.einsum("bkl,k->bl", x, weights_t)
        pred = pred * m.float()
        pred = pred / pred.sum(dim=1, keepdim=True).clamp_min(1e-8)

        y = y * m.float()
        y = y / y.sum(dim=1, keepdim=True).clamp_min(1e-8)

        corr = masked_pearson_batch(y, pred, m).detach().cpu().numpy()

        for j, gidx in enumerate(batch_global):
            row = top_tiles.iloc[start + j].to_dict()
            row["global_index"] = int(gidx)
            row["reconstruction_pearson"] = float(corr[j])
            candidate_rows.append(row)

    candidates = pd.DataFrame(candidate_rows)
    candidates = candidates.sort_values(
        ["reconstruction_pearson", "gclip_raw_sum"],
        ascending=[False, False],
    )

    candidates.to_csv(out_dir / "candidate_example_tiles.csv", index=False)

    chosen = candidates.iloc[0]
    global_idx = int(chosen["global_index"])
    split_idx = int(chosen["split_index"])
    tile_name = str(chosen["name"])
    gclip_sum = float(chosen["gclip_raw_sum"])
    recon_corr = float(chosen["reconstruction_pearson"])

    print("\nChosen example tile:")
    print("  name:", tile_name)
    print("  split index:", split_idx)
    print("  global index:", global_idx)
    print("  gCLIP raw sum:", gclip_sum)
    print("  reconstruction Pearson:", recon_corr)

    # Prepare profiles for plotting.
    x = parnet_profiles[global_idx].float()
    y = profiles[global_idx].float()
    m = valid_mask[global_idx].bool()

    weights_cpu = torch.tensor(weights, dtype=torch.float32)
    pred = torch.einsum("kl,k->l", x, weights_cpu)
    pred = pred * m.float()
    pred = pred / pred.sum().clamp_min(1e-8)

    y = y * m.float()
    y = y / y.sum().clamp_min(1e-8)

    pos = np.arange(y.shape[0])

    y_np = y.numpy()
    pred_np = pred.numpy()

    # Figure: observed/reconstructed + top tracks.
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.2, 5.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.8]},
    )

    axes[0].plot(pos, y_np, label="Observed globalCLIP", linewidth=1.6)
    axes[0].plot(pos, pred_np, label="Reconstructed globalCLIP", linewidth=1.6)
    axes[0].set_ylabel("Normalized profile", fontsize=9)
    axes[0].set_title(
        f"Example reconstruction at high-signal tile\n{tile_name} | Pearson={recon_corr:.3f} | raw gCLIP sum={gclip_sum:.0f}",
        fontsize=10,
    )
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.25)

    for idx, label, w in zip(top_track_indices, top_track_labels, top_track_weights):
        track_profile = x[idx].numpy()
        axes[1].plot(pos, track_profile, linewidth=1.2, label=f"{label} (w={w:.3f})")

    axes[1].set_xlabel("Position in 600-nt window", fontsize=9)
    axes[1].set_ylabel("Parnet profile", fontsize=9)
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()

    pdf_path = out_dir / "example_reconstruction_profile.pdf"
    png_path = out_dir / "example_reconstruction_profile.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved:")
    print(" ", pdf_path)
    print(" ", png_path)
    print(" ", out_dir / "candidate_example_tiles.csv")


if __name__ == "__main__":
    main()
