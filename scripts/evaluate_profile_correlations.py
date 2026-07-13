#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch


def load_torch_dict(path: Path) -> dict:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise SystemExit(f"Expected a dict in {path}, got {type(obj)}")
    return obj


def pick_key(d: dict, keys: list[str], what: str):
    for k in keys:
        if k in d:
            return d[k]
    raise SystemExit(f"Could not find {what}. Tried keys: {keys}. Available keys: {list(d.keys())}")


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def squeeze_profile(x):
    x = to_numpy(x)
    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]
    return x


def split_to_strings(x) -> list[str]:
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


def read_weights(path: Path, n_tracks: int) -> np.ndarray:
    weights = np.zeros(n_tracks, dtype=np.float64)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["track_index"])
            weights[idx] = float(row["weight"])
    s = weights.sum()
    if not np.isfinite(s) or s <= 0:
        raise SystemExit(f"Invalid weights from {path}; sum={s}")
    weights = weights / s
    return weights


def average_rankdata(a: np.ndarray) -> np.ndarray:
    """Tie-aware average ranks, 1-based, similar to scipy.stats.rankdata(method='average')."""
    a = np.asarray(a)
    n = len(a)
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    ranks = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2:
        return math.nan
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom <= 0 or not np.isfinite(denom):
        return math.nan
    return float(np.sum(a * b) / denom)


def renormalize_valid(p: np.ndarray, mask: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    p = np.where(mask, p, 0.0)
    s = p.sum(axis=1, keepdims=True)
    return np.divide(p, s, out=np.zeros_like(p), where=s > 0)


def compute_row_metrics(obs: np.ndarray, pred: np.ndarray, mask: np.ndarray):
    y = obs[mask]
    p = pred[mask]

    if len(y) == 0:
        return math.nan, math.nan, math.nan

    mse = float(np.mean((y - p) ** 2))
    pearson = safe_corr(y, p)
    spearman = safe_corr(average_rankdata(y), average_rankdata(p))
    return pearson, spearman, mse


def summarize(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return math.nan, math.nan
    return float(np.mean(arr)), float(np.median(arr))


def fmt(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.10f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("results/reconstruction_head/interphase_50"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        help="Use 'test' first. Later you can run: --splits train valid test",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    run_dir = args.run_dir
    metadata_path = run_dir / "metadata.json"
    weights_path = run_dir / "reconstruction_weights_track_level.csv"
    out_path = run_dir / "profile_correlation_metrics.csv"

    metadata = json.loads(metadata_path.read_text())
    parnet_path = Path(metadata["input_paths"]["parnet_predictions"])
    target_path = Path(metadata["input_paths"]["target_profiles"])

    print(f"Loading target profiles: {target_path}")
    target_data = load_torch_dict(target_path)
    obs = squeeze_profile(pick_key(target_data, ["profiles", "profile", "target_profiles"], "observed profiles")).astype(np.float64)
    valid_mask = squeeze_profile(pick_key(target_data, ["valid_mask", "mask"], "valid mask")).astype(bool)
    splits = split_to_strings(pick_key(target_data, ["split", "splits", "split_labels"], "split labels"))

    print(f"Loading Parnet profiles: {parnet_path}")
    parnet_data = load_torch_dict(parnet_path)
    parnet_profiles = pick_key(
        parnet_data,
        ["parnet_total_profile", "total_profile", "profiles", "predictions"],
        "Parnet total profile",
    )

    if parnet_profiles.shape[0] != obs.shape[0]:
        raise SystemExit(f"N mismatch: Parnet {parnet_profiles.shape[0]} vs observed {obs.shape[0]}")

    n_windows, n_tracks, length = parnet_profiles.shape
    weights = read_weights(weights_path, n_tracks)
    weights_t = torch.tensor(weights, dtype=torch.float32)

    splits_arr = np.asarray(splits)
    train_idx = np.where(splits_arr == "train")[0]
    if len(train_idx) == 0:
        raise SystemExit("Could not find train split in target profiles.")

    train_obs = obs[train_idx]
    train_mask = valid_mask[train_idx]
    train_sum = (train_obs * train_mask).sum(axis=0)
    train_count = train_mask.sum(axis=0)
    train_mean = np.divide(train_sum, train_count, out=np.zeros_like(train_sum), where=train_count > 0)
    if train_mean.sum() > 0:
        train_mean = train_mean / train_mean.sum()

    rows = []

    for split in args.splits:
        idx = np.where(splits_arr == split)[0]
        if len(idx) == 0:
            print(f"Skipping split={split}; no windows found.")
            continue

        print(f"Evaluating split={split}, n={len(idx)}")

        metrics = {
            "reconstruction_head": {"pearson": [], "spearman": [], "mse": []},
            "parnet_unweighted_average": {"pearson": [], "spearman": [], "mse": []},
            "uniform_valid_profile": {"pearson": [], "spearman": [], "mse": []},
            "train_mean_observed_profile": {"pearson": [], "spearman": [], "mse": []},
        }

        for start in range(0, len(idx), args.batch_size):
            batch_idx = idx[start : start + args.batch_size]
            obs_b = obs[batch_idx]
            mask_b = valid_mask[batch_idx]

            x = parnet_profiles[batch_idx].float()

            pred_recon = torch.einsum("bkl,k->bl", x, weights_t).cpu().numpy()
            pred_recon = renormalize_valid(pred_recon, mask_b)

            pred_unweighted = x.mean(dim=1).cpu().numpy()
            pred_unweighted = renormalize_valid(pred_unweighted, mask_b)

            pred_uniform = np.zeros_like(obs_b, dtype=np.float64)
            valid_counts = mask_b.sum(axis=1)
            for r in range(len(batch_idx)):
                if valid_counts[r] > 0:
                    pred_uniform[r, mask_b[r]] = 1.0 / valid_counts[r]

            pred_train_mean = np.tile(train_mean[None, :], (len(batch_idx), 1))
            pred_train_mean = renormalize_valid(pred_train_mean, mask_b)

            pred_map = {
                "reconstruction_head": pred_recon,
                "parnet_unweighted_average": pred_unweighted,
                "uniform_valid_profile": pred_uniform,
                "train_mean_observed_profile": pred_train_mean,
            }

            for model_name, pred_b in pred_map.items():
                for r in range(len(batch_idx)):
                    pearson, spearman, mse = compute_row_metrics(obs_b[r], pred_b[r], mask_b[r])
                    metrics[model_name]["pearson"].append(pearson)
                    metrics[model_name]["spearman"].append(spearman)
                    metrics[model_name]["mse"].append(mse)

        for model_name, m in metrics.items():
            mean_p, med_p = summarize(m["pearson"])
            mean_s, med_s = summarize(m["spearman"])
            mean_mse, med_mse = summarize(m["mse"])

            rows.append(
                {
                    "split": split,
                    "model": model_name,
                    "n_windows": len(idx),
                    "mean_pearson": mean_p,
                    "median_pearson": med_p,
                    "mean_spearman": mean_s,
                    "median_spearman": med_s,
                    "mean_mse": mean_mse,
                    "median_mse": med_mse,
                }
            )

    with out_path.open("w", newline="") as f:
        fieldnames = [
            "split",
            "model",
            "n_windows",
            "mean_pearson",
            "median_pearson",
            "mean_spearman",
            "median_spearman",
            "mean_mse",
            "median_mse",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "split": row["split"],
                    "model": row["model"],
                    "n_windows": row["n_windows"],
                    "mean_pearson": fmt(row["mean_pearson"]),
                    "median_pearson": fmt(row["median_pearson"]),
                    "mean_spearman": fmt(row["mean_spearman"]),
                    "median_spearman": fmt(row["median_spearman"]),
                    "mean_mse": fmt(row["mean_mse"]),
                    "median_mse": fmt(row["median_mse"]),
                }
            )

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
