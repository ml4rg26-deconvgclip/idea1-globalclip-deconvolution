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


def optional_key(d: dict, keys: list[str]):
    for k in keys:
        if k in d:
            return d[k]
    return None


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


def names_to_strings(x, n: int) -> list[str]:
    if x is None:
        return ["" for _ in range(n)]
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
    if len(out) != n:
        return ["" for _ in range(n)]
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
    return weights / s


def average_rankdata(a: np.ndarray) -> np.ndarray:
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


def assign_quartiles(values: np.ndarray, idx: np.ndarray) -> dict[int, str]:
    split_values = values[idx].astype(float)
    ranks = np.argsort(np.argsort(split_values, kind="mergesort"), kind="mergesort")
    n = len(split_values)

    quartile_map = {}
    for local_i, global_i in enumerate(idx):
        q = int(np.floor(4 * ranks[local_i] / max(n, 1))) + 1
        q = min(q, 4)
        quartile_map[int(global_i)] = f"Q{q}"
    return quartile_map


def fmt(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.10f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    run_dir = args.run_dir
    metadata = json.loads((run_dir / "metadata.json").read_text())

    parnet_path = Path(metadata["input_paths"]["parnet_predictions"])
    target_path = Path(metadata["input_paths"]["target_profiles"])
    weights_path = run_dir / "reconstruction_weights_track_level.csv"
    out_path = run_dir / "per_window_profile_correlations.csv"

    print(f"Loading target profiles: {target_path}")
    target_data = load_torch_dict(target_path)

    obs = squeeze_profile(
        pick_key(target_data, ["profiles", "profile", "target_profiles"], "observed profiles")
    ).astype(np.float64)

    valid_mask = squeeze_profile(
        pick_key(target_data, ["valid_mask", "mask"], "valid mask")
    ).astype(bool)

    splits = split_to_strings(
        pick_key(target_data, ["split", "splits", "split_labels"], "split labels")
    )

    names = names_to_strings(
        optional_key(target_data, ["name", "names", "window_names", "intervals"]),
        n=obs.shape[0],
    )

    total_counts_raw = optional_key(target_data, ["total_counts", "counts", "globalclip_total_counts"])
    if total_counts_raw is not None:
        total_counts = to_numpy(total_counts_raw).reshape(-1).astype(float)
    else:
        total_counts = np.sum(obs * valid_mask, axis=1).astype(float)

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
        quartile_map = assign_quartiles(total_counts, idx)

        for start in range(0, len(idx), args.batch_size):
            batch_idx = idx[start:start + args.batch_size]

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
                for r, global_i in enumerate(batch_idx):
                    pearson, spearman, mse = compute_row_metrics(obs_b[r], pred_b[r], mask_b[r])

                    rows.append({
                        "split": split,
                        "window_index": int(global_i),
                        "window_name": names[int(global_i)],
                        "model": model_name,
                        "gclip_total_count": fmt(float(total_counts[int(global_i)])),
                        "gclip_signal_quartile": quartile_map[int(global_i)],
                        "pearson": fmt(pearson),
                        "spearman": fmt(spearman),
                        "mse": fmt(mse),
                    })

    with out_path.open("w", newline="") as f:
        fieldnames = [
            "split",
            "window_index",
            "window_name",
            "model",
            "gclip_total_count",
            "gclip_signal_quartile",
            "pearson",
            "spearman",
            "mse",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out_path}")
    print(f"Rows written: {len(rows)}")


if __name__ == "__main__":
    main()
