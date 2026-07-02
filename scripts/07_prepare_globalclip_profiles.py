#!/usr/bin/env python3
"""Prepare normalized globalCLIP probability profiles from raw count tracks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from profile_utils import counts_to_profile


DEFAULT_SPLITS = ("train", "valid", "test")


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required to load and save .pt globalCLIP files.") from exc
    return torch


def sparse_dict_to_dense(x: Any, *, target_len: int = 600) -> torch.Tensor:
    """Convert the sparse signal encodings used by existing globalCLIP scripts."""
    torch = require_torch()

    if isinstance(x, dict):
        if "indices" not in x or "values" not in x:
            raise ValueError(f"Sparse dict is missing indices/values keys: {list(x.keys())}")

        indices = x["indices"].long()
        values = x["values"].float()
        shape = None
        for key in ("size", "shape", "dense_shape"):
            if key in x:
                shape = x[key]
                break

        if shape is None:
            if indices.ndim == 2 and indices.shape[0] == 2:
                shape = (1, target_len)
            elif indices.ndim == 1:
                shape = (target_len,)
            else:
                raise ValueError(f"Cannot infer sparse shape from indices shape {tuple(indices.shape)}")

        if isinstance(shape, torch.Tensor):
            shape = tuple(shape.tolist())
        else:
            shape = tuple(shape)

        return torch.sparse_coo_tensor(indices, values, size=shape).coalesce().to_dense()

    if isinstance(x, torch.Tensor) and x.is_sparse:
        return x.coalesce().to_dense()

    if isinstance(x, torch.Tensor):
        return x

    return torch.as_tensor(x)


def iter_samples(data: Any, requested_split: str | None) -> list[tuple[str, int, dict[str, Any]]]:
    if isinstance(data, dict) and requested_split is None:
        available_splits = [split for split in DEFAULT_SPLITS if split in data]
        if available_splits:
            rows = []
            for split in available_splits:
                rows.extend((split, index, sample) for index, sample in enumerate(data[split]))
            return rows

    if isinstance(data, dict) and requested_split is not None:
        if requested_split not in data:
            raise KeyError(f"Split {requested_split!r} not found. Available keys: {list(data.keys())}")
        return [(requested_split, index, sample) for index, sample in enumerate(data[requested_split])]

    if isinstance(data, list):
        split = requested_split or "all"
        return [(split, index, sample) for index, sample in enumerate(data)]

    raise TypeError(
        "Expected a .pt object containing train/valid/test split lists or one list of samples; "
        f"got {type(data).__name__}"
    )


def get_signal(sample: dict[str, Any], condition: str, *, target_len: int) -> torch.Tensor:
    if "outputs" not in sample or not isinstance(sample["outputs"], dict):
        raise KeyError("Sample is missing dict field: outputs")
    if condition not in sample["outputs"]:
        raise KeyError(
            f"Condition {condition!r} not found in sample outputs. "
            f"Available output keys: {list(sample['outputs'].keys())}"
        )

    dense = sparse_dict_to_dense(sample["outputs"][condition], target_len=target_len).float()
    if dense.ndim == 1:
        return dense
    if dense.ndim == 2 and dense.shape[0] == 1:
        return dense
    if dense.ndim == 2:
        return dense

    raise ValueError(f"Expected signal shape [L], [1, L], or [N, L], got {tuple(dense.shape)}")


def sample_name(sample: dict[str, Any], fallback: str) -> str:
    meta = sample.get("meta", {})
    if isinstance(meta, dict) and "name" in meta:
        return str(meta["name"])
    return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw globalCLIP count tracks in a .pt dataset into per-window "
            "probability profiles using (counts + eps) / (counts.sum + eps * length)."
        )
    )
    parser.add_argument("--input-pt", type=Path, required=True, help="Input globalCLIP .pt file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/globalclip_profiles"),
        help="Directory for the prepared profile .pt output.",
    )
    parser.add_argument(
        "--condition",
        default="globalCLIP",
        help="Sample outputs key to normalize, e.g. globalCLIP or control.",
    )
    parser.add_argument("--eps", type=float, default=1e-6, help="Pseudocount added per position.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of windows to process for debugging.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split to process. By default, all train/valid/test splits are processed.",
    )
    parser.add_argument(
        "--target-len",
        type=int,
        default=600,
        help="Expected window length used when sparse signals omit shape metadata.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional output filename. Defaults to <input-stem>_<condition>_profiles.pt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit must be non-negative")

    torch = require_torch()

    print(f"Loading: {args.input_pt}")
    data = torch.load(args.input_pt, map_location="cpu", weights_only=False)
    rows = iter_samples(data, args.split)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No samples selected.")

    profiles = []
    total_counts = []
    split_names = []
    indices = []
    names = []

    print(f"Preparing condition: {args.condition}")
    print(f"Selected windows: {len(rows)}")
    print(f"eps: {args.eps:g}")

    for split, index, sample in rows:
        signal = get_signal(sample, args.condition, target_len=args.target_len)
        profile = counts_to_profile(signal, eps=args.eps, length_axis=-1)

        profiles.append(profile.detach().cpu().to(dtype=torch.float32))
        total_counts.append(signal.detach().cpu().float().sum(dim=-1))
        split_names.append(split)
        indices.append(index)
        names.append(sample_name(sample, f"{split}[{index}]"))

    profile_tensor = torch.stack(profiles, dim=0)
    total_counts_tensor = torch.stack(total_counts, dim=0)
    profile_sums = profile_tensor.sum(dim=-1)

    output = {
        "profiles": profile_tensor,
        "total_counts": total_counts_tensor,
        "condition": args.condition,
        "eps": args.eps,
        "length_axis": -1,
        "source": str(args.input_pt),
        "split": split_names,
        "index": torch.tensor(indices, dtype=torch.long),
        "name": names,
        "summary": {
            "profile_shape": tuple(profile_tensor.shape),
            "profile_dtype": str(profile_tensor.dtype),
            "profile_min": float(profile_tensor.min().item()),
            "profile_max": float(profile_tensor.max().item()),
            "profile_sum_min": float(profile_sums.min().item()),
            "profile_sum_mean": float(profile_sums.mean().item()),
            "profile_sum_max": float(profile_sums.max().item()),
            "total_count_min": float(total_counts_tensor.min().item()),
            "total_count_mean": float(total_counts_tensor.float().mean().item()),
            "total_count_max": float(total_counts_tensor.max().item()),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"{args.input_pt.stem}_{args.condition}_profiles.pt"
    output_path = args.output_dir / output_name
    if output_path.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output_path}")

    torch.save(output, output_path)

    print("\nProfile tensor:")
    print(f"  shape: {tuple(profile_tensor.shape)}")
    print(f"  dtype: {profile_tensor.dtype}")
    print(f"  min:   {profile_tensor.min().item():.6g}")
    print(f"  max:   {profile_tensor.max().item():.6g}")
    print("Profile sums over length axis:")
    print(f"  min:  {profile_sums.min().item():.8f}")
    print(f"  mean: {profile_sums.mean().item():.8f}")
    print(f"  max:  {profile_sums.max().item():.8f}")
    print("Total counts per window:")
    print(f"  shape: {tuple(total_counts_tensor.shape)}")
    print(f"  min:   {total_counts_tensor.min().item():.6g}")
    print(f"  mean:  {total_counts_tensor.float().mean().item():.6g}")
    print(f"  max:   {total_counts_tensor.max().item():.6g}")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
