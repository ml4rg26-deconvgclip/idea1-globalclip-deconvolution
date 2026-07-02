#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch
import pandas as pd


def sparse_dict_to_dense(x):
    if isinstance(x, dict):
        indices = x["indices"].long()
        values = x["values"].float()
        size = tuple(x["size"])
        return torch.sparse_coo_tensor(indices, values, size=size).coalesce().to_dense()
    if isinstance(x, torch.Tensor) and x.is_sparse:
        return x.to_dense()
    return x


def summarize_signal(x):
    dense = sparse_dict_to_dense(x).float()
    flat = dense.reshape(-1)
    return {
        "sum": float(flat.sum().item()),
        "max": float(flat.max().item()),
        "nonzero": int((flat != 0).sum().item()),
        "length": int(flat.numel()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-per-split", type=int, default=None)
    args = parser.parse_args()

    path = Path(args.path)
    print(f"Loading: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)

    rows = []

    for split in ["train", "valid", "test"]:
        samples = data[split]
        if args.max_per_split is not None:
            samples = samples[:args.max_per_split]

        print(f"Processing {split}: {len(samples)} samples")

        for i, sample in enumerate(samples):
            seq = sample["inputs"]["sequence"]
            global_stats = summarize_signal(sample["outputs"]["globalCLIP"])
            control_stats = summarize_signal(sample["outputs"]["control"])

            rows.append({
                "split": split,
                "index": i,
                "name": sample["meta"]["name"],
                "pad_side": sample["meta"]["pad_side"],
                "seq_len": len(seq),
                "alphabet": "".join(sorted(set(seq))),
                "global_sum": global_stats["sum"],
                "global_max": global_stats["max"],
                "global_nonzero": global_stats["nonzero"],
                "global_len": global_stats["length"],
                "control_sum": control_stats["sum"],
                "control_max": control_stats["max"],
                "control_nonzero": control_stats["nonzero"],
                "control_len": control_stats["length"],
            })

    df = pd.DataFrame(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\nSaved summary to: {out}")

    print("\nSplit sizes:")
    print(df.groupby("split").size())

    print("\nSequence length counts:")
    print(df["seq_len"].value_counts().head())

    print("\nGlobalCLIP total count summary:")
    print(df["global_sum"].describe())

    print("\nControl total count summary:")
    print(df["control_sum"].describe())

    print("\nTop 10 globalCLIP windows:")
    cols = ["split", "index", "name", "global_sum", "global_max", "global_nonzero", "control_sum"]
    print(df.sort_values("global_sum", ascending=False).head(10)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
