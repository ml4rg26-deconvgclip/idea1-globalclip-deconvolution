#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def sparse_dict_to_dense(x):
    if isinstance(x, dict):
        indices = x["indices"].long()
        values = x["values"].float()
        size = tuple(x["size"])
        return torch.sparse_coo_tensor(indices, values, size=size).coalesce().to_dense()
    if isinstance(x, torch.Tensor) and x.is_sparse:
        return x.to_dense()
    return x


def get_dense_signal(sample, key):
    x = sample["outputs"][key]
    dense = sparse_dict_to_dense(x).float()
    return dense.reshape(-1).numpy()


def safe_filename(text):
    return (
        text.replace(":", "_")
        .replace("-", "_")
        .replace("+", "plus")
        .replace("/", "_")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to globalCLIP .pt file")
    parser.add_argument("--candidates", required=True, help="Candidate windows CSV")
    parser.add_argument("--outdir", required=True, help="Output plot directory")
    parser.add_argument("--n", type=int, default=5, help="Number of candidate windows to plot")
    args = parser.parse_args()

    data_path = Path(args.data)
    candidate_path = Path(args.candidates)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data: {data_path}")
    data = torch.load(data_path, map_location="cpu", weights_only=False)

    candidates = pd.read_csv(candidate_path).head(args.n)
    print(f"Plotting {len(candidates)} candidate windows")

    for row_id, row in candidates.iterrows():
        split = row["split"]
        index = int(row["index"])
        name = row["name"]

        sample = data[split][index]

        seq = sample["inputs"]["sequence"]
        global_signal = get_dense_signal(sample, "globalCLIP")
        control_signal = get_dense_signal(sample, "control")

        positions = np.arange(1, len(global_signal) + 1)

        print(
            f"[{row_id}] {split}[{index}] {name} | "
            f"seq_len={len(seq)} | "
            f"global_sum={global_signal.sum():.1f} | "
            f"control_sum={control_signal.sum():.1f}"
        )

        plt.figure(figsize=(12, 4))
        plt.plot(positions, global_signal, label="globalCLIP")
        plt.plot(positions, control_signal, label="control")
        plt.xlabel("Position in 600 nt window")
        plt.ylabel("Count")
        plt.title(
            f"{name} | {split}[{index}] | "
            f"global={global_signal.sum():.0f}, control={control_signal.sum():.0f}"
        )
        plt.legend()
        plt.tight_layout()

        filename = f"{row_id:02d}_{split}_{index}_{safe_filename(name)}.png"
        plt.savefig(outdir / filename, dpi=200)
        plt.close()

    print(f"\nSaved profile plots to: {outdir}")


if __name__ == "__main__":
    main()
