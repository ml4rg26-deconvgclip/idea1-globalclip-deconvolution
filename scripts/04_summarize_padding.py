#!/usr/bin/env python3

import argparse
from pathlib import Path
import torch
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data = torch.load(Path(args.path), map_location="cpu", weights_only=False)

    rows = []
    for split in ["train", "valid", "test"]:
        for i, sample in enumerate(data[split]):
            seq = sample["inputs"]["sequence"]
            rows.append({
                "split": split,
                "index": i,
                "name": sample["meta"]["name"],
                "pad_side": sample["meta"]["pad_side"],
                "seq_len": len(seq),
                "needs_padding": len(seq) != 600,
            })

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Saved to {out}")

    print("\nSequence length summary:")
    print(df["seq_len"].describe())

    print("\nSequence length counts, top 20:")
    print(df["seq_len"].value_counts().head(20))

    print("\nPad side counts:")
    print(df["pad_side"].value_counts(dropna=False))

    print("\nPadding needed by split:")
    print(pd.crosstab(df["split"], df["needs_padding"]))

    print("\nExamples needing padding:")
    print(df[df["needs_padding"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
