#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--full-length-only", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    if args.full_length_only:
        df = df[(df["seq_len"] == 600) & (df["pad_side"] == -1)].copy()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df["log1p_global_sum"] = np.log1p(df["global_sum"])
    df["log1p_control_sum"] = np.log1p(df["control_sum"])
    df["log1p_global_max"] = np.log1p(df["global_max"])
    df["global_control_ratio"] = (df["global_sum"] + 1) / (df["control_sum"] + 1)

    # 1. globalCLIP total count distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df["log1p_global_sum"], bins=60)
    plt.xlabel("log1p(globalCLIP total count)")
    plt.ylabel("Number of windows")
    plt.title("Distribution of globalCLIP total counts")
    plt.tight_layout()
    plt.savefig(outdir / "hist_log1p_global_sum.png", dpi=200)
    plt.close()

    # 2. control total count distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df["log1p_control_sum"], bins=60)
    plt.xlabel("log1p(control total count)")
    plt.ylabel("Number of windows")
    plt.title("Distribution of control total counts")
    plt.tight_layout()
    plt.savefig(outdir / "hist_log1p_control_sum.png", dpi=200)
    plt.close()

    # 3. overlay globalCLIP and control
    plt.figure(figsize=(8, 5))
    plt.hist(df["log1p_global_sum"], bins=60, alpha=0.6, label="globalCLIP")
    plt.hist(df["log1p_control_sum"], bins=60, alpha=0.6, label="control")
    plt.xlabel("log1p(total count)")
    plt.ylabel("Number of windows")
    plt.title("globalCLIP vs control total count distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "hist_overlay_global_vs_control.png", dpi=200)
    plt.close()

    # 4. scatter globalCLIP vs control
    plt.figure(figsize=(6, 6))
    plt.scatter(df["log1p_global_sum"], df["log1p_control_sum"], s=4, alpha=0.25)
    plt.xlabel("log1p(globalCLIP total count)")
    plt.ylabel("log1p(control total count)")
    plt.title("globalCLIP vs control total counts")
    plt.tight_layout()
    plt.savefig(outdir / "scatter_global_vs_control.png", dpi=200)
    plt.close()

    # 5. ratio distribution
    plt.figure(figsize=(8, 5))
    plt.hist(np.log2(df["global_control_ratio"]), bins=60)
    plt.xlabel("log2((globalCLIP + 1) / (control + 1))")
    plt.ylabel("Number of windows")
    plt.title("globalCLIP enrichment over control")
    plt.tight_layout()
    plt.savefig(outdir / "hist_log2_global_control_ratio.png", dpi=200)
    plt.close()

    # 6. sequence length distribution from original summary
    seq_len_counts = pd.read_csv(args.csv)["seq_len"].value_counts().sort_index()
    plt.figure(figsize=(10, 5))
    plt.bar(seq_len_counts.index.astype(str), seq_len_counts.values)
    plt.xlabel("Sequence length")
    plt.ylabel("Number of windows")
    plt.title("Sequence length distribution")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(outdir / "seq_len_distribution.png", dpi=200)
    plt.close()

    # Candidate windows for later detailed profile plots.
    # We avoid very extreme outliers and only use full-length windows.
    candidates = df[
        (df["seq_len"] == 600)
        & (df["pad_side"] == -1)
        & (df["global_sum"] >= df["global_sum"].quantile(0.90))
        & (df["global_sum"] <= df["global_sum"].quantile(0.995))
    ].copy()

    candidates["global_control_ratio"] = (candidates["global_sum"] + 1) / (candidates["control_sum"] + 1)

    top_by_ratio = candidates.sort_values(
        ["global_control_ratio", "global_sum"],
        ascending=False
    ).head(20)

    top_by_global = candidates.sort_values(
        "global_sum",
        ascending=False
    ).head(20)

    top_by_ratio.to_csv(outdir / "candidate_windows_high_global_vs_control.csv", index=False)
    top_by_global.to_csv(outdir / "candidate_windows_high_global.csv", index=False)

    with open(outdir / "summary.txt", "w") as f:
        f.write("Data used for plots\n")
        f.write(f"n_windows = {len(df)}\n")
        f.write(f"full_length_only = {args.full_length_only}\n\n")

        f.write("global_sum summary\n")
        f.write(str(df["global_sum"].describe()))
        f.write("\n\ncontrol_sum summary\n")
        f.write(str(df["control_sum"].describe()))
        f.write("\n\nTop candidate windows by global/control ratio\n")
        f.write(top_by_ratio[["split", "index", "name", "global_sum", "control_sum", "global_control_ratio"]].to_string(index=False))

    print(f"Saved plots and candidate windows to: {outdir}")
    print("\nTop candidate windows by global/control ratio:")
    print(top_by_ratio[["split", "index", "name", "global_sum", "control_sum", "global_control_ratio"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
