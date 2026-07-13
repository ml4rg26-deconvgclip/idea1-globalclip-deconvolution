from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def find_weight_file(seed_dir):
    candidates = [
        seed_dir / "top30_track_weights_annotated.csv",
        seed_dir / "top30_track_weights.csv",
        seed_dir / "reconstruction_weights_track_level_annotated.csv",
        seed_dir / "reconstruction_weights_track_level.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stability-dir", required=True)
    parser.add_argument(
        "--rbp-cts",
        default="/home/twang/storage_ml4rg26-deconvgclip/provided_data/600nt_globalCLIP_synchronized_datasets/eCLIP_600bp_signalfiltered.rbp_cts.tsv",
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    stability_dir = Path(args.stability_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rbp = pd.read_csv(args.rbp_cts, sep="\t")
    rbp["track_index"] = np.arange(len(rbp))

    seed_dirs = sorted(
        [p for p in stability_dir.glob("seed_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[-1])
    )

    print("Found seed dirs:", [p.name for p in seed_dirs])

    all_weights = []
    all_eval = []

    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.split("_")[-1])

        weights_path = find_weight_file(seed_dir)
        if weights_path is None:
            print(f"No weight file found in {seed_dir}, skipping")
            continue

        print(f"Reading seed {seed}: {weights_path.name}")

        df = pd.read_csv(weights_path)

        if "track_index" not in df.columns:
            df["track_index"] = np.arange(len(df))

        if "weight" in df.columns:
            weight_col = "weight"
        elif "reconstruction_weight" in df.columns:
            weight_col = "reconstruction_weight"
        else:
            raise ValueError(f"No weight column in {weights_path}. Columns: {list(df.columns)}")

        df = df.copy()
        df["weight"] = df[weight_col].astype(float)
        df = df.sort_values("weight", ascending=False).copy()
        df["rank"] = np.arange(1, len(df) + 1)
        df["seed"] = seed

        all_weights.append(df[["seed", "track_index", "rank", "weight"]])

        eval_path = seed_dir / "evaluation_metrics.csv"
        if eval_path.exists():
            ev = pd.read_csv(eval_path)
            ev["seed"] = seed
            all_eval.append(ev)

    if not all_weights:
        raise SystemExit("No seed weight files found.")

    all_df = pd.concat(all_weights, ignore_index=True)
    all_df = all_df.merge(rbp, on="track_index", how="left")
    all_df.to_csv(out_dir / "all_seed_weights_ranked.csv", index=False)

    n_seeds = all_df["seed"].nunique()

    summary = (
        all_df.groupby(["track_index", "rbp_ct", "rbp", "ct"], dropna=False)
        .agg(
            selected_top5=("rank", lambda x: int((x <= 5).sum())),
            selected_top8=("rank", lambda x: int((x <= 8).sum())),
            selected_top10=("rank", lambda x: int((x <= 10).sum())),
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            mean_weight=("weight", "mean"),
            median_weight=("weight", "median"),
            max_weight=("weight", "max"),
            min_weight=("weight", "min"),
            n_seen=("seed", "count"),
        )
        .reset_index()
    )

    summary["n_seeds"] = n_seeds
    summary["top5_frequency"] = summary["selected_top5"].astype(str) + "/" + str(n_seeds)
    summary["top8_frequency"] = summary["selected_top8"].astype(str) + "/" + str(n_seeds)
    summary["top10_frequency"] = summary["selected_top10"].astype(str) + "/" + str(n_seeds)

    summary = summary.sort_values(
        ["selected_top5", "selected_top8", "mean_weight"],
        ascending=[False, False, False],
    )

    summary.to_csv(out_dir / "stability_summary.csv", index=False)

    if all_eval:
        eval_df = pd.concat(all_eval, ignore_index=True)
        eval_df.to_csv(out_dir / "seed_evaluation_metrics.csv", index=False)

        test_recon = eval_df[
            (eval_df["split"] == "test") &
            (eval_df["model"] == "reconstruction_head")
        ].copy()

        if len(test_recon) > 0:
            print("\nTest reconstruction loss across seeds:")
            print(test_recon[["seed", "loss"]].to_string(index=False))
            print("mean test loss:", test_recon["loss"].mean())
            print("std test loss:", test_recon["loss"].std())

    print("\nStability summary:")
    cols = [
        "track_index", "rbp_ct", "top5_frequency", "top8_frequency",
        "top10_frequency", "mean_rank", "mean_weight"
    ]
    print(summary[cols].head(20).to_string(index=False))

    print("\nSaved:")
    print(out_dir / "all_seed_weights_ranked.csv")
    print(out_dir / "stability_summary.csv")
    print(out_dir / "seed_evaluation_metrics.csv")


if __name__ == "__main__":
    main()
