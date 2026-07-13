from pathlib import Path
import argparse
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
})


def pick_first(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def make_track_label(df):
    c = pick_first(df.columns, ["track_label", "rbp_ct", "rbp_ct_x", "rbp_ct_y", "label"])
    if c is not None:
        return df[c].astype(str)

    if "rbp" in df.columns and "ct" in df.columns:
        return df["rbp"].astype(str) + "_" + df["ct"].astype(str)

    if "rbp_x" in df.columns and "ct_x" in df.columns:
        return df["rbp_x"].astype(str) + "_" + df["ct_x"].astype(str)

    return "track_" + df["track_index"].astype(str)


def save_barh(ax, labels, values, title, xlabel, out_pdf, out_png, annotate_fmt="{:.3f}", extra_text=None, xlim=None):
    labels = list(labels)
    values = np.asarray(values, dtype=float)

    order = np.arange(len(labels))
    bars = ax.barh(order, values)
    ax.set_yticks(order)
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25)

    if xlim is not None:
        ax.set_xlim(*xlim)

    xmax = ax.get_xlim()[1]
    pad = xmax * 0.015

    for i, (bar, v) in enumerate(zip(bars, values)):
        txt = annotate_fmt.format(v)
        if extra_text is not None:
            txt += extra_text(i)
        ax.text(
            min(v + pad, xmax * 0.98),
            bar.get_y() + bar.get_height() / 2,
            txt,
            va="center",
            ha="left",
            fontsize=11,
        )

    plt.tight_layout()
    ax.figure.savefig(out_pdf, bbox_inches="tight")
    ax.figure.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(ax.figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--stability", required=True)
    parser.add_argument("--correlation", required=True)
    parser.add_argument("--concentration", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-n", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- weights ----------
    weights = pd.read_csv(args.weights)
    if "track_index" not in weights.columns:
        weights["track_index"] = np.arange(len(weights))

    wcol = pick_first(weights.columns, ["reconstruction_weight", "weight"])
    if wcol is None:
        raise ValueError(f"No weight column found in weights file. Columns: {list(weights.columns)}")

    weights = weights.copy()
    weights["track_label"] = make_track_label(weights)
    weights["reconstruction_weight"] = weights[wcol].astype(float)
    weights = weights[["track_index", "track_label", "reconstruction_weight"]]
    weights = weights.sort_values("reconstruction_weight", ascending=False).reset_index(drop=True)

    # ---------- stability ----------
    stability = pd.read_csv(args.stability)
    keep_stab = [
        "track_index",
        "selected_top5",
        "selected_top8",
        "selected_top10",
        "top5_frequency",
        "top8_frequency",
        "top10_frequency",
        "mean_rank",
        "mean_weight",
        "n_seeds",
    ]
    stability = stability[[c for c in keep_stab if c in stability.columns]].copy()

    # ---------- correlation ----------
    corr = pd.read_csv(args.correlation)
    keep_corr = [
        "track_index",
        "mean_corr_top1000",
        "median_corr_rank_top1000",
        "fraction_rank_top10",
        "fraction_rank_top20",
    ]
    corr = corr[[c for c in keep_corr if c in corr.columns]].copy()

    merged = weights.merge(stability, on="track_index", how="left")
    merged = merged.merge(corr, on="track_index", how="left")

    top = merged.head(args.top_n).copy()
    top.insert(0, "rank_by_weight", np.arange(1, len(top) + 1))
    top["fraction_rank_top20_pct"] = 100.0 * top["fraction_rank_top20"]

    # round for report table
    table = top.copy()
    table["reconstruction_weight"] = table["reconstruction_weight"].round(6)
    if "mean_weight" in table.columns:
        table["mean_weight"] = table["mean_weight"].round(6)
    if "mean_rank" in table.columns:
        table["mean_rank"] = table["mean_rank"].round(2)
    if "mean_corr_top1000" in table.columns:
        table["mean_corr_top1000"] = table["mean_corr_top1000"].round(3)
    if "median_corr_rank_top1000" in table.columns:
        table["median_corr_rank_top1000"] = table["median_corr_rank_top1000"].round(1)
    table["fraction_rank_top20_pct"] = table["fraction_rank_top20_pct"].round(1)

    table_cols = [
        "rank_by_weight",
        "track_label",
        "reconstruction_weight",
        "top5_frequency",
        "top8_frequency",
        "mean_corr_top1000",
        "median_corr_rank_top1000",
        "fraction_rank_top20_pct",
    ]
    table_cols = [c for c in table_cols if c in table.columns]
    table_out = table[table_cols].copy()

    table_out.to_csv(out_dir / "top8_summary_for_report.csv", index=False)
    with open(out_dir / "top8_summary_for_report.txt", "w") as f:
        f.write(table_out.to_string(index=False))

    # ---------- key numbers ----------
    if args.concentration:
        conc_path = Path(args.concentration)
        if conc_path.exists():
            conc = pd.read_csv(conc_path)
            with open(out_dir / "report_key_numbers.txt", "w") as f:
                f.write("Weight concentration summary\n")
                f.write("=" * 32 + "\n")
                for _, row in conc.iterrows():
                    f.write(f"{row['metric']}: {row['value']}\n")

    # ---------- plot 1: weights ----------
    top_rev = top.iloc[::-1].copy()

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    x_max = max(0.32, float(top["reconstruction_weight"].max()) * 1.28)
    save_barh(
        ax=ax,
        labels=top_rev["track_label"],
        values=top_rev["reconstruction_weight"],
        title="Top 8 reconstruction weights",
        xlabel="Reconstruction weight",
        out_pdf=out_dir / "top8_reconstruction_weights.pdf",
        out_png=out_dir / "top8_reconstruction_weights.png",
        annotate_fmt="{:.3f}",
        xlim=(0, x_max),
    )

    # ---------- plot 2: correlation support ----------
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    vals = top_rev["fraction_rank_top20_pct"].fillna(0).to_numpy(dtype=float)
    medr = top_rev["median_corr_rank_top1000"].fillna(np.nan).to_numpy(dtype=float)

    def corr_extra(i):
        if np.isnan(medr[i]):
            return ""
        return f"   | median rank {medr[i]:.0f}"

    save_barh(
        ax=ax,
        labels=top_rev["track_label"],
        values=vals,
        title="Direct eCLIP–globalCLIP correlation support",
        xlabel="% of top 1000 high-signal tiles\nwhere the track ranked in the top 20 by Pearson correlation",
        out_pdf=out_dir / "top8_direct_correlation_support.pdf",
        out_png=out_dir / "top8_direct_correlation_support.png",
        annotate_fmt="{:.1f}%",
        extra_text=corr_extra,
        xlim=(0, max(70, float(np.nanmax(vals)) * 1.35 if len(vals) else 70)),
    )

        # ---------- plot 3: stability ----------
    fig, ax = plt.subplots(figsize=(9.8, 6.8))

    y = np.arange(len(top_rev))
    top5 = top_rev["selected_top5"].fillna(0).to_numpy(dtype=float)
    top8 = top_rev["selected_top8"].fillna(0).to_numpy(dtype=float)

    bar_h = 0.25
    offset = 0.16

    ax.barh(
        y - offset,
        top5,
        height=bar_h,
        label="Selected in top 5"
    )

    ax.barh(
        y + offset,
        top8,
        height=bar_h,
        label="Selected in top 8"
    )

    ax.set_yticks(y)
    ax.set_yticklabels(top_rev["track_label"])
    ax.set_xlabel("Number of runs (out of 10 seeds)")
    ax.set_title("Selection stability across 10 random seeds", pad=18)
    ax.set_xlim(0, 10.9)
    ax.grid(axis="x", alpha=0.25)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False
    )

    def annotate(vals, ypos):
        for v, yy in zip(vals, ypos):
            label = f"{int(v)}/10"

            if v >= 1.2:
                ax.text(
                    v - 0.15,
                    yy,
                    label,
                    va="center",
                    ha="right",
                    color="white",
                    fontsize=11,
                    fontweight="bold",
                )
            elif v > 0:
                ax.text(
                    v + 0.12,
                    yy,
                    label,
                    va="center",
                    ha="left",
                    color="black",
                    fontsize=11,
                )
            # Do not show 0/10 labels; they make the figure crowded.

    annotate(top5, y - offset)
    annotate(top8, y + offset)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])
    fig.savefig(out_dir / "top8_stability_across_seeds.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "top8_stability_across_seeds.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved files in:", out_dir)
    for p in sorted(out_dir.iterdir()):
        print(" ", p.name)


if __name__ == "__main__":

    main()
