from pathlib import Path
import argparse
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stability", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-n", type=int, default=8)
    args = parser.parse_args()

    stability_path = Path(args.stability)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(stability_path)

    # keep top 8 by mean_weight / stability order
    df = df.sort_values(
        ["selected_top5", "selected_top8", "mean_weight"],
        ascending=[False, False, False],
    ).head(args.top_n).copy()

    labels = df["rbp_ct"].astype(str).tolist()
    top5 = df["selected_top5"].astype(float).to_numpy()
    top8 = df["selected_top8"].astype(float).to_numpy()

    # reverse so strongest appears at top
    labels = labels[::-1]
    top5 = top5[::-1]
    top8 = top8[::-1]

    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9.6, 6.4))

    bar_h = 0.28
    offset = 0.18

    ax.barh(y - offset, top5, height=bar_h, label="Selected in top 5")
    ax.barh(y + offset, top8, height=bar_h, label="Selected in top 8")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Number of runs out of 10 random seeds")
    ax.set_title("Selection stability across random seeds", pad=34, fontsize=17)

    ax.set_xlim(0, 10.8)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Put legend inside an empty upper-left area, not on the bars
    ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=2,
    frameon=False,
    columnspacing=1.5,
    handletextpad=0.6,)


    # label only non-zero bars
    for vals, yy in [(top5, y - offset), (top8, y + offset)]:
        for v, ypos in zip(vals, yy):
            if v <= 0:
                continue
            label = f"{int(v)}/10"

            if v >= 7:
                ax.text(
                    v - 0.18,
                    ypos,
                    label,
                    va="center",
                    ha="right",
                    color="white",
                    fontsize=12,
                    fontweight="bold",
                )
            else:
                ax.text(
                    v + 0.12,
                    ypos,
                    label,
                    va="center",
                    ha="left",
                    color="black",
                    fontsize=12,
                    fontweight="bold",
                )

    fig.subplots_adjust(left=0.24, right=0.98, top=0.84, bottom=0.14)

    pdf_path = out_dir / "top8_stability_across_seeds_clean.pdf"
    png_path = out_dir / "top8_stability_across_seeds_clean.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    print("Saved:")
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
