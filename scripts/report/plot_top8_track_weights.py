from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find columns {candidates}. Available columns: {list(df.columns)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="results/reconstruction_head/interphase_50")
    parser.add_argument("--top-n", type=int, default=8)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    candidate_files = [
        run_dir / "reconstruction_weights_track_level_annotated.csv",
        run_dir / "top30_track_weights_annotated.csv",
        run_dir / "reconstruction_weights_track_level.csv",
        run_dir / "top30_track_weights.csv",
    ]

    weight_file = None
    for f in candidate_files:
        if f.exists():
            weight_file = f
            break

    if weight_file is None:
        raise FileNotFoundError(
            "No weight CSV found. Checked:\n" +
            "\n".join(str(f) for f in candidate_files)
        )

    df = pd.read_csv(weight_file)

    weight_col = pick_col(df, ["reconstruction_weight", "weight"])

    label_col = None
    for c in [
        "track_label",
        "rbp_cell_line",
        "rbp_cellline",
        "RBP_cell_line",
        "track_name",
        "label",
        "rbp",
        "RBP",
    ]:
        if c in df.columns:
            label_col = c
            break

    if label_col is None:
        if "track_index" in df.columns:
            df["track_label"] = "track_" + df["track_index"].astype(str)
            label_col = "track_label"
        else:
            df["track_label"] = df.index.astype(str)
            label_col = "track_label"

    top = (
        df.sort_values(weight_col, ascending=False)
          .head(args.top_n)
          .sort_values(weight_col, ascending=True)
    )

    out_dir = run_dir / "report_assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.barh(top[label_col].astype(str), top[weight_col])
    ax.set_xlabel("Reconstruction weight", fontsize=8)
    ax.set_title(f"Top {args.top_n} track-level reconstruction weights", fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()

    pdf_path = out_dir / f"top{args.top_n}_track_weights.pdf"
    png_path = out_dir / f"top{args.top_n}_track_weights.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    print(f"Read: {weight_file}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")


if __name__ == "__main__":
    main()
