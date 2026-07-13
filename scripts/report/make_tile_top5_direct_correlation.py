#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def normalize(text):
    """Normalize labels so SND1_K562 and SND1 K562 can match."""
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def choose_column(columns, candidates, excluded=None):
    excluded = set(excluded or [])
    columns = [c for c in columns if c not in excluded]

    normalized_columns = {normalize(c): c for c in columns}

    # Prefer exact normalized matches
    for candidate in candidates:
        candidate_norm = normalize(candidate)
        if candidate_norm in normalized_columns:
            return normalized_columns[candidate_norm]

    # Then allow partial matches
    for candidate in candidates:
        candidate_norm = normalize(candidate)
        for column in columns:
            if candidate_norm and candidate_norm in normalize(column):
                return column

    return None


def load_metadata_labels(metadata_path):
    """Create track-index → readable RBP/cell-line label mapping."""
    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        return {}

    try:
        metadata = pd.read_csv(metadata_path, sep=None, engine="python")
    except Exception:
        try:
            metadata = pd.read_csv(metadata_path, sep="\t")
        except Exception:
            return {}

    index_column = choose_column(
        metadata.columns,
        [
            "track_index",
            "track_idx",
            "task_index",
            "task_idx",
            "index",
            "id",
        ],
    )

    label_column = choose_column(
        metadata.columns,
        [
            "track_name",
            "task_name",
            "label",
            "name",
            "experiment",
        ],
    )

    rbp_column = choose_column(
        metadata.columns,
        [
            "rbp",
            "rbp_name",
            "protein",
            "target",
        ],
    )

    cell_column = choose_column(
        metadata.columns,
        [
            "cell_line",
            "cellline",
            "cell_type",
            "celltype",
        ],
    )

    if label_column is not None:
        labels = metadata[label_column].astype(str)

    elif rbp_column is not None and cell_column is not None:
        labels = (
            metadata[rbp_column].astype(str)
            + "_"
            + metadata[cell_column].astype(str)
        )

    elif rbp_column is not None:
        labels = metadata[rbp_column].astype(str)

    else:
        return {}

    if index_column is not None:
        indices = pd.to_numeric(
            metadata[index_column],
            errors="coerce",
        )
    else:
        indices = pd.Series(metadata.index, index=metadata.index)

    mapping = {}

    for index, label in zip(indices, labels):
        if pd.notna(index):
            mapping[int(index)] = str(label)

    return mapping


def extract_long_format(data, tile, metadata_mapping):
    tile_column = choose_column(
        data.columns,
        [
            "tile",
            "tile_name",
            "window",
            "window_name",
            "sample_name",
            "sample",
            "interval",
            "region",
            "genomic_interval",
            "name",
        ],
    )

    if tile_column is None:
        first_column = data.columns[0]
        if (
            str(first_column).lower().startswith("unnamed")
            or data[first_column].dtype == object
        ):
            tile_column = first_column

    if tile_column is None:
        raise ValueError(
            "Could not identify the tile/window column.\n"
            f"Available columns: {list(data.columns)}"
        )

    tile_values = data[tile_column].astype(str)

    # Try exact match first
    mask = tile_values == tile

    # Then substring match
    if not mask.any():
        mask = tile_values.str.contains(
            re.escape(tile),
            regex=True,
            na=False,
        )

    # Finally normalized matching
    if not mask.any():
        target_norm = normalize(tile)
        mask = tile_values.map(normalize) == target_norm

    if not mask.any():
        chromosome = tile.split(":")[0]
        examples = (
            tile_values[tile_values.str.contains(chromosome, na=False)]
            .drop_duplicates()
            .head(20)
            .tolist()
        )

        raise ValueError(
            f"Tile not found: {tile}\n"
            f"Tile column: {tile_column}\n"
            f"Examples containing {chromosome}: {examples}"
        )

    subset = data.loc[mask].copy()

    remaining_columns = [
        column
        for column in data.columns
        if column != tile_column
    ]

    correlation_column = choose_column(
        remaining_columns,
        [
            "pearson",
            "pearson_r",
            "pearson_corr",
            "pearson_correlation",
            "pcc",
            "correlation",
            "corr",
        ],
    )

    track_column = choose_column(
        remaining_columns,
        [
            "track_name",
            "rbp_track",
            "track",
            "task_name",
            "task",
            "rbp_cell_line",
            "rbp_name",
            "rbp",
            "label",
            "track_index",
            "track_idx",
        ],
        excluded=[correlation_column] if correlation_column else [],
    )

    # Long format: one row per tile × track
    if correlation_column is not None and track_column is not None:
        output = subset[[track_column, correlation_column]].copy()

        output.columns = ["track_name", "pearson"]
        output["pearson"] = pd.to_numeric(
            output["pearson"],
            errors="coerce",
        )
        output = output.dropna(subset=["pearson"])

        # Map numeric track indices to names if needed
        numeric_track = pd.to_numeric(
            output["track_name"],
            errors="coerce",
        )

        if numeric_track.notna().all() and metadata_mapping:
            output["track_index"] = numeric_track.astype(int)

            output["track_name"] = output["track_index"].map(
                metadata_mapping
            ).fillna(
                output["track_index"].map(
                    lambda value: f"track_{value}"
                )
            )

        return output, tile_column

    # Wide format: one tile per row, tracks stored as columns
    excluded_terms = [
        "index",
        "rank",
        "count",
        "sum",
        "length",
        "split",
        "position",
        "globalclip",
        "reconstruction",
        "sample",
        "window",
        "tile",
        "coverage",
        "valid",
        "pvalue",
        "qvalue",
        "weight",
    ]

    rows = []

    for column in remaining_columns:
        column_norm = normalize(column)

        if any(term in column_norm for term in excluded_terms):
            continue

        values = pd.to_numeric(
            subset[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        # Pearson correlations must be between -1 and 1
        if not values.between(-1.000001, 1.000001).all():
            continue

        rows.append(
            {
                "track_name": str(column),
                "pearson": float(values.mean()),
            }
        )

    if not rows:
        raise ValueError(
            "The file did not look like either long-format or "
            "wide-format correlation data.\n"
            f"Available columns: {list(data.columns)}"
        )

    return pd.DataFrame(rows), tile_column


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="tile_by_track_correlations.csv",
    )

    parser.add_argument(
        "--tile",
        required=True,
        help="Genomic tile/window identifier",
    )

    parser.add_argument(
        "--metadata",
        default="",
        help="Optional full_rbp_set.tsv",
    )

    parser.add_argument(
        "--out-dir",
        required=True,
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--model-selected",
        default="",
        help="Comma-separated model-selected track names",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_directory = Path(args.out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_path)

    metadata_mapping = (
        load_metadata_labels(args.metadata)
        if args.metadata
        else {}
    )

    correlations, tile_column = extract_long_format(
        data=data,
        tile=args.tile,
        metadata_mapping=metadata_mapping,
    )

    correlations["track_name"] = (
        correlations["track_name"]
        .astype(str)
        .str.strip()
    )

    correlations = (
        correlations
        .dropna(subset=["pearson"])
        .sort_values("pearson", ascending=False)
        .drop_duplicates(subset=["track_name"])
        .reset_index(drop=True)
    )

    selected_tracks = [
        track.strip()
        for track in args.model_selected.split(",")
        if track.strip()
    ]

    selected_normalized = {
        normalize(track)
        for track in selected_tracks
    }

    correlations["model_selected"] = correlations[
        "track_name"
    ].map(
        lambda track: normalize(track) in selected_normalized
    )

    top = correlations.head(args.top_n).copy()
    top.insert(0, "rank", range(1, len(top) + 1))

    overlap = int(top["model_selected"].sum())

    output_csv = (
        output_directory
        / "top5_direct_correlation_at_example_tile.csv"
    )

    top.to_csv(output_csv, index=False)

    # Plot in reverse order so rank 1 appears at the top
    plot_data = top.iloc[::-1].copy()

    plot_labels = [
        ("★ " if selected else "") + track
        for track, selected in zip(
            plot_data["track_name"],
            plot_data["model_selected"],
        )
    ]

    figure_height = max(4.2, 0.62 * len(plot_data) + 2.2)

    fig, ax = plt.subplots(
        figsize=(8.4, figure_height)
    )

    bars = ax.barh(
        plot_labels,
        plot_data["pearson"],
    )

    ax.set_xlabel(
        "Pearson correlation with observed globalCLIP"
    )

    ax.set_title(
        "Top direct RBP-profile matches at the example tile",
        pad=14,
    )

    ax.axvline(
        0,
        linewidth=0.8,
    )

    for bar, value in zip(
        bars,
        plot_data["pearson"],
    ):
        offset = 0.008 if value >= 0 else -0.008
        horizontal_alignment = (
            "left" if value >= 0 else "right"
        )

        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha=horizontal_alignment,
            fontsize=10,
        )

    all_values = plot_data["pearson"].tolist()

    if all_values:
        minimum = min(0, min(all_values))
        maximum = max(0, max(all_values))
        padding = max(0.04, (maximum - minimum) * 0.12)

        ax.set_xlim(
            minimum - padding,
            maximum + padding,
        )

    fig.text(
        0.01,
        0.01,
        (
            f"★ Model-selected track. "
            f"Overlap with model-selected top tracks: "
            f"{overlap}/{len(top)}."
        ),
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    output_png = (
        output_directory
        / "top5_direct_correlation_at_example_tile.png"
    )

    output_pdf = (
        output_directory
        / "top5_direct_correlation_at_example_tile.pdf"
    )

    fig.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        output_pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print("Tile column:", tile_column)
    print("Selected tile:", args.tile)
    print()
    print(top.to_string(index=False))
    print()
    print(
        "Overlap with model-selected tracks:",
        f"{overlap}/{len(top)}",
    )
    print()
    print("Created:")
    print(output_csv)
    print(output_png)
    print(output_pdf)


if __name__ == "__main__":
    main()
