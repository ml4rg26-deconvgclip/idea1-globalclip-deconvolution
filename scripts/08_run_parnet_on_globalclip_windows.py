#!/usr/bin/env python3
"""Run pretrained PARNET on globalCLIP sequence windows and save predictions."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from smoke_test_parnet import (
    choose_device,
    get_section,
    load_config,
    output_shape,
    project_root_from_config,
    resolve_config_path,
    resolve_model_name,
    sequences_to_onehot,
)


DEFAULT_SPLITS = ("train", "valid", "test")
BASE_TO_CHANNEL = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}
TRACKS = ("total", "target", "control")
OUTPUT_SPACES = ("prob", "logprob", "both")
TRACK_OUTPUT_ALIASES = {
    "total": ("total",),
    "target": ("target", "parnet_target"),
    "control": ("control", "parnet_control"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run pretrained PARNET on real globalCLIP sequence windows and save "
            "the predicted RBP-cell-line profiles."
        )
    )
    parser.add_argument("--input-pt", type=Path, required=True, help="Input globalCLIP .pt file.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="PARNET checkpoint path. Defaults to paths.parnet_checkpoint from --config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vm.yaml"),
        help="Path to the VM YAML/JSON config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/parnet_predictions"),
        help="Directory for the saved PARNET prediction .pt file.",
    )
    parser.add_argument(
        "--condition-name",
        default="interphase",
        help="Condition label to save in metadata and default output filename.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Device for inference.",
    )
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
        "--output-name",
        default=None,
        help="Optional output filename. Defaults to <condition-name>_parnet_predictions.pt.",
    )
    parser.add_argument(
        "--output-track",
        choices=(*TRACKS, "all"),
        default="total",
        help="PARNET output track to save. Default saves only total to limit output size.",
    )
    parser.add_argument(
        "--output-space",
        choices=OUTPUT_SPACES,
        default="prob",
        help="Save probability profiles, raw log probabilities, or both.",
    )
    parser.add_argument(
        "--save-mix-coeff",
        action="store_true",
        help="Save out['mix_coeff'] when available. Disabled by default to save space.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16"),
        default="float16",
        help="Dtype for saved prediction tensors.",
    )
    parser.add_argument("--target-len", type=int, default=600, help="PARNET input length.")
    parser.add_argument(
        "--model-name",
        default=None,
        help="Parnet model name. Defaults to model.parnet_model_name or parnet.7m-0.0.",
    )
    return parser.parse_args()


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for PARNET inference.") from exc
    return torch


def load_infer_pad_sizes(project_root: Path) -> Any:
    try:
        from parnet_demo_utils import infer_pad_sizes
    except ImportError:
        candidates = [
            project_root.parent / "parnet--demo--train-models" / "src",
            project_root.parent.parent / "parnet--demo--train-models" / "src",
        ]
        for candidate in candidates:
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
                try:
                    from parnet_demo_utils import infer_pad_sizes
                except ImportError:
                    continue
                return infer_pad_sizes
        raise SystemExit(
            "parnet_demo_utils is required for infer_pad_sizes. Install the sibling "
            "parnet--demo--train-models package or run with its src directory on PYTHONPATH."
        )
    return infer_pad_sizes


def load_parnet_api() -> tuple[Any, Any]:
    try:
        from parnet_additional_utils import ParnetModelName, load_parnet_model
    except ImportError as exc:
        raise SystemExit(
            "parnet_additional_utils is required. Run inside the parnet demo pixi environment."
        ) from exc
    return ParnetModelName, load_parnet_model


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


def validate_requested_device(torch: Any, requested: str, resolved: str) -> None:
    if resolved == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable. Use --device auto or cpu.")
    if resolved == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise SystemExit("--device mps requested but MPS is unavailable. Use --device auto or cpu.")


def get_sample_fields(sample: dict[str, Any]) -> tuple[str, str, int]:
    try:
        sequence = sample["inputs"]["sequence"]
        name = sample["meta"]["name"]
        pad_side = sample["meta"]["pad_side"]
    except KeyError as exc:
        raise KeyError(f"Sample is missing required field: {exc}") from exc

    if not isinstance(sequence, str):
        raise TypeError(f"sample['inputs']['sequence'] must be str, got {type(sequence).__name__}")
    return sequence, str(name), int(pad_side)


def reconstruct_padded_sequence(
    *,
    sequence: str,
    name: str,
    pad_side: int,
    target_len: int,
    infer_pad_sizes: Any,
) -> tuple[str, int, int]:
    if pad_side == -1:
        left_pad, right_pad = infer_pad_sizes(name, pad_side, target_len)
        if len(sequence) != target_len or left_pad != 0 or right_pad != 0:
            raise ValueError(
                f"{name}: pad_side=-1 means no edge padding, but original length={len(sequence)} "
                f"and infer_pad_sizes returned left={left_pad}, right={right_pad}."
            )
    elif pad_side in (0, 1, 2):
        left_pad, right_pad = infer_pad_sizes(name, pad_side, target_len)
    else:
        raise ValueError(f"{name}: unsupported pad_side={pad_side}; expected -1, 0, 1, or 2.")

    padded = "N" * left_pad + sequence + "N" * right_pad
    if len(padded) != target_len:
        raise ValueError(
            f"{name}: padded sequence length is {len(padded)}, expected {target_len}. "
            f"original_length={len(sequence)}, pad_side={pad_side}, "
            f"left_pad={left_pad}, right_pad={right_pad}."
        )
    return padded, left_pad, right_pad


def count_unknown_bases(sequences: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for sequence in sequences:
        for base in sequence.upper():
            if base not in BASE_TO_CHANNEL:
                counts[base] += 1
    return counts


def validate_track_output(
    value: Any,
    *,
    track: str,
    batch_size: int,
    profile_count: int,
    target_len: int,
) -> None:
    observed_shape = tuple(value.shape)
    expected_shape = (batch_size, profile_count, target_len)
    if observed_shape != expected_shape:
        raise ValueError(f"Expected out[{track!r}] shape {expected_shape}, observed {observed_shape}")


def validate_mix_coeff_output(value: Any, *, batch_size: int, profile_count: int) -> None:
    observed_shape = tuple(value.shape)
    expected_shape = (batch_size, profile_count)
    if observed_shape != expected_shape:
        raise ValueError(f"Expected out['mix_coeff'] shape {expected_shape}, observed {observed_shape}")


def selected_tracks(output_track: str) -> list[str]:
    return list(TRACKS) if output_track == "all" else [output_track]


def selected_spaces(output_space: str) -> list[str]:
    if output_space == "both":
        return ["prob", "logprob"]
    return [output_space]


def saved_dtype(torch: Any, dtype_name: str) -> Any:
    return {"float32": torch.float32, "float16": torch.float16}[dtype_name]


def saved_prediction_key(track: str, space: str) -> str:
    suffix = "profile" if space == "prob" else "logprob"
    return f"parnet_{track}_{suffix}"


def resolve_track_output(out: dict[str, Any], track: str) -> tuple[str, Any]:
    for key in TRACK_OUTPUT_ALIASES[track]:
        if key in out:
            return key, out[key]
    raise KeyError(
        f"Expected one of {TRACK_OUTPUT_ALIASES[track]} for output track {track!r}; "
        f"observed keys: {list(out.keys())}"
    )


def profile_sum_stats(tensor: Any, *, chunk_size: int = 128) -> dict[str, float]:
    torch = require_torch()
    min_value = float("inf")
    max_value = float("-inf")
    total = 0.0
    count = 0

    for start in range(0, tensor.shape[0], chunk_size):
        sums = tensor[start : start + chunk_size].to(dtype=torch.float32).sum(dim=-1)
        min_value = min(min_value, float(sums.min().item()))
        max_value = max(max_value, float(sums.max().item()))
        total += float(sums.sum().item())
        count += sums.numel()

    return {
        "min": min_value,
        "mean": total / count,
        "max": max_value,
    }


def summarize_saved_outputs(saved_outputs: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in saved_outputs.items():
        summary[f"{key}_shape"] = tuple(value.shape)
        summary[f"{key}_saved_dtype"] = str(value.dtype)

        if key.endswith("_logprob"):
            summary[f"{key}_logprob_min"] = float(value.min().item())
            summary[f"{key}_logprob_max"] = float(value.max().item())
        elif key.endswith("_profile"):
            summary[f"{key}_profile_min"] = float(value.min().item())
            summary[f"{key}_profile_max"] = float(value.max().item())
            sums = profile_sum_stats(value)
            summary[f"{key}_profile_sum_min"] = sums["min"]
            summary[f"{key}_profile_sum_mean"] = sums["mean"]
            summary[f"{key}_profile_sum_max"] = sums["max"]
        elif key == "mix_coeff":
            summary[f"{key}_min"] = float(value.min().item())
            summary[f"{key}_max"] = float(value.max().item())

    return summary


def prepare_sequences(
    rows: list[tuple[str, int, dict[str, Any]]],
    *,
    target_len: int,
    infer_pad_sizes: Any,
) -> tuple[list[str], list[str], list[int], list[str], list[int], list[int], list[int]]:
    sequences = []
    split_names = []
    indices = []
    names = []
    pad_sides = []
    left_pads = []
    right_pads = []

    for split, index, sample in rows:
        sequence, name, pad_side = get_sample_fields(sample)
        padded, left_pad, right_pad = reconstruct_padded_sequence(
            sequence=sequence,
            name=name,
            pad_side=pad_side,
            target_len=target_len,
            infer_pad_sizes=infer_pad_sizes,
        )
        sequences.append(padded)
        split_names.append(split)
        indices.append(index)
        names.append(name)
        pad_sides.append(pad_side)
        left_pads.append(left_pad)
        right_pads.append(right_pad)

    return sequences, split_names, indices, names, pad_sides, left_pads, right_pads


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.target_len <= 0:
        raise SystemExit("--target-len must be positive")
    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit must be non-negative")

    config_path = args.config.resolve()
    config = load_config(config_path)
    project_root = project_root_from_config(config_path)
    paths = get_section(config, "paths")
    model_cfg = get_section(config, "model")

    profile_count = int(model_cfg.get("rbp_cell_line_profiles", 223))
    model_name = args.model_name or str(model_cfg.get("parnet_model_name", "parnet.7m-0.0"))

    raw_checkpoint = args.checkpoint or paths.get("parnet_checkpoint")
    if raw_checkpoint is None:
        raise SystemExit("Missing checkpoint path: provide --checkpoint or paths.parnet_checkpoint")
    checkpoint_path = resolve_config_path(raw_checkpoint, project_root=project_root)
    if not checkpoint_path.exists():
        raise SystemExit(f"PARNET checkpoint does not exist: {checkpoint_path}")

    torch = require_torch()
    infer_pad_sizes = load_infer_pad_sizes(project_root)
    ParnetModelName, load_parnet_model = load_parnet_api()
    device = choose_device(torch, args.device)
    validate_requested_device(torch, args.device, device)
    model_enum_value = resolve_model_name(model_name, ParnetModelName)

    print(f"Config: {config_path}")
    print(f"Input .pt: {args.input_pt}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model name: {model_name}")
    print(f"Device: {device}")
    print(f"Target length: {args.target_len}")
    print(f"Expected total output profiles: {profile_count}")

    data = torch.load(args.input_pt, map_location="cpu", weights_only=False)
    rows = iter_samples(data, args.split)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No samples selected.")

    print(f"Selected windows: {len(rows)}")
    sequences, split_names, indices, names, pad_sides, left_pads, right_pads = prepare_sequences(
        rows,
        target_len=args.target_len,
        infer_pad_sizes=infer_pad_sizes,
    )

    unknown_counts = count_unknown_bases(sequences)
    unknown_total = sum(unknown_counts.values())
    print(f"Unknown/all-zero encoded bases: {unknown_total}")
    if unknown_counts:
        print(f"Unknown base counts: {dict(sorted(unknown_counts.items()))}")

    model = load_parnet_model(
        model_enum_value,
        checkpoint_path,
        dtype=torch.float32,
        device=device,
    )
    model.eval()

    sample_count = len(sequences)
    tracks_to_save = selected_tracks(args.output_track)
    spaces_to_save = selected_spaces(args.output_space)
    prediction_dtype = saved_dtype(torch, args.dtype)
    saved_outputs = {
        saved_prediction_key(track, space): torch.empty(
            (sample_count, profile_count, args.target_len),
            dtype=prediction_dtype,
        )
        for track in tracks_to_save
        for space in spaces_to_save
    }
    if args.save_mix_coeff:
        saved_outputs["mix_coeff"] = torch.empty(
            (sample_count, profile_count),
            dtype=prediction_dtype,
        )

    print(f"Running inference in batches of {args.batch_size}")
    print(f"Saving tracks: {', '.join(tracks_to_save)}")
    print(f"Saving spaces: {', '.join(spaces_to_save)}")
    print(f"Saved prediction dtype: {prediction_dtype}")
    with torch.no_grad():
        for start in range(0, sample_count, args.batch_size):
            end = min(start + args.batch_size, sample_count)
            batch_sequences = sequences[start:end]
            batch = sequences_to_onehot(torch, batch_sequences).to(device)
            out = model(batch)

            if not isinstance(out, dict):
                raise TypeError(f"Expected model output dict, got {type(out).__name__}")

            for track in tracks_to_save:
                source_key, logprob = resolve_track_output(out, track)
                validate_track_output(
                    logprob,
                    track=source_key,
                    batch_size=end - start,
                    profile_count=profile_count,
                    target_len=args.target_len,
                )

                if "logprob" in spaces_to_save:
                    saved_outputs[saved_prediction_key(track, "logprob")][start:end] = (
                        logprob.detach().cpu().to(dtype=prediction_dtype)
                    )

                if "prob" in spaces_to_save:
                    saved_outputs[saved_prediction_key(track, "prob")][start:end] = (
                        logprob.exp().detach().cpu().to(dtype=prediction_dtype)
                    )

            if args.save_mix_coeff:
                if "mix_coeff" not in out:
                    raise KeyError(f"Expected output key 'mix_coeff'; observed keys: {list(out.keys())}")
                validate_mix_coeff_output(
                    out["mix_coeff"],
                    batch_size=end - start,
                    profile_count=profile_count,
                )
                saved_outputs["mix_coeff"][start:end] = (
                    out["mix_coeff"].detach().cpu().to(dtype=prediction_dtype)
                )

            batch_number = (start // args.batch_size) + 1
            batch_count = (sample_count + args.batch_size - 1) // args.batch_size
            if batch_number == 1 or batch_number == batch_count or batch_number % 10 == 0:
                print(f"  batch {batch_number}/{batch_count}: windows {start}-{end - 1}")

    summary = {
        "output_track": args.output_track,
        "output_space": args.output_space,
        "saved_tracks": tracks_to_save,
        "saved_spaces": spaces_to_save,
        "saved_dtype": args.dtype,
        "unknown_base_count": int(unknown_total),
        "unknown_base_counts": dict(sorted(unknown_counts.items())),
    }
    summary.update(summarize_saved_outputs(saved_outputs))

    output = {
        **saved_outputs,
        "split": split_names,
        "index": torch.tensor(indices, dtype=torch.long),
        "name": names,
        "pad_side": torch.tensor(pad_sides, dtype=torch.long),
        "left_pad": torch.tensor(left_pads, dtype=torch.long),
        "right_pad": torch.tensor(right_pads, dtype=torch.long),
        "source": str(args.input_pt),
        "checkpoint": str(checkpoint_path),
        "model_name": model_name,
        "condition_name": args.condition_name,
        "summary": summary,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"{args.condition_name}_parnet_predictions.pt"
    output_path = args.output_dir / output_name
    if output_path.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output_path}")

    torch.save(output, output_path)

    print("\nSaved output shapes:")
    for saved_key, value in saved_outputs.items():
        print(f"  {saved_key}: {output_shape(value)}")
        print(f"    dtype: {value.dtype}")
        if saved_key.endswith("_profile"):
            print(f"    min:   {summary[f'{saved_key}_profile_min']:.6g}")
            print(f"    max:   {summary[f'{saved_key}_profile_max']:.6g}")
            print("    sums over length axis:")
            print(f"      min:  {summary[f'{saved_key}_profile_sum_min']:.8f}")
            print(f"      mean: {summary[f'{saved_key}_profile_sum_mean']:.8f}")
            print(f"      max:  {summary[f'{saved_key}_profile_sum_max']:.8f}")
        elif saved_key.endswith("_logprob"):
            print(f"    min:   {summary[f'{saved_key}_logprob_min']:.6g}")
            print(f"    max:   {summary[f'{saved_key}_logprob_max']:.6g}")
        elif saved_key == "mix_coeff":
            print(f"    min:   {summary[f'{saved_key}_min']:.6g}")
            print(f"    max:   {summary[f'{saved_key}_max']:.6g}")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
