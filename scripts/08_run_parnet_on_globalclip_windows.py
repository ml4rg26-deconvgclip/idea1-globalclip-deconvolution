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
OPTIONAL_OUTPUT_KEYS = {
    "target": "parnet_target",
    "control": "parnet_control",
    "mix_coeff": "mix_coeff",
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


def validate_total_output(total: Any, *, batch_size: int, profile_count: int, target_len: int) -> None:
    observed_shape = tuple(total.shape)
    expected_shape = (batch_size, profile_count, target_len)
    if observed_shape != expected_shape:
        raise ValueError(f"Expected out['total'] shape {expected_shape}, observed {observed_shape}")


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
    parnet_total = torch.empty((sample_count, profile_count, args.target_len), dtype=torch.float32)
    optional_outputs: dict[str, Any] = {}
    expected_optional_keys: set[str] | None = None

    print(f"Running inference in batches of {args.batch_size}")
    with torch.no_grad():
        for start in range(0, sample_count, args.batch_size):
            end = min(start + args.batch_size, sample_count)
            batch_sequences = sequences[start:end]
            batch = sequences_to_onehot(torch, batch_sequences).to(device)
            out = model(batch)

            if not isinstance(out, dict):
                raise TypeError(f"Expected model output dict, got {type(out).__name__}")
            if "total" not in out:
                raise KeyError(f"Expected output key 'total'; observed keys: {list(out.keys())}")

            validate_total_output(
                out["total"],
                batch_size=end - start,
                profile_count=profile_count,
                target_len=args.target_len,
            )
            parnet_total[start:end] = out["total"].detach().cpu().to(dtype=torch.float32)

            observed_optional_keys = {
                source_key for source_key in OPTIONAL_OUTPUT_KEYS if source_key in out
            }
            if expected_optional_keys is None:
                expected_optional_keys = observed_optional_keys
            elif observed_optional_keys != expected_optional_keys:
                raise ValueError(
                    "Inconsistent optional PARNET output keys across batches: "
                    f"expected {sorted(expected_optional_keys)}, observed {sorted(observed_optional_keys)}"
                )

            for source_key, saved_key in OPTIONAL_OUTPUT_KEYS.items():
                if source_key not in out:
                    continue
                value = out[source_key].detach().cpu().to(dtype=torch.float32)
                if value.shape[0] != end - start:
                    raise ValueError(
                        f"Expected out[{source_key!r}] batch dimension {end - start}, "
                        f"observed {tuple(value.shape)}"
                    )
                if saved_key not in optional_outputs:
                    optional_outputs[saved_key] = torch.empty(
                        (sample_count, *value.shape[1:]),
                        dtype=torch.float32,
                    )
                optional_outputs[saved_key][start:end] = value

            batch_number = (start // args.batch_size) + 1
            batch_count = (sample_count + args.batch_size - 1) // args.batch_size
            if batch_number == 1 or batch_number == batch_count or batch_number % 10 == 0:
                print(f"  batch {batch_number}/{batch_count}: windows {start}-{end - 1}")

    total_sums = parnet_total.sum(dim=-1)
    summary = {
        "parnet_total_shape": tuple(parnet_total.shape),
        "parnet_total_dtype": str(parnet_total.dtype),
        "parnet_total_min": float(parnet_total.min().item()),
        "parnet_total_max": float(parnet_total.max().item()),
        "parnet_total_sum_min": float(total_sums.min().item()),
        "parnet_total_sum_mean": float(total_sums.mean().item()),
        "parnet_total_sum_max": float(total_sums.max().item()),
        "unknown_base_count": int(unknown_total),
        "unknown_base_counts": dict(sorted(unknown_counts.items())),
    }
    for saved_key, value in optional_outputs.items():
        summary[f"{saved_key}_shape"] = tuple(value.shape)
        summary[f"{saved_key}_dtype"] = str(value.dtype)

    output = {
        "parnet_total": parnet_total,
        **optional_outputs,
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

    print("\nObserved output shapes:")
    print(f"  total: {output_shape(parnet_total)}")
    for saved_key, value in optional_outputs.items():
        print(f"  {saved_key}: {output_shape(value)}")
    print("PARNET total summary:")
    print(f"  dtype: {parnet_total.dtype}")
    print(f"  min:   {parnet_total.min().item():.6g}")
    print(f"  max:   {parnet_total.max().item():.6g}")
    print("PARNET total sums over length axis:")
    print(f"  min:  {total_sums.min().item():.8f}")
    print(f"  mean: {total_sums.mean().item():.8f}")
    print(f"  max:  {total_sums.max().item():.8f}")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
