#!/usr/bin/env python3
"""Train a softmax reconstruction head on precomputed ParnET profiles."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any


EXPECTED_NUM_TRACKS = 223
SPLITS = ("train", "valid", "test")
BASELINE_MODELS = (
    "uniform_valid_profile",
    "parnet_unweighted_average",
    "train_mean_observed_profile",
)
PRED_METADATA_FIELDS = ("name", "split", "index")
PRED_REQUIRED_KEYS = ("parnet_total_profile", *PRED_METADATA_FIELDS)
TARGET_REQUIRED_KEYS = ("profiles", "valid_mask", "total_counts", *PRED_METADATA_FIELDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the Step 4 Idea 1 reconstruction head: a global softmax-weighted "
            "mixture of 223 precomputed ParnET RBP-cell-line profiles."
        )
    )
    parser.add_argument(
        "--parnet-predictions",
        type=Path,
        required=True,
        help="Path to the .pt file containing pred['parnet_total_profile'].",
    )
    parser.add_argument(
        "--target-profiles",
        type=Path,
        required=True,
        help="Path to the .pt file containing observed globalCLIP target profiles.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where checkpoints, metrics, weights, and metadata are written.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Adam learning rate.")
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Training device. 'auto' prefers CUDA, then MPS, then CPU.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument("--eps", type=float, default=1e-8, help="Numerical stability epsilon.")
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stopping patience in epochs, monitored on validation loss.",
    )
    parser.add_argument(
        "--weight-by-total-count",
        action="store_true",
        help="Weight profile NLL by total_counts normalized to mean 1 within each batch.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise SystemExit(f"--batch-size must be positive, got {args.batch_size}")
    if args.epochs <= 0:
        raise SystemExit(f"--epochs must be positive, got {args.epochs}")
    if args.lr <= 0:
        raise SystemExit(f"--lr must be positive, got {args.lr}")
    if args.eps <= 0:
        raise SystemExit(f"--eps must be positive, got {args.eps}")
    if args.patience <= 0:
        raise SystemExit(f"--patience must be positive, got {args.patience}")


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required to train the reconstruction head.") from exc
    return torch


def torch_load_cpu(torch: Any, path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"{label} file does not exist: {path}")

    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise SystemExit(f"Failed to load {label} file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise SystemExit(f"Expected {label} file to contain a dict, got {type(data).__name__}")
    return data


def require_keys(data: dict[str, Any], required_keys: tuple[str, ...], label: str) -> None:
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise ValueError(f"{label} file is missing required key(s): {', '.join(missing)}")


def require_tensor(torch: Any, data: dict[str, Any], key: str, label: str) -> Any:
    value = data[key]
    if not torch.is_tensor(value):
        raise TypeError(f"{label}[{key!r}] must be a torch.Tensor, got {type(value).__name__}")
    return value


def validate_tensors(
    torch: Any,
    pred: dict[str, Any],
    target: dict[str, Any],
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    require_keys(pred, PRED_REQUIRED_KEYS, "ParnET predictions")
    require_keys(target, TARGET_REQUIRED_KEYS, "target profiles")

    x = require_tensor(torch, pred, "parnet_total_profile", "ParnET predictions")
    y = require_tensor(torch, target, "profiles", "target profiles")
    valid_mask = require_tensor(torch, target, "valid_mask", "target profiles")
    total_counts = require_tensor(torch, target, "total_counts", "target profiles")

    if x.ndim != 3:
        raise ValueError(f"pred['parnet_total_profile'] must have shape [N, 223, 600], got {list(x.shape)}")
    if y.ndim != 3:
        raise ValueError(f"target['profiles'] must have shape [N, 1, 600], got {list(y.shape)}")
    if valid_mask.ndim != 3:
        raise ValueError(f"target['valid_mask'] must have shape [N, 1, 600], got {list(valid_mask.shape)}")
    if total_counts.ndim not in (1, 2):
        raise ValueError(f"target['total_counts'] must have shape [N] or [N, 1], got {list(total_counts.shape)}")

    num_windows, num_tracks, target_len = x.shape
    if num_tracks != EXPECTED_NUM_TRACKS:
        raise ValueError(
            f"Expected {EXPECTED_NUM_TRACKS} ParnET tracks, got {num_tracks} in pred['parnet_total_profile']"
        )
    if y.shape != (num_windows, 1, target_len):
        raise ValueError(
            "target['profiles'] shape must match pred windows and length as [N, 1, L]; "
            f"got {list(y.shape)} for pred shape {list(x.shape)}"
        )
    if valid_mask.shape != y.shape:
        raise ValueError(f"target['valid_mask'] shape {list(valid_mask.shape)} must match profiles {list(y.shape)}")
    if total_counts.shape[0] != num_windows:
        raise ValueError(
            f"target['total_counts'] first dimension must be N={num_windows}, got {list(total_counts.shape)}"
        )
    if total_counts.ndim == 2 and total_counts.shape[1] != 1:
        raise ValueError(f"target['total_counts'] must have shape [N, 1], got {list(total_counts.shape)}")

    shape_info = {
        "parnet_total_profile": {"shape": list(x.shape), "dtype": str(x.dtype)},
        "profiles": {"shape": list(y.shape), "dtype": str(y.dtype)},
        "valid_mask": {"shape": list(valid_mask.shape), "dtype": str(valid_mask.dtype)},
        "total_counts": {"shape": list(total_counts.shape), "dtype": str(total_counts.dtype)},
        "num_windows": int(num_windows),
        "num_tracks": int(num_tracks),
        "target_length": int(target_len),
    }
    return x, y, valid_mask, total_counts, shape_info


def to_python_scalar(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, list) and len(value) == 1:
        return to_python_scalar(value[0])
    return value


def metadata_to_list(torch: Any, value: Any) -> list[Any]:
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.ndim == 0:
            return [to_python_scalar(value.item())]
        return [to_python_scalar(item) for item in value.tolist()]

    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            value = value.tolist()
        except (TypeError, ValueError):
            pass

    if isinstance(value, (list, tuple)):
        return [to_python_scalar(item) for item in value]
    return [to_python_scalar(value)]


def validate_alignment(
    torch: Any,
    pred: dict[str, Any],
    target: dict[str, Any],
    num_windows: int,
) -> dict[str, Any]:
    for field in PRED_METADATA_FIELDS:
        pred_values = metadata_to_list(torch, pred[field])
        target_values = metadata_to_list(torch, target[field])

        if len(pred_values) != num_windows:
            raise ValueError(
                f"pred[{field!r}] length must be {num_windows}, got {len(pred_values)}"
            )
        if len(target_values) != num_windows:
            raise ValueError(
                f"target[{field!r}] length must be {num_windows}, got {len(target_values)}"
            )

        if pred_values != target_values:
            mismatch = next(
                index
                for index, (pred_value, target_value) in enumerate(zip(pred_values, target_values))
                if pred_value != target_value
            )
            raise ValueError(
                f"Input files are not aligned for metadata field {field!r} at row {mismatch}: "
                f"pred={pred_values[mismatch]!r}, target={target_values[mismatch]!r}"
            )

    return {"status": "aligned", "checked_fields": list(PRED_METADATA_FIELDS)}


def canonical_split_label(value: Any) -> str:
    value = to_python_scalar(value)
    label = value if isinstance(value, str) else str(value)
    normalized = label.strip().lower()
    aliases = {
        "training": "train",
        "validation": "valid",
        "val": "valid",
    }
    return aliases.get(normalized, normalized)


def make_split_indices(torch: Any, split_values: list[Any]) -> tuple[dict[str, Any], list[str]]:
    indices_by_split: dict[str, list[int]] = {split: [] for split in SPLITS}
    available_labels: set[str] = set()

    for index, split_value in enumerate(split_values):
        label = canonical_split_label(split_value)
        available_labels.add(label)
        if label in indices_by_split:
            indices_by_split[label].append(index)

    missing = [split for split, values in indices_by_split.items() if not values]
    if missing:
        raise ValueError(
            "Missing required split label(s): "
            f"{', '.join(missing)}. Available canonical labels: {sorted(available_labels)}"
        )

    tensor_indices = {
        split: torch.tensor(values, dtype=torch.long)
        for split, values in indices_by_split.items()
    }
    return tensor_indices, sorted(available_labels)


def resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable. Use --device auto or --device cpu.")
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise SystemExit("--device mps requested but MPS is unavailable. Use --device auto or --device cpu.")
    return requested


def set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def iter_index_batches(
    torch: Any,
    indices: Any,
    batch_size: int,
    *,
    shuffle: bool,
    generator: Any | None = None,
) -> Any:
    if indices.numel() == 0:
        return

    if shuffle:
        order = indices[torch.randperm(indices.numel(), generator=generator)]
    else:
        order = indices

    for start in range(0, int(order.numel()), batch_size):
        yield order[start : start + batch_size]


def gather_target_batch(
    Y: Any,
    valid_mask: Any,
    total_counts: Any,
    batch_indices: Any,
    device: str,
) -> tuple[Any, Any, Any]:
    target_batch = Y.index_select(0, batch_indices).squeeze(1).to(device=device).float()
    valid = valid_mask.index_select(0, batch_indices).squeeze(1).to(device=device).float()
    counts = total_counts.index_select(0, batch_indices).reshape(-1).to(device=device).float()
    return target_batch, valid, counts


def gather_x_batch(X: Any, batch_indices: Any, device: str) -> Any:
    return X.index_select(0, batch_indices).to(device=device).float()


def mask_and_renormalize(profile: Any, valid: Any, eps: float) -> Any:
    profile = profile * valid
    return profile / (profile.sum(dim=-1, keepdim=True) + eps)


def reconstruction_forward(torch: Any, logits: Any, x_batch: Any, valid: Any, eps: float) -> Any:
    weights = torch.softmax(logits, dim=0)
    pred_global = (x_batch * weights[None, :, None]).sum(dim=1)
    return mask_and_renormalize(pred_global, valid, eps)


def profile_nll(
    torch: Any,
    pred_global: Any,
    target_batch: Any,
    total_counts: Any,
    *,
    weight_by_total_count: bool,
    eps: float,
) -> Any:
    loss_per_window = -(target_batch * torch.log(pred_global + eps)).sum(dim=-1)
    if weight_by_total_count:
        count_weights = total_counts.float()
        count_weights = count_weights / (count_weights.mean() + eps)
        return (loss_per_window * count_weights).mean()
    return loss_per_window.mean()


def train_one_epoch(
    torch: Any,
    logits: Any,
    optimizer: Any,
    X: Any,
    Y: Any,
    valid_mask: Any,
    total_counts: Any,
    train_indices: Any,
    *,
    batch_size: int,
    device: str,
    eps: float,
    weight_by_total_count: bool,
    generator: Any,
) -> float:
    total_loss = 0.0
    total_windows = 0

    for batch_indices in iter_index_batches(
        torch,
        train_indices,
        batch_size,
        shuffle=True,
        generator=generator,
    ):
        optimizer.zero_grad(set_to_none=True)
        x_batch = gather_x_batch(X, batch_indices, device)
        target_batch, valid, counts = gather_target_batch(Y, valid_mask, total_counts, batch_indices, device)
        pred_global = reconstruction_forward(torch, logits, x_batch, valid, eps)
        loss = profile_nll(
            torch,
            pred_global,
            target_batch,
            counts,
            weight_by_total_count=weight_by_total_count,
            eps=eps,
        )
        loss_value = float(loss.detach().cpu())
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Non-finite training loss encountered: {loss_value}")

        loss.backward()
        optimizer.step()

        batch_size_actual = int(batch_indices.numel())
        total_loss += loss_value * batch_size_actual
        total_windows += batch_size_actual

    return total_loss / total_windows


def evaluate_reconstruction_head(
    torch: Any,
    logits: Any,
    X: Any,
    Y: Any,
    valid_mask: Any,
    total_counts: Any,
    indices: Any,
    *,
    batch_size: int,
    device: str,
    eps: float,
    weight_by_total_count: bool,
) -> float:
    total_loss = 0.0
    total_windows = 0

    with torch.no_grad():
        for batch_indices in iter_index_batches(torch, indices, batch_size, shuffle=False):
            x_batch = gather_x_batch(X, batch_indices, device)
            target_batch, valid, counts = gather_target_batch(Y, valid_mask, total_counts, batch_indices, device)
            pred_global = reconstruction_forward(torch, logits, x_batch, valid, eps)
            loss = profile_nll(
                torch,
                pred_global,
                target_batch,
                counts,
                weight_by_total_count=weight_by_total_count,
                eps=eps,
            )
            batch_size_actual = int(batch_indices.numel())
            total_loss += float(loss.detach().cpu()) * batch_size_actual
            total_windows += batch_size_actual

    return total_loss / total_windows


def compute_train_mean_observed_profile(
    torch: Any,
    Y: Any,
    train_indices: Any,
    *,
    batch_size: int,
    eps: float,
) -> Any:
    profile_sum = torch.zeros(Y.shape[-1], dtype=torch.float64)
    total_windows = 0

    for batch_indices in iter_index_batches(torch, train_indices, batch_size, shuffle=False):
        target_batch = Y.index_select(0, batch_indices).squeeze(1).double()
        profile_sum += target_batch.sum(dim=0)
        total_windows += int(batch_indices.numel())

    if total_windows == 0:
        raise ValueError("Cannot compute train-set mean observed profile: train split is empty")

    mean_profile = (profile_sum / float(total_windows)).float()
    return mean_profile / (mean_profile.sum() + eps)


def evaluate_baseline(
    torch: Any,
    model_name: str,
    X: Any,
    Y: Any,
    valid_mask: Any,
    total_counts: Any,
    indices: Any,
    *,
    train_mean_profile: Any,
    batch_size: int,
    device: str,
    eps: float,
    weight_by_total_count: bool,
) -> float:
    total_loss = 0.0
    total_windows = 0
    mean_profile_device = None
    if model_name == "train_mean_observed_profile":
        mean_profile_device = train_mean_profile.to(device=device).float()

    with torch.no_grad():
        for batch_indices in iter_index_batches(torch, indices, batch_size, shuffle=False):
            target_batch, valid, counts = gather_target_batch(Y, valid_mask, total_counts, batch_indices, device)

            if model_name == "uniform_valid_profile":
                pred_global = valid / (valid.sum(dim=-1, keepdim=True) + eps)
            elif model_name == "parnet_unweighted_average":
                x_batch = gather_x_batch(X, batch_indices, device)
                pred_global = mask_and_renormalize(x_batch.mean(dim=1), valid, eps)
            elif model_name == "train_mean_observed_profile":
                pred_global = mask_and_renormalize(mean_profile_device[None, :], valid, eps)
            else:
                raise ValueError(f"Unknown baseline model: {model_name}")

            loss = profile_nll(
                torch,
                pred_global,
                target_batch,
                counts,
                weight_by_total_count=weight_by_total_count,
                eps=eps,
            )
            batch_size_actual = int(batch_indices.numel())
            total_loss += float(loss.detach().cpu()) * batch_size_actual
            total_windows += batch_size_actual

    return total_loss / total_windows


def evaluate_all_models(
    torch: Any,
    logits: Any,
    X: Any,
    Y: Any,
    valid_mask: Any,
    total_counts: Any,
    split_indices: dict[str, Any],
    train_mean_profile: Any,
    *,
    batch_size: int,
    device: str,
    eps: float,
    weight_by_total_count: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for split in SPLITS:
        indices = split_indices[split]
        reconstruction_loss = evaluate_reconstruction_head(
            torch,
            logits,
            X,
            Y,
            valid_mask,
            total_counts,
            indices,
            batch_size=batch_size,
            device=device,
            eps=eps,
            weight_by_total_count=weight_by_total_count,
        )
        rows.append({"split": split, "model": "reconstruction_head", "loss": reconstruction_loss})

        for baseline_name in BASELINE_MODELS:
            baseline_loss = evaluate_baseline(
                torch,
                baseline_name,
                X,
                Y,
                valid_mask,
                total_counts,
                indices,
                train_mean_profile=train_mean_profile,
                batch_size=batch_size,
                device=device,
                eps=eps,
                weight_by_total_count=weight_by_total_count,
            )
            rows.append({"split": split, "model": baseline_name, "loss": baseline_loss})

    return rows


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in vars(args).items():
        values[key] = str(value) if isinstance(value, Path) else value
    return values


def write_training_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "train_loss", "valid_loss"))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "train_loss": f"{row['train_loss']:.10f}",
                    "valid_loss": f"{row['valid_loss']:.10f}",
                }
            )


def write_evaluation_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("split", "model", "loss"))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "split": row["split"],
                    "model": row["model"],
                    "loss": f"{row['loss']:.10f}",
                }
            )


def write_track_weights(path: Path, logits: Any, weights: Any) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("track_index", "weight", "logit"))
        writer.writeheader()
        for track_index, (weight, logit) in enumerate(zip(weights.tolist(), logits.tolist())):
            writer.writerow(
                {
                    "track_index": track_index,
                    "weight": f"{float(weight):.10f}",
                    "logit": f"{float(logit):.10f}",
                }
            )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def find_loss(rows: list[dict[str, Any]], split: str, model: str) -> float:
    for row in rows:
        if row["split"] == split and row["model"] == model:
            return float(row["loss"])
    raise KeyError(f"Could not find loss for split={split!r}, model={model!r}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch = require_torch()
    set_seed(torch, args.seed)
    device = resolve_device(torch, args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ParnET predictions from {args.parnet_predictions}")
    pred = torch_load_cpu(torch, args.parnet_predictions, "ParnET predictions")
    print(f"Loading target profiles from {args.target_profiles}")
    target = torch_load_cpu(torch, args.target_profiles, "target profiles")

    X, Y, valid_mask, total_counts, shape_info = validate_tensors(torch, pred, target)
    alignment_info = validate_alignment(torch, pred, target, shape_info["num_windows"])
    split_values = metadata_to_list(torch, target["split"])
    split_indices, available_split_labels = make_split_indices(torch, split_values)
    split_counts = {split: int(indices.numel()) for split, indices in split_indices.items()}

    print(
        "Loaded aligned tensors: "
        f"N={shape_info['num_windows']}, tracks={shape_info['num_tracks']}, "
        f"length={shape_info['target_length']}"
    )
    print(f"Split counts: {split_counts}")
    print(f"Training on device: {device}")

    train_mean_profile = compute_train_mean_observed_profile(
        torch,
        Y,
        split_indices["train"],
        batch_size=args.batch_size,
        eps=args.eps,
    )

    logits = torch.nn.Parameter(torch.zeros(shape_info["num_tracks"], device=device, dtype=torch.float32))
    optimizer = torch.optim.Adam([logits], lr=args.lr)
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    metrics_rows: list[dict[str, Any]] = []
    best_epoch = 0
    best_valid_loss = math.inf
    best_logits = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            torch,
            logits,
            optimizer,
            X,
            Y,
            valid_mask,
            total_counts,
            split_indices["train"],
            batch_size=args.batch_size,
            device=device,
            eps=args.eps,
            weight_by_total_count=args.weight_by_total_count,
            generator=generator,
        )
        valid_loss = evaluate_reconstruction_head(
            torch,
            logits,
            X,
            Y,
            valid_mask,
            total_counts,
            split_indices["valid"],
            batch_size=args.batch_size,
            device=device,
            eps=args.eps,
            weight_by_total_count=args.weight_by_total_count,
        )
        metrics_rows.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
        print(f"epoch {epoch:03d} train_loss={train_loss:.6f} valid_loss={valid_loss:.6f}", flush=True)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_logits = logits.detach().cpu().clone()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}.")
                break

    if best_logits is None:
        raise RuntimeError("Training finished without producing a best checkpoint.")

    with torch.no_grad():
        logits.copy_(best_logits.to(device=device, dtype=torch.float32))

    best_weights = torch.softmax(best_logits, dim=0).cpu()
    evaluation_rows = evaluate_all_models(
        torch,
        logits,
        X,
        Y,
        valid_mask,
        total_counts,
        split_indices,
        train_mean_profile,
        batch_size=args.batch_size,
        device=device,
        eps=args.eps,
        weight_by_total_count=args.weight_by_total_count,
    )
    test_loss = find_loss(evaluation_rows, "test", "reconstruction_head")

    checkpoint = {
        "logits": best_logits,
        "weights": best_weights,
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid_loss,
        "num_tracks": shape_info["num_tracks"],
        "target_length": shape_info["target_length"],
        "args": serializable_args(args),
    }
    torch.save(checkpoint, args.output_dir / "checkpoint.pt")

    write_training_metrics(args.output_dir / "training_metrics.csv", metrics_rows)
    write_track_weights(args.output_dir / "reconstruction_weights_track_level.csv", best_logits, best_weights)
    write_evaluation_metrics(args.output_dir / "evaluation_metrics.csv", evaluation_rows)

    metadata = {
        "input_paths": {
            "parnet_predictions": str(args.parnet_predictions),
            "target_profiles": str(args.target_profiles),
        },
        "args": serializable_args(args),
        "shape_info": shape_info,
        "alignment": alignment_info,
        "available_split_labels": available_split_labels,
        "split_counts": split_counts,
        "device": {"requested": args.device, "resolved": device},
        "outputs": {
            "checkpoint": str(args.output_dir / "checkpoint.pt"),
            "training_metrics": str(args.output_dir / "training_metrics.csv"),
            "track_weights": str(args.output_dir / "reconstruction_weights_track_level.csv"),
            "evaluation_metrics": str(args.output_dir / "evaluation_metrics.csv"),
        },
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid_loss,
        "test_loss": test_loss,
    }
    write_json(args.output_dir / "metadata.json", metadata)

    top_k = min(20, int(best_weights.numel()))
    top_weights, top_indices = torch.topk(best_weights, k=top_k)

    print()
    print(f"Best validation loss: {best_valid_loss:.6f} at epoch {best_epoch}")
    print(f"Test loss: {test_loss:.6f}")
    print("Top 20 track weights:")
    for rank, (track_index, weight) in enumerate(zip(top_indices.tolist(), top_weights.tolist()), start=1):
        print(f"  {rank:02d}. track_index={track_index} weight={float(weight):.8f}")
    print(f"Sum of weights: {float(best_weights.sum()):.8f}")
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
