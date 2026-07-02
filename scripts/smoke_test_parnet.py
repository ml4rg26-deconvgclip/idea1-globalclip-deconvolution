#!/usr/bin/env python3
"""Load pretrained PARNET and run a minimal forward pass.

The sibling training notebooks establish the model contract:

    model = load_parnet_model(ParnetModelName.PARNET_7M_0_0, checkpoint, ...)
    out = model(batch["inputs"]["sequence"])

where the input tensor has shape ``(B, 4, 600)`` and the pretrained output
contains ``out["total"]`` with shape ``(B, 223, 600)``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test pretrained PARNET loading and inference."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vm.yaml"),
        help="Path to the VM YAML/JSON config.",
    )
    parser.add_argument(
        "--num-windows",
        type=int,
        default=2,
        help="Number of 600 nt windows in the smoke batch.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Device for inference.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Parnet model name. Defaults to model.parnet_model_name or parnet.7m-0.0.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint path override.",
    )
    parser.add_argument(
        "--windows",
        type=Path,
        default=None,
        help="Optional FASTA file with windows to use instead of synthetic sequences.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve configuration and input batch only; do not import/load PARNET.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    config_text = config_path.read_text(encoding="utf-8")

    try:
        import yaml
    except ImportError:
        try:
            config = json.loads(config_text)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                "PyYAML is required for non-JSON YAML configs. "
                "Install it or write the config as JSON."
            ) from exc
    else:
        config = yaml.safe_load(config_text) or {}

    if not isinstance(config, dict):
        raise SystemExit(f"Expected a mapping in config file: {config_path}")

    return config


def project_root_from_config(config_path: Path) -> Path:
    return config_path.resolve().parent.parent


def resolve_config_path(raw_path: str | Path, *, project_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else project_root / path


def get_section(config: dict[str, Any], section: str) -> dict[str, Any]:
    values = config.get(section, {})
    if not isinstance(values, dict):
        raise SystemExit(f"Expected config section to be a mapping: {section}")
    return values


def choose_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def parse_fasta(path: Path, *, limit: int) -> list[str]:
    sequences: list[str] = []
    chunks: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if chunks:
                    sequences.append("".join(chunks))
                    chunks = []
                    if len(sequences) >= limit:
                        return sequences
            else:
                chunks.append(line)

    if chunks and len(sequences) < limit:
        sequences.append("".join(chunks))

    return sequences


def synthetic_sequences(count: int, length: int) -> list[str]:
    motif = "ACGT"
    return [(motif * ((length // len(motif)) + 1))[:length] for _ in range(count)]


def load_sequences(
    *,
    windows_path: Path | None,
    count: int,
    length: int,
) -> tuple[list[str], str]:
    if windows_path is not None and windows_path.exists():
        sequences = parse_fasta(windows_path, limit=count)
        source = str(windows_path)
    else:
        sequences = synthetic_sequences(count, length)
        source = "synthetic ACGT windows"

    if not sequences:
        raise SystemExit("No sequences available for smoke-test batch.")

    bad_lengths = [len(seq) for seq in sequences if len(seq) != length]
    if bad_lengths:
        raise SystemExit(
            f"PARNET smoke test expects {length} nt windows; observed lengths: {bad_lengths[:10]}"
        )

    return sequences, source


def sequences_to_onehot(torch: Any, sequences: list[str]) -> Any:
    length = len(sequences[0])
    batch = torch.zeros((len(sequences), 4, length), dtype=torch.float32)
    base_to_channel = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}

    for batch_index, seq in enumerate(sequences):
        for position, base in enumerate(seq.upper()):
            channel = base_to_channel.get(base)
            if channel is not None:
                batch[batch_index, channel, position] = 1.0

    return batch


def resolve_model_name(raw_name: str, enum_cls: Any) -> Any:
    try:
        return enum_cls(raw_name)
    except Exception:
        attr = raw_name.upper().replace(".", "_").replace("-", "_")
        if not attr.startswith("PARNET_"):
            attr = "PARNET_" + attr
        try:
            return getattr(enum_cls, attr)
        except AttributeError as exc:
            valid = [getattr(member, "value", str(member)) for member in enum_cls]
            raise SystemExit(f"Unknown PARNET model name {raw_name!r}. Known values: {valid}") from exc


def output_shape(value: Any) -> tuple[int, ...] | str:
    shape = getattr(value, "shape", None)
    return tuple(shape) if shape is not None else type(value).__name__


def validate_outputs(
    out: Any,
    *,
    batch_size: int,
    profile_count: int,
    seq_len: int,
) -> None:
    if not isinstance(out, dict):
        raise SystemExit(f"Expected model output dict, got {type(out).__name__}")
    if "total" not in out:
        raise SystemExit(f"Expected output key 'total'; observed keys: {list(out.keys())}")

    total_shape = tuple(out["total"].shape)
    expected_total_shape = (batch_size, profile_count, seq_len)
    if total_shape != expected_total_shape:
        raise SystemExit(
            f"Expected out['total'] shape {expected_total_shape}, observed {total_shape}"
        )

    if "mix_coeff" in out:
        mix_shape = tuple(out["mix_coeff"].shape)
        expected_mix_shape = (batch_size, profile_count)
        if mix_shape != expected_mix_shape:
            raise SystemExit(
                f"Expected out['mix_coeff'] shape {expected_mix_shape}, observed {mix_shape}"
            )


def main() -> None:
    args = parse_args()
    if args.num_windows <= 0:
        raise SystemExit("--num-windows must be positive")

    config_path = args.config.resolve()
    config = load_config(config_path)
    project_root = project_root_from_config(config_path)
    paths = get_section(config, "paths")
    model_cfg = get_section(config, "model")

    seq_len = int(model_cfg.get("input_window_nt", 600))
    profile_count = int(model_cfg.get("rbp_cell_line_profiles", 223))
    model_name = args.model_name or str(model_cfg.get("parnet_model_name", "parnet.7m-0.0"))

    raw_checkpoint = args.checkpoint or paths.get("parnet_checkpoint")
    if raw_checkpoint is None:
        raise SystemExit("Missing checkpoint path: provide --checkpoint or paths.parnet_checkpoint")
    checkpoint_path = resolve_config_path(raw_checkpoint, project_root=project_root)

    raw_windows = args.windows or paths.get("globalclip_windows")
    windows_path = (
        resolve_config_path(raw_windows, project_root=project_root)
        if raw_windows is not None
        else None
    )

    sequences, sequence_source = load_sequences(
        windows_path=windows_path,
        count=args.num_windows,
        length=seq_len,
    )

    print(f"Config: {config_path}")
    print(f"Project root: {project_root}")
    print(f"Model name: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint exists: {checkpoint_path.exists()}")
    print(f"Sequence source: {sequence_source}")
    print(f"Smoke batch: ({len(sequences)}, 4, {seq_len})")
    print(f"Expected total output: ({len(sequences)}, {profile_count}, {seq_len})")

    if args.dry_run:
        print("Dry run complete. PARNET was not imported or loaded.")
        return

    if not checkpoint_path.exists():
        raise SystemExit(f"Configured PARNET checkpoint does not exist: {checkpoint_path}")

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for the PARNET smoke test. Run inside the model repo pixi env."
        ) from exc

    try:
        from parnet_additional_utils import ParnetModelName, load_parnet_model
    except ImportError as exc:
        raise SystemExit(
            "parnet_additional_utils is required. Run inside the parnet demo pixi environment."
        ) from exc

    device = choose_device(torch, args.device)
    model_enum_value = resolve_model_name(model_name, ParnetModelName)
    batch = sequences_to_onehot(torch, sequences).to(device)

    model = load_parnet_model(
        model_enum_value,
        checkpoint_path,
        dtype=torch.float32,
        device=device,
    )
    model.eval()

    with torch.no_grad():
        out = model(batch)

    print("Observed output shapes:")
    for key, value in out.items():
        print(f"  {key}: {output_shape(value)}")

    validate_outputs(
        out,
        batch_size=len(sequences),
        profile_count=profile_count,
        seq_len=seq_len,
    )
    print("PARNET smoke test passed.")


if __name__ == "__main__":
    main()
