#!/usr/bin/env python3
"""Inspect configured globalCLIP inputs and the PARNET-compatible data contract.

This script intentionally does not assume a globalCLIP-specific Python API.  The
dataset sibling repo documents raw eCLIP loading from BigWig/BED files, while the
model sibling repo documents the PARNET-ready row schema:

    {"inputs": {"sequence": ...}, "outputs": {...}, "meta": {...}}

The probe below inspects the configured manifest, window, and profile locations
using common on-disk formats and reports the keys/shapes it can observe.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


TEXT_SUFFIXES = {".csv", ".tsv", ".txt", ".bed"}
FASTA_SUFFIXES = {".fa", ".fasta", ".fna"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect globalCLIP manifest/windows/profiles before model wiring."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vm.yaml"),
        help="Path to the VM YAML/JSON config.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of rows/records/files to preview.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if configured input paths are missing.",
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


def require_path(
    config: dict[str, Any],
    key: str,
    *,
    project_root: Path,
) -> Path:
    paths = get_section(config, "paths")
    if key not in paths:
        raise SystemExit(f"Missing required config entry: paths.{key}")
    return resolve_config_path(paths[key], project_root=project_root)


def open_text(path: Path) -> Iterator[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            yield from handle
    else:
        with path.open("r", encoding="utf-8", newline="") as handle:
            yield from handle


def sniff_delimiter(path: Path, sample_text: str) -> str:
    suffixes = path.suffixes
    if ".tsv" in suffixes or ".bed" in suffixes:
        return "\t"
    if ".csv" in suffixes:
        return ","
    try:
        return csv.Sniffer().sniff(sample_text, delimiters=",\t;").delimiter
    except csv.Error:
        return "\t" if "\t" in sample_text else ","


def preview_table(path: Path, *, limit: int) -> None:
    if not path.exists():
        print(f"  missing: {path}")
        return

    lines = open_text(path)
    sample_lines: list[str] = []
    for _ in range(20):
        try:
            sample_lines.append(next(lines))
        except StopIteration:
            break
    sample_text = "".join(sample_lines)
    delimiter = sniff_delimiter(path, sample_text)

    with (
        gzip.open(path, "rt", encoding="utf-8", newline="")
        if path.suffix == ".gz"
        else path.open("r", encoding="utf-8", newline="")
    ) as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        print(f"  columns ({len(reader.fieldnames or [])}): {reader.fieldnames or []}")
        for index, row in enumerate(reader):
            if index >= limit:
                break
            print(f"  row {index + 1}: {row}")


def fasta_records(path: Path, *, limit: int) -> list[tuple[str, int, str]]:
    records: list[tuple[str, int, str]] = []
    name: str | None = None
    chunks: list[str] = []

    for raw_line in open_text(path):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                seq = "".join(chunks)
                records.append((name, len(seq), seq[:40]))
                if len(records) >= limit:
                    return records
            name = line[1:].strip()
            chunks = []
        else:
            chunks.append(line)

    if name is not None and len(records) < limit:
        seq = "".join(chunks)
        records.append((name, len(seq), seq[:40]))

    return records


def preview_windows(path: Path, *, limit: int, expected_len: int | None) -> None:
    if not path.exists():
        print(f"  missing: {path}")
        return

    suffixes = set(path.suffixes)
    plain_suffixes = {suffix.lower() for suffix in suffixes if suffix != ".gz"}
    if plain_suffixes & FASTA_SUFFIXES:
        records = fasta_records(path, limit=limit)
        print(f"  format: FASTA")
        for index, (name, seq_len, preview) in enumerate(records, start=1):
            status = "ok" if expected_len is None or seq_len == expected_len else "length-mismatch"
            print(f"  record {index}: name={name!r} length={seq_len} status={status} seq[:40]={preview}")
        return

    if plain_suffixes & TEXT_SUFFIXES:
        print("  format: delimited text")
        preview_table(path, limit=limit)
        return

    print(f"  format: unknown from suffixes {path.suffixes}; showing path only")


def describe_array_file(path: Path) -> None:
    suffixes = path.suffixes
    if path.suffix == ".npz":
        import numpy as np

        with np.load(path) as data:
            print(f"  npz keys: {list(data.files)}")
            for key in data.files:
                arr = data[key]
                print(f"    {key}: shape={arr.shape} dtype={arr.dtype}")
        return

    if path.suffix == ".npy":
        import numpy as np

        arr = np.load(path, mmap_mode="r")
        print(f"  npy: shape={arr.shape} dtype={arr.dtype}")
        return

    if ".pt" in suffixes:
        import torch

        obj = torch.load(path, mmap=True, weights_only=False)
        describe_object(obj, indent="  ")
        return

    if path.suffix.lower() in TEXT_SUFFIXES or any(s in TEXT_SUFFIXES for s in suffixes):
        preview_table(path, limit=3)
        return

    print(f"  unsupported profile file suffixes: {path.suffixes}")


def describe_object(obj: Any, *, indent: str = "") -> None:
    if isinstance(obj, dict):
        print(f"{indent}dict keys: {list(obj.keys())}")
        for key, value in list(obj.items())[:8]:
            print(f"{indent}  {key!r}: {short_type(value)}")
        return

    if isinstance(obj, list):
        print(f"{indent}list len: {len(obj)}")
        if obj:
            print(f"{indent}first item: {short_type(obj[0])}")
            describe_object(obj[0], indent=indent + "  ")
        return

    print(f"{indent}{short_type(obj)}")


def short_type(value: Any) -> str:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        dtype_text = f" dtype={dtype}" if dtype is not None else ""
        return f"{type(value).__name__} shape={tuple(shape)}{dtype_text}"
    return type(value).__name__


def preview_profiles(path: Path, *, limit: int) -> None:
    if not path.exists():
        print(f"  missing: {path}")
        return

    if path.is_file():
        print(f"  file: {path}")
        describe_array_file(path)
        return

    files = sorted(p for p in path.iterdir() if p.is_file())
    print(f"  directory files: {len(files)}")
    for index, file_path in enumerate(files[:limit], start=1):
        print(f"  file {index}: {file_path.name}")
        try:
            describe_array_file(file_path)
        except ImportError as exc:
            print(f"    cannot inspect without optional dependency: {exc.name}")
        except Exception as exc:
            print(f"    cannot inspect: {exc}")


def print_parnet_contract(config: dict[str, Any]) -> None:
    model = get_section(config, "model")
    input_window_nt = model.get("input_window_nt", 600)
    profiles = model.get("rbp_cell_line_profiles", 223)

    print("PARNET-compatible row contract from sibling model repo:")
    print("  sample keys: inputs, outputs, meta")
    print("  inputs keys: sequence")
    print("  stored outputs keys: eCLIP, control")
    print("  loader output keys: total, control")
    print("  sparse signal keys: indices, values, size")
    print("  meta keys observed: name, pad_side")
    print(f"  model input: batch['inputs']['sequence'] -> (B, 4, {input_window_nt})")
    print(f"  pretrained total output: out['total'] -> (B, {profiles}, {input_window_nt})")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    project_root = project_root_from_config(config_path)
    expected_len = int(get_section(config, "model").get("input_window_nt", 600))

    manifest_path = require_path(config, "globalclip_manifest", project_root=project_root)
    windows_path = require_path(config, "globalclip_windows", project_root=project_root)
    profiles_path = require_path(config, "globalclip_profiles", project_root=project_root)

    missing = [p for p in (manifest_path, windows_path, profiles_path) if not p.exists()]
    if args.strict and missing:
        raise SystemExit("Missing configured input path(s): " + ", ".join(str(p) for p in missing))

    print(f"Config: {config_path}")
    print(f"Project root: {project_root}")
    print()

    print("Configured globalCLIP inputs:")
    print(f"  manifest: {manifest_path}")
    print(f"  windows:  {windows_path}")
    print(f"  profiles: {profiles_path}")
    print()

    print("Manifest schema:")
    preview_table(manifest_path, limit=args.limit)
    print()

    print("Window schema:")
    preview_windows(windows_path, limit=args.limit, expected_len=expected_len)
    print()

    print("Profile schema:")
    preview_profiles(profiles_path, limit=args.limit)
    print()

    print_parnet_contract(config)


if __name__ == "__main__":
    main()
