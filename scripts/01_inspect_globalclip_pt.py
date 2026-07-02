#!/usr/bin/env python3

import argparse
from pathlib import Path
import torch


def preview_value(x, n=3):
    if isinstance(x, (list, tuple)):
        return x[:n]
    return x


def describe_tensor(name, x):
    print(f"\n[{name}]")
    print(f"type      : {type(x)}")
    print(f"shape     : {tuple(x.shape)}")
    print(f"dtype     : {x.dtype}")
    print(f"is_sparse : {x.is_sparse if hasattr(x, 'is_sparse') else False}")

    if hasattr(x, "is_sparse") and x.is_sparse:
        print(f"nnz       : {x._nnz()} non-zero values")
        print("note      : sparse tensor, not converting whole tensor to dense")
        return

    if x.numel() > 0 and x.numel() < 5_000_000:
        xf = x.float()
        print(f"min       : {xf.min().item():.4g}")
        print(f"max       : {xf.max().item():.4g}")
        print(f"mean      : {xf.mean().item():.4g}")
        print(f"sum       : {xf.sum().item():.4g}")
    else:
        print("note      : tensor too large for full summary")


def describe_obj(name, obj, depth=0, max_depth=3):
    indent = "  " * depth

    if depth > max_depth:
        print(f"{indent}{name}: ... max depth reached")
        return

    if isinstance(obj, torch.Tensor):
        describe_tensor(indent + str(name), obj)

    elif isinstance(obj, dict):
        print(f"{indent}{name}: dict with {len(obj)} keys")
        print(f"{indent}keys: {list(obj.keys())}")
        for k, v in obj.items():
            describe_obj(k, v, depth + 1, max_depth)

    elif isinstance(obj, (list, tuple)):
        print(f"{indent}{name}: {type(obj).__name__}, length={len(obj)}")
        for i, item in enumerate(obj[:3]):
            describe_obj(f"[{i}]", item, depth + 1, max_depth)
        if len(obj) > 3:
            print(f"{indent}  ... {len(obj) - 3} more items")

    else:
        text = repr(obj)
        if len(text) > 300:
            text = text[:300] + "..."
        print(f"{indent}{name}: {type(obj).__name__} = {text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    path = Path(args.path)
    print(f"Loading file: {path}")
    print(f"File size   : {path.stat().st_size / 1024 / 1024:.2f} MB")

    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")

    print("\nLoaded successfully.\n")
    describe_obj("root", data)


if __name__ == "__main__":
    main()
