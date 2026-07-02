#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch


def sparse_dict_to_dense(x, target_len=600):
    """
    Convert a dict-style sparse signal into a dense tensor.

    Expected keys are usually:
      - indices
      - values
      - size / shape / dense_shape  (if available)

    If shape is missing, we assume one track and target_len positions: [1, 600].
    """
    if not isinstance(x, dict):
        return x

    print(f"dict keys  : {list(x.keys())}")

    if "indices" not in x or "values" not in x:
        print("Cannot convert dict to dense: missing indices or values")
        return x

    indices = x["indices"]
    values = x["values"]

    print(f"indices shape: {tuple(indices.shape)}")
    print(f"values shape : {tuple(values.shape)}")
    print(f"values dtype : {values.dtype}")

    shape = None
    for key in ["size", "shape", "dense_shape"]:
        if key in x:
            shape = x[key]
            print(f"{key}       : {shape}")
            break

    if shape is None:
        # Most likely one track x 600 positions.
        # indices probably has shape [2, nnz], first row = track index, second row = position.
        if indices.ndim == 2 and indices.shape[0] == 2:
            shape = (1, target_len)
        elif indices.ndim == 1:
            shape = (target_len,)
        else:
            raise ValueError(f"Cannot infer sparse shape from indices shape {tuple(indices.shape)}")

        print(f"inferred shape: {shape}")

    if isinstance(shape, torch.Tensor):
        shape = tuple(shape.tolist())
    elif not isinstance(shape, (tuple, list, torch.Size)):
        shape = tuple(shape)
    else:
        shape = tuple(shape)

    dense = torch.sparse_coo_tensor(indices.long(), values.float(), size=shape).coalesce().to_dense()
    return dense


def summarize_signal(name, x, target_len=600):
    print(f"\n[{name}]")
    print(f"type      : {type(x)}")

    if isinstance(x, dict):
        dense = sparse_dict_to_dense(x, target_len=target_len)
    elif isinstance(x, torch.Tensor) and x.is_sparse:
        print(f"shape     : {tuple(x.shape)}")
        print(f"dtype     : {x.dtype}")
        print(f"is_sparse : {x.is_sparse}")
        print(f"nnz       : {x._nnz()} non-zero values")
        dense = x.to_dense()
    else:
        dense = x

    if isinstance(dense, torch.Tensor):
        print(f"dense shape: {tuple(dense.shape)}")
        print(f"dense dtype: {dense.dtype}")

        dense = dense.float()
        print(f"sum       : {dense.sum().item():.4f}")
        print(f"max       : {dense.max().item():.4f}")
        print(f"mean      : {dense.mean().item():.6f}")
        print(f"nonzero   : {(dense != 0).sum().item()} positions")

        flat = dense.reshape(-1)
        nz = torch.nonzero(flat, as_tuple=False).reshape(-1)
        print(f"first 20 nonzero indices: {nz[:20].tolist()}")

        if len(nz) > 0:
            print(f"first 20 nonzero values : {flat[nz[:20]].tolist()}")

        # Show a small window around the strongest signal position.
        if flat.numel() > 0:
            max_idx = int(torch.argmax(flat).item())
            left = max(0, max_idx - 10)
            right = min(flat.numel(), max_idx + 11)
            print(f"max index : {max_idx}")
            print(f"local signal around max [{left}:{right}]:")
            print(flat[left:right].tolist())

    else:
        print(f"value     : {repr(dense)[:500]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--target-len", type=int, default=600)
    args = parser.parse_args()

    path = Path(args.path)
    print(f"Loading: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)

    print(f"\nTop-level keys: {list(data.keys())}")
    print("Split sizes:")
    for split in ["train", "valid", "test"]:
        print(f"  {split}: {len(data[split])}")

    sample = data[args.split][args.index]

    print(f"\nInspecting sample: {args.split}[{args.index}]")
    print(f"Sample keys: {list(sample.keys())}")

    print("\n[meta]")
    for k, v in sample["meta"].items():
        print(f"{k}: {v}")

    seq = sample["inputs"]["sequence"]
    print("\n[sequence]")
    print(f"type      : {type(seq)}")
    print(f"length    : {len(seq)}")
    print(f"first 80  : {seq[:80]}")
    print(f"last 80   : {seq[-80:]}")
    print(f"alphabet  : {sorted(set(seq))}")

    summarize_signal("outputs/globalCLIP", sample["outputs"]["globalCLIP"], target_len=args.target_len)
    summarize_signal("outputs/control", sample["outputs"]["control"], target_len=args.target_len)

    print("\nDone. This script only reads one sample and does not modify data.")


if __name__ == "__main__":
    main()
