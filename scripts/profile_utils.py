#!/usr/bin/env python3
"""Utilities for converting count tracks into probability profiles."""

from __future__ import annotations

from typing import Any


def counts_to_profile(
    counts: Any,
    eps: float = 1e-6,
    length_axis: int = -1,
    mask: Any | None = None,
) -> Any:
    """Normalize count tracks into probability profiles.

    The transformation is:

        (counts + eps) / (counts.sum(length_axis) + eps * L)

    where L is the number of unmasked positions along ``length_axis``. The input
    shape is preserved. NumPy arrays and torch tensors are supported.
    """
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}")

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is optional for numpy users.
        torch = None

    if torch is not None and isinstance(counts, torch.Tensor):
        return _torch_counts_to_profile(counts, eps=eps, length_axis=length_axis, mask=mask)

    import numpy as np

    if not isinstance(counts, np.ndarray):
        counts = np.asarray(counts)
    return _numpy_counts_to_profile(counts, eps=eps, length_axis=length_axis, mask=mask)


def _normalize_axis(axis: int, ndim: int) -> int:
    if not -ndim <= axis < ndim:
        raise ValueError(f"length_axis {axis} is out of bounds for {ndim} dimensions")
    return axis % ndim


def _numpy_counts_to_profile(counts: Any, *, eps: float, length_axis: int, mask: Any | None) -> Any:
    import numpy as np

    if counts.size == 0:
        raise ValueError("counts must not be empty")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")

    axis = _normalize_axis(length_axis, counts.ndim)
    work = counts.astype(np.float64, copy=False)

    if mask is None:
        length = work.shape[axis]
        numerator = work + eps
        denominator = work.sum(axis=axis, keepdims=True) + eps * length
    else:
        mask_arr = np.asarray(mask, dtype=bool)
        mask_arr = np.broadcast_to(mask_arr, work.shape)
        length = mask_arr.sum(axis=axis, keepdims=True)
        if np.any(length == 0):
            raise ValueError("mask must leave at least one position per profile")
        numerator = np.where(mask_arr, work + eps, 0.0)
        denominator = np.where(mask_arr, work, 0.0).sum(axis=axis, keepdims=True) + eps * length

    return numerator / denominator


def _torch_counts_to_profile(counts: Any, *, eps: float, length_axis: int, mask: Any | None) -> Any:
    import torch

    if counts.numel() == 0:
        raise ValueError("counts must not be empty")
    if torch.any(counts < 0):
        raise ValueError("counts must be non-negative")

    axis = _normalize_axis(length_axis, counts.ndim)
    work = counts.to(dtype=torch.float64)

    if mask is None:
        length = work.shape[axis]
        numerator = work + eps
        denominator = work.sum(dim=axis, keepdim=True) + eps * length
    else:
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, device=work.device)
        mask = mask.to(device=work.device, dtype=torch.bool)
        mask = torch.broadcast_to(mask, work.shape)
        length = mask.sum(dim=axis, keepdim=True)
        if torch.any(length == 0):
            raise ValueError("mask must leave at least one position per profile")
        numerator = torch.where(mask, work + eps, torch.zeros_like(work))
        denominator = torch.where(mask, work, torch.zeros_like(work)).sum(dim=axis, keepdim=True)
        denominator = denominator + eps * length.to(dtype=work.dtype)

    return numerator / denominator


def _self_test() -> None:
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None:
        counts = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.float32)
        profile = counts_to_profile(counts, eps=1e-6)

        if not torch.allclose(profile.sum(dim=-1), torch.ones(2, dtype=profile.dtype)):
            raise AssertionError("profile rows do not sum to 1")
        if not torch.isfinite(profile).all():
            raise AssertionError("profile contains NaN or inf")
        if not (profile >= 0).all():
            raise AssertionError("profile contains negative values")

        try:
            counts_to_profile(torch.tensor([[0.0, -1.0, 2.0]]))
        except ValueError:
            pass
        else:
            raise AssertionError("negative counts were not rejected")

        padded_counts = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
        valid_mask = torch.tensor([[True, True, False, False]])
        masked_profile = counts_to_profile(padded_counts, eps=1e-6, mask=valid_mask)
        if masked_profile[0, 2:].abs().sum().item() != 0.0:
            raise AssertionError("masked padded positions are not exactly zero")
        if not torch.allclose(masked_profile.sum(dim=-1), torch.ones(1, dtype=masked_profile.dtype)):
            raise AssertionError("masked profile row does not sum to 1")
    else:
        import numpy as np

        counts = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.float32)
        profile = counts_to_profile(counts, eps=1e-6)

        if not np.allclose(profile.sum(axis=-1), 1.0):
            raise AssertionError("profile rows do not sum to 1")
        if not np.isfinite(profile).all():
            raise AssertionError("profile contains NaN or inf")
        if not (profile >= 0).all():
            raise AssertionError("profile contains negative values")

        try:
            counts_to_profile(np.array([[0.0, -1.0, 2.0]]))
        except ValueError:
            pass
        else:
            raise AssertionError("negative counts were not rejected")

        padded_counts = np.array([[1.0, 2.0, 0.0, 0.0]])
        valid_mask = np.array([[True, True, False, False]])
        masked_profile = counts_to_profile(padded_counts, eps=1e-6, mask=valid_mask)
        if masked_profile[0, 2:].sum() != 0.0:
            raise AssertionError("masked padded positions are not exactly zero")
        if not np.allclose(masked_profile.sum(axis=-1), 1.0):
            raise AssertionError("masked profile row does not sum to 1")

    print("profile_utils self-test passed")


if __name__ == "__main__":
    _self_test()
