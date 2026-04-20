"""Scalar metrics computed over SDRs."""

from __future__ import annotations

import logging

import numpy as np

from halo.core.sdr import SDR

logger = logging.getLogger(__name__)

__all__ = ["sparsity", "overlap_score", "entropy"]


def sparsity(sdr: SDR) -> float:
    """Return the fraction of active columns in *sdr*."""
    return sdr.sparsity


def overlap_score(a: SDR, b: SDR) -> float:
    """Normalised overlap: shared_bits / min(|a|, |b|).

    Returns 0.0 if either SDR has no active bits.
    """
    shared = a.overlap(b)
    denom = min(int(a.bits.sum()), int(b.bits.sum()))
    if denom == 0:
        return 0.0
    return float(shared) / float(denom)


def entropy(sdrs: list[SDR]) -> float:
    """Shannon entropy over column activation frequencies across *sdrs*.

    Computes the per-column activation probability across all SDRs, then
    returns the mean binary entropy H(p) = -p log2(p) - (1-p) log2(1-p).

    Returns 0.0 for an empty list.
    """
    if not sdrs:
        return 0.0

    stacked = np.stack([s.bits.astype(float) for s in sdrs], axis=0)  # (T, n)
    p = stacked.mean(axis=0)  # per-column activation probability

    # Avoid log(0): clip to (eps, 1-eps)
    eps = 1e-10
    p_clipped = np.clip(p, eps, 1.0 - eps)
    h = -p_clipped * np.log2(p_clipped) - (1.0 - p_clipped) * np.log2(1.0 - p_clipped)
    return float(h.mean())
