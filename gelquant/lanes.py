"""Lane detection from the column profile of a gel image."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks


@dataclass
class Lane:
    number: int        # 1-based, left to right
    center: float      # x pixel
    width: float       # pixels used for the profile (central part of the lane)

    @property
    def x0(self) -> int:
        return int(round(self.center - self.width / 2))

    @property
    def x1(self) -> int:
        return int(round(self.center + self.width / 2))


def column_profile(sig: np.ndarray) -> np.ndarray:
    """Per-column band signal.

    Each row's own background (its 20th percentile across the gel) is
    removed first, so slow background gradients cancel; then the 99th
    percentile down each column picks up bands even in lanes that hold a
    single band.
    """
    s = gaussian_filter(sig, 2)
    s = np.clip(s - np.percentile(s, 20, axis=1, keepdims=True), 0, None)
    return np.percentile(s, 99, axis=0)


def _half_max_center(prof: np.ndarray, peak: int) -> float:
    """Midpoint of the half-maximum edges around ``peak``.

    Lanes are flat-topped, so the argmax lands anywhere on the plateau; the
    midpoint between the edges where the profile drops to half the peak
    (relative to the local minima on either side) is a stable centre.
    """
    n = len(prof)
    lo = peak
    while lo > 0 and prof[lo - 1] <= prof[lo] * 1.05 + 1e-12 and lo > peak - n // 4:
        lo -= 1
    hi = peak
    while hi < n - 1 and prof[hi + 1] <= prof[hi] * 1.05 + 1e-12 and hi < peak + n // 4:
        hi += 1
    base = max(prof[lo], prof[hi])
    half = base + 0.5 * (prof[peak] - base)
    a = peak
    while a > lo and prof[a - 1] >= half:
        a -= 1
    b = peak
    while b < hi and prof[b + 1] >= half:
        b += 1
    return 0.5 * (a + b)


def _fit_grid(found: np.ndarray, n: int, width: int) -> np.ndarray:
    """n evenly spaced centres (inside the image) closest to the found ones."""
    if len(found) < 2:
        return np.linspace(width / (2 * n), width - width / (2 * n), n)
    cands = []
    for s in np.linspace(0.5 * width / n, width / max(n - 1, 1), 300):
        # Offsets that put lane k of the grid on one of the found lanes.
        for f in found:
            for k in range(n):
                grid = f - k * s + s * np.arange(n)
                if grid[0] < 0 or grid[-1] > width:
                    continue
                cost = np.abs(found[:, None] - grid[None, :]).min(axis=1).mean() / s
                cands.append((cost, s, grid))
    if not cands:
        return np.linspace(found[0], found[-1], n)
    # Half (or a third of) the true spacing also fits the found lanes, so
    # among near-perfect grids take the widest one.
    min_cost = min(c[0] for c in cands)
    good = [c for c in cands if c[0] <= min_cost + 0.03]
    grid = max(good, key=lambda c: (c[1], -c[0]))[2]
    # Refine spacing and offset by least squares on the lanes actually found.
    s0 = grid[1] - grid[0]
    k = np.rint((found - grid[0]) / s0)
    if len(np.unique(k)) >= 2:
        s1, o1 = np.polyfit(k, found, 1)
        grid = o1 + s1 * np.arange(n)
    return grid


def detect_lanes(sig: np.ndarray, n_lanes: int | None = None,
                 centers: list[float] | None = None,
                 width_frac: float = 0.6) -> list[Lane]:
    """Find lane centres; lane width is ``width_frac`` of the median spacing.

    ``centers`` (pixels) overrides detection; ``n_lanes`` keeps the n most
    prominent lanes (or, if fewer are found, divides the span evenly).
    """
    w = sig.shape[1]
    if centers:
        c = np.sort(np.asarray(centers, float))
    else:
        prof = column_profile(sig)
        prof = gaussian_filter1d(prof - np.percentile(prof, 5), sigma=max(1.0, w / 300))
        min_dist = max(3, w // 40)
        peaks, props = find_peaks(prof, distance=min_dist,
                                  prominence=0.08 * max(prof.max(), 1e-12))
        c = np.array([_half_max_center(prof, int(pk)) for pk in peaks], float)
        if n_lanes and len(c) > n_lanes:
            keep = np.argsort(props["prominences"])[::-1][:n_lanes]
            c = np.sort(c[keep])
        elif n_lanes and len(c) < n_lanes:
            # Empty or faint lanes were missed: comb wells are evenly spaced,
            # so place n lanes on the regular grid that best fits those found.
            c = _fit_grid(c, n_lanes, w)
    if len(c) == 0:
        raise ValueError("no lanes found; pass the lane count (--lanes N) or centres")
    spacing = float(np.median(np.diff(c))) if len(c) > 1 else w * 0.1
    width = width_frac * spacing
    return [Lane(i + 1, float(x), width) for i, x in enumerate(c)]
