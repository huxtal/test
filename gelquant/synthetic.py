"""Synthetic Coomassie-style gel images with known band amounts (for testing)."""

from __future__ import annotations

import numpy as np


def make_gel(lanes: list[list[tuple[float, float]]], height: int = 600, width: int = 800,
             top: float = 40, slope: float = 330, ref_mw: float = 250,
             band_sigma: float = 4.0, lane_width: float = 50, background: float = 220,
             gradient: float = 20, noise: float = 1.5, absorb: float = 120,
             rotate_deg: float = 0.0, seed: int = 0) -> np.ndarray:
    """Return a uint8 image: dark bands on a light, slightly uneven background.

    ``lanes`` holds, per lane, (MW kDa, amount) pairs; a band's integrated
    absorbance is proportional to its amount.  Band rows follow
    y = top + slope * log10(ref_mw / MW).
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(float)
    if rotate_deg:
        a = np.deg2rad(rotate_deg)
        cx, cy = width / 2, height / 2
        xr = (xx - cx) * np.cos(a) + (yy - cy) * np.sin(a) + cx
        yr = -(xx - cx) * np.sin(a) + (yy - cy) * np.cos(a) + cy
        xx, yy = xr, yr
    n = len(lanes)
    spacing = width / (n + 1)
    absorb_img = np.zeros((height, width))
    for i, bands in enumerate(lanes):
        cx = spacing * (i + 1)
        # Flat-topped lane profile with soft edges.
        xprof = 1 / (1 + np.exp((np.abs(xx - cx) - lane_width / 2) / 1.5))
        for mw, amount in bands:
            y0 = top + slope * np.log10(ref_mw / mw)
            yprof = np.exp(-0.5 * ((yy - y0) / band_sigma) ** 2) / (band_sigma * np.sqrt(2 * np.pi))
            absorb_img += amount * xprof * yprof
    bg = background - gradient * (yy / height) - 0.5 * gradient * (xx / width)
    img = bg - absorb * absorb_img + rng.normal(0, noise, (height, width))
    return np.clip(np.round(img), 0, 255).astype(np.uint8)
