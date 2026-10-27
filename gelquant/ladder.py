"""Molecular-weight calibration from a protein ladder lane."""

from __future__ import annotations

from dataclasses import dataclass

import itertools

import numpy as np
from scipy.interpolate import PchipInterpolator

from .bands import Band

# Band sizes (kDa), top to bottom, as listed by the manufacturers for
# Tris-glycine SDS-PAGE.  Prestained markers migrate somewhat differently
# in other buffer systems; pass --ladder-mw to use your own list.
LADDERS: dict[str, list[float]] = {
    "benchmark": [220, 160, 120, 100, 90, 80, 70, 60, 50, 40, 30, 25,
                  20, 15, 10],                                              # Invitrogen 10747-012
    "pageruler-plus": [250, 130, 100, 70, 55, 35, 25, 15, 10],                 # Thermo 26619
    "pageruler-prestained": [180, 130, 100, 70, 55, 40, 35, 25, 15, 10],       # Thermo 26616
    "pageruler-unstained": [200, 150, 120, 100, 85, 70, 60, 50, 40, 30,
                            25, 20, 15, 10],                                    # Thermo 26614
    "precision-plus": [250, 150, 100, 75, 50, 37, 25, 20, 15, 10],             # Bio-Rad 161-0373/4
}

# Protein per band per uL of ladder, where the manufacturer states it.
# BenchMark user guide: "each protein ... approximately 0.1 ug/uL".  The
# intensified reference bands' amount is not stated, so they are excluded
# from mass calibration unless --ladder-ref-ug is given.
LADDER_UG_PER_UL: dict[str, float] = {
    "benchmark": 0.1,
}

# Sizes that are deliberately more intense in each ladder (orientation bands).
REFERENCE_BANDS: dict[str, list[float]] = {
    "benchmark": [50, 20],
    "pageruler-plus": [70, 25],
    "pageruler-prestained": [70],
    "pageruler-unstained": [50],
    "precision-plus": [75, 25],
}


@dataclass
class Calibration:
    y: np.ndarray           # ladder band rows used (top to bottom)
    mw: np.ndarray          # their sizes (kDa)
    unmatched: int          # ladder sizes without a detected band
    reference_ok: bool      # intensified reference bands landed on strong peaks

    def predict(self, y) -> np.ndarray:
        """MW (kDa) by monotone interpolation of log10(MW) between ladder bands;
        linear extrapolation from the end segments outside the ladder."""
        y = np.asarray(y, float)
        lm = np.log10(self.mw)
        out = PchipInterpolator(self.y, lm, extrapolate=False)(y)
        lo = y < self.y[0]
        hi = y > self.y[-1]
        out = np.where(lo, lm[0] + (y - self.y[0]) * (lm[1] - lm[0]) / (self.y[1] - self.y[0]), out)
        out = np.where(hi, lm[-1] + (y - self.y[-1]) * (lm[-1] - lm[-2]) / (self.y[-1] - self.y[-2]), out)
        return 10 ** out

    def extrapolated(self, y) -> np.ndarray:
        y = np.asarray(y, float)
        return (y < self.y[0]) | (y > self.y[-1])


def calibrate(bands: list[Band], sizes: list[float],
              reference: list[float] | None = None) -> Calibration:
    """Assign ladder sizes to detected ladder-lane peaks.

    Ladder bands are sharp, so the tallest peaks in the lane are taken as the
    ladder (background bumps are low and broad) and paired top-to-bottom
    with the sizes.  If fewer peaks than sizes are found, the missing sizes
    are chosen so that the ladder's intensified ``reference`` bands fall on the
    strongest peaks, then so that log10(MW) vs. position is smoothest.
    """
    sizes = sorted(sizes, reverse=True)
    if len(bands) < 2:
        raise ValueError("fewer than 2 bands detected in the ladder lane")
    ns = len(sizes)
    tallest = sorted(bands, key=lambda b: b.height, reverse=True)[:ns]
    chosen = sorted(tallest, key=lambda b: b.y)
    y = np.array([b.y for b in chosen])
    h = np.array([b.height for b in chosen])
    n_ref = len(reference or [])
    strong = set(np.argsort(-h)[:n_ref + 1]) if n_ref else set()

    def ref_hits(size_idx) -> int:
        return sum(1 for j, i in enumerate(size_idx) if sizes[i] in (reference or []) and j in strong)

    if len(chosen) == ns:
        idx = list(range(ns))
    else:
        best = None
        logm = np.log10(sizes)
        for idx_c in itertools.combinations(range(ns), len(chosen)):
            idx_c = list(idx_c)
            lm = logm[idx_c]
            deg = min(3, len(y) - 1)
            c = np.polyfit(y, lm, deg)
            rms = float(np.sqrt(np.mean((np.polyval(c, y) - lm) ** 2)))
            score = (ref_hits(idx_c), -rms)
            if best is None or score > best[0]:
                best = (score, idx_c)
        idx = best[1]
    mw = np.array([sizes[i] for i in idx], float)
    ok = (ref_hits(idx) == sum(1 for i in idx if sizes[i] in (reference or []))) if n_ref else True
    return Calibration(y, mw, ns - len(idx), ok)
