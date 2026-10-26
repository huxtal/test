"""Lane profiles, background removal and band detection / integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import (gaussian_filter1d, maximum_filter1d, minimum_filter1d,
                           uniform_filter1d)
from scipy.signal import find_peaks, peak_widths

from .lanes import Lane


@dataclass
class Band:
    lane: int
    index: int          # 1-based, top to bottom within the lane
    y: float            # peak row (pixels)
    y0: int             # integration start row
    y1: int             # integration end row (exclusive)
    height: float       # background-subtracted peak signal
    volume: float       # integrated signal (sum over rows x lane width)
    mw: float = float("nan")      # kDa, from the ladder calibration
    percent: float = float("nan")  # % of the lane's total band volume


def finalize(lp: "LaneProfile") -> "LaneProfile":
    total = sum(b.volume for b in lp.bands)
    for b in lp.bands:
        b.percent = 100 * b.volume / total if total > 0 else float("nan")
    return lp


@dataclass
class LaneProfile:
    lane: Lane
    raw: np.ndarray         # mean signal per row across the lane
    background: np.ndarray
    corrected: np.ndarray   # raw - background, >= 0
    bands: list[Band]

    @property
    def total(self) -> float:
        return float(sum(b.volume for b in self.bands))


def lane_profile(sig: np.ndarray, lane: Lane) -> np.ndarray:
    x0 = max(lane.x0, 0)
    x1 = min(max(lane.x1, x0 + 1), sig.shape[1])
    return sig[:, x0:x1].mean(axis=1)


def subtract_background(profile: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Background under a lane profile, removing features narrower than 2*radius.

    A grey-scale opening (minimum then maximum filter over 2*radius rows)
    follows the profile everywhere except under bands; it is then smoothed
    to remove its steps and kept at or below the profile.  Unlike a rolling
    ball this does not depend on the intensity scale, so faint and strong
    bands are treated alike.
    """
    size = max(3, int(round(2 * radius)) | 1)
    pad = size
    # Smooth first so the minimum filter follows the background, not noise dips.
    padded = np.pad(gaussian_filter1d(profile, 2.0), pad, mode="reflect")
    opened = maximum_filter1d(minimum_filter1d(padded, size), size)
    bg = uniform_filter1d(opened, size)[pad:-pad]
    bg = np.minimum(bg, profile)
    return bg, profile - bg


def find_bands(lp_corr: np.ndarray, lane_no: int, width: float,
               min_prominence: float = 0.02, smooth: float = 1.0,
               min_distance: int = 3, floor: float = 0.0, edge: int = 5) -> list[Band]:
    """Peaks in a background-corrected lane profile, integrated valley-to-valley.

    ``min_prominence`` is relative to the strongest band in the lane; it is
    also at least ``floor`` (absolute, e.g. a fraction of the strongest band
    on the gel, so empty lanes don't report noise) and 4x the profile noise.
    Peaks within ``edge`` rows of either end are ignored (background edges).
    Each band is integrated out to the nearest of: the minimum separating it
    from a neighbour, the point where the signal falls below 2% of its peak,
    or 2.5 x its FWHM from the peak.
    """
    y = gaussian_filter1d(lp_corr, smooth) if smooth > 0 else lp_corr
    noise = 1.4826 * np.median(np.abs(np.diff(lp_corr))) / np.sqrt(2)
    thr = max(min_prominence * y.max(), floor, 4 * noise, 1e-9)
    peaks, props = find_peaks(y, prominence=thr, distance=min_distance)
    peaks = peaks[(peaks >= edge) & (peaks < len(y) - edge)]
    fwhm = peak_widths(y, peaks, rel_height=0.5)[0] if len(peaks) else []
    bands: list[Band] = []
    n = len(y)
    for i, p in enumerate(peaks):
        reach = max(3, int(np.ceil(2.5 * fwhm[i])))
        # Valleys between neighbouring peaks bound the integration window.
        lo_lim = peaks[i - 1] if i > 0 else 0
        hi_lim = peaks[i + 1] if i + 1 < len(peaks) else n - 1
        left = lo_lim + int(np.argmin(y[lo_lim:p + 1])) if p > lo_lim else p
        right = p + int(np.argmin(y[p:hi_lim + 1])) if hi_lim > p else p
        left, right = max(left, p - reach), min(right, p + reach)
        floor = 0.02 * y[p]
        a = p
        while a > left and y[a - 1] > floor:
            a -= 1
        b = p
        while b < right and y[b + 1] > floor:
            b += 1
        vol = float(lp_corr[a:b + 1].sum() * width)
        bands.append(Band(lane_no, i + 1, float(p), int(a), int(b + 1),
                          float(lp_corr[p]), vol))
    return bands


def analyse_lane(sig: np.ndarray, lane: Lane, radius: float,
                 min_prominence: float = 0.02, smooth: float = 1.0,
                 floor: float = 0.0) -> LaneProfile:
    raw = lane_profile(sig, lane)
    bg, corr = subtract_background(raw, radius)
    bands = find_bands(corr, lane.number, lane.width, min_prominence, smooth, floor=floor)
    return finalize(LaneProfile(lane, raw, bg, corr, bands))
