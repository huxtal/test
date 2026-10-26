"""Gel analysis pipeline: lanes -> bands -> MW -> purity / target / standards."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bands import Band, LaneProfile, analyse_lane
from .image import GelImage
from .ladder import Calibration, calibrate
from .lanes import Lane, detect_lanes


@dataclass
class StandardCurve:
    lanes: list[int]
    amounts: np.ndarray     # ug loaded
    volumes: np.ndarray     # band volume
    slope: float
    intercept: float
    r_squared: float

    def amount(self, volume):
        return (np.asarray(volume, float) - self.intercept) / self.slope


@dataclass
class GelResult:
    gel: GelImage
    lanes: list[Lane]
    profiles: list[LaneProfile]
    ladder_lanes: list[int] = field(default_factory=list)
    ladder_cals: dict[int, Calibration] = field(default_factory=dict)   # per ladder lane
    lane_cals: dict[int, Calibration] = field(default_factory=dict)     # per lane (interpolated)
    target_mw: float | None = None
    targets: dict[int, Band | None] = field(default_factory=dict)
    standard: StandardCurve | None = None
    warnings: list[str] = field(default_factory=list)

    def low_signal(self, lane_no: int, frac: float = 0.05) -> bool:
        """Lane whose strongest band is under ``frac`` of the strongest sample band
        on the gel (or that has no bands): its percentages are not meaningful."""
        top = max((b.height for p in self.sample_profiles for b in p.bands), default=0.0)
        own = max((b.height for b in self.profile(lane_no).bands), default=0.0)
        return top > 0 and own < frac * top

    def profile(self, lane_no: int) -> LaneProfile:
        return self.profiles[lane_no - 1]

    @property
    def calibrated(self) -> bool:
        return bool(self.lane_cals)

    @property
    def sample_profiles(self) -> list[LaneProfile]:
        return [p for p in self.profiles if p.lane.number not in self.ladder_lanes]


def _closest_band(bands: list[Band], mw: float, tol: float) -> Band | None:
    """Band nearest (in log MW) to ``mw`` within a relative tolerance."""
    cands = [b for b in bands if np.isfinite(b.mw) and abs(np.log(b.mw / mw)) <= np.log1p(tol)]
    if not cands:
        return None
    return min(cands, key=lambda b: (abs(np.log(b.mw / mw)), -b.volume))


def lane_calibration(res: GelResult, lane: Lane) -> Calibration:
    """Calibration for one lane.

    With one ladder it is used as is.  With several, each ladder size's row
    is interpolated linearly in x between the ladder lanes (held constant
    beyond the outermost ladders), which corrects for a tilted or smiling gel.
    """
    cals = [(res.profile(ll).lane.center, res.ladder_cals[ll]) for ll in res.ladder_lanes]
    if len(cals) == 1:
        return cals[0][1]
    common = sorted(set.intersection(*(set(c.mw.tolist()) for _, c in cals)), reverse=True)
    if len(common) < 2:
        return min(cals, key=lambda c: abs(c[0] - lane.center))[1]
    xs = np.array([x for x, _ in cals])
    ys = np.array([[c.y[list(c.mw).index(m)] for m in common] for _, c in cals])
    y = np.array([np.interp(lane.center, xs, ys[:, k]) for k in range(len(common))])
    return Calibration(y, np.array(common, float), 0, True)


def analyse_gel(gel: GelImage, n_lanes: int | None = None,
                lane_centers: list[float] | None = None, lane_width_frac: float = 0.6,
                ball_radius: float | None = None, min_prominence: float = 0.02,
                gel_floor: float = 0.03,
                ladder_lanes: list[int] | None = None, ladder_sizes: list[float] | None = None,
                ladder_reference: list[float] | None = None,
                target_mw: float | None = None, target_tol: float = 0.1,
                standards: dict[int, float] | None = None,
                standard_mw: float | None = None) -> GelResult:
    sig = gel.signal
    lanes = detect_lanes(sig, n_lanes, lane_centers, lane_width_frac)
    radius = ball_radius or max(10.0, sig.shape[0] / 20)
    # First pass finds the strongest band on the gel; the second requires
    # every band to reach ``gel_floor`` of it, so empty lanes stay empty.
    first = [analyse_lane(sig, ln, radius, min_prominence) for ln in lanes]
    gel_max = max((b.height for p in first for b in p.bands), default=0.0)
    profiles = [analyse_lane(sig, ln, radius, min_prominence, floor=gel_floor * gel_max)
                for ln in lanes]
    res = GelResult(gel, lanes, profiles, warnings=list(gel.warnings))

    if ladder_lanes:
        if not ladder_sizes:
            raise ValueError("ladder lane given without ladder sizes")
        for ll in ladder_lanes:
            if not 1 <= ll <= len(lanes):
                raise ValueError(f"ladder lane {ll} out of range 1..{len(lanes)}")
            cal = calibrate(res.profile(ll).bands, ladder_sizes, reference=ladder_reference)
            res.ladder_cals[ll] = cal
            if cal.unmatched:
                res.warnings.append(f"ladder lane {ll}: {cal.unmatched} size(s) not found; "
                                    "check the annotated image")
            if not cal.reference_ok:
                res.warnings.append(f"ladder lane {ll}: the intensified reference bands did "
                                    "not land on the strongest peaks; check the lane and "
                                    "--ladder type")
        res.ladder_lanes = sorted(ladder_lanes)
        for ln in lanes:
            res.lane_cals[ln.number] = lane_calibration(res, ln)
        for p in profiles:
            cal = res.lane_cals[p.lane.number]
            for b in p.bands:
                b.mw = float(cal.predict(b.y))
        # Ladder bands get their nominal sizes where matched.
        for ll, cal in res.ladder_cals.items():
            for b in res.profile(ll).bands:
                i = int(np.argmin(np.abs(cal.y - b.y)))
                if abs(cal.y[i] - b.y) < 1:
                    b.mw = float(cal.mw[i])

    if target_mw is not None:
        if not res.calibrated:
            raise ValueError("--target-mw needs a ladder lane for the MW calibration")
        res.target_mw = target_mw
        for p in res.sample_profiles:
            if p.lane.number not in (standards or {}):
                res.targets[p.lane.number] = _closest_band(p.bands, target_mw, target_tol)

    if standards:
        smw = standard_mw or target_mw
        vols, amts, used = [], [], []
        for lane_no, amt in sorted(standards.items()):
            bands = res.profile(lane_no).bands
            if not bands:
                continue
            if smw is not None and res.calibrated:
                b = _closest_band(bands, smw, max(target_tol, 0.15))
            else:
                b = max(bands, key=lambda q: q.volume)
            if b is None:
                res.warnings.append(f"standard lane {lane_no}: no band near {smw} kDa")
                continue
            vols.append(b.volume)
            amts.append(amt)
            used.append(lane_no)
        if len(vols) >= 2:
            amts_a, vols_a = np.array(amts), np.array(vols)
            slope, intercept = np.polyfit(amts_a, vols_a, 1)
            pred = slope * amts_a + intercept
            ss_tot = np.sum((vols_a - vols_a.mean()) ** 2)
            r2 = float(1 - np.sum((vols_a - pred) ** 2) / ss_tot) if ss_tot > 0 else float("nan")
            res.standard = StandardCurve(used, amts_a, vols_a, float(slope),
                                         float(intercept), r2)
            if r2 < 0.98:
                res.warnings.append(f"standard curve R² = {r2:.3f}: check for saturation or "
                                    "loading errors")
            out = [ln for ln, b in res.targets.items() if b is not None and ln not in standards
                   and not (0.9 * vols_a.min() <= b.volume <= 1.1 * vols_a.max())]
            if out:
                res.warnings.append("target outside the standard range (amount extrapolated) "
                                    f"in lane(s) {', '.join(map(str, out))}")
        else:
            res.warnings.append("standard curve needs at least 2 standard lanes with bands")
    return res
