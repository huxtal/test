"""Baseline construction and peak integration of the raw thermogram."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nitc import ITCRun

_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2.0 compatibility


@dataclass
class Integration:
    baseline: np.ndarray        # uJ/s, same grid as run.time
    node_x: np.ndarray          # s
    node_y: np.ndarray          # uJ/s
    area: np.ndarray            # uJ, raw signed peak area (instrument sign)
    heat: np.ndarray            # uJ, heat absorbed by the sample (exothermic < 0)
    noise: np.ndarray           # uJ, area uncertainty estimate from baseline scatter


def auto_baseline_nodes(run: ITCRun, window: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """One node just before every injection plus one at the end of the last.

    Each node is the median heat rate over the ``window`` seconds that end at
    the node time, i.e. the equilibrated signal at the end of the previous
    injection interval.
    """
    xs = list(run.inj_start) + [run.inj_end[-1]]
    t, h = run.time, run.heat_rate
    ys = []
    for x in xs:
        mask = (t <= x) & (t > x - window)
        if mask.sum() < 3:
            mask = np.argsort(np.abs(t - x))[:5]
        ys.append(float(np.median(h[mask])))
    return np.asarray(xs, float), np.asarray(ys, float)


def integrate(run: ITCRun, baseline: str = "auto", window: float = 20.0,
              exotherm_up: bool | None = None) -> Integration:
    """Integrate every injection peak against a piecewise-linear baseline.

    ``baseline`` is ``"auto"`` (nodes from :func:`auto_baseline_nodes`) or
    ``"file"`` (the nodes saved by ITCRun in the .nitc file).
    """
    if baseline == "file" and run.baseline_nodes is not None:
        nx, ny = run.baseline_nodes
    else:
        nx, ny = auto_baseline_nodes(run, window)

    t, h = run.time, run.heat_rate
    base = np.interp(t, nx, ny)
    signal = h - base

    area = np.empty(run.n_injections)
    noise = np.empty(run.n_injections)
    for i, (t0, t1) in enumerate(zip(run.inj_start, run.inj_end)):
        m = (t >= t0) & (t <= t1)
        area[i] = _trapz(signal[m], t[m])
        # Baseline scatter at the tail of the interval -> integrated noise.
        tail = m & (t > t1 - window)
        sd = np.std(signal[tail]) if tail.sum() > 2 else 0.0
        noise[i] = sd * np.sqrt(max(t1 - t0, 1.0))

    up = run.exotherm_up if exotherm_up is None else exotherm_up
    heat = -area if up else area
    return Integration(base, nx, ny, area, heat, noise)
