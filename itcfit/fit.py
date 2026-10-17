"""Nonlinear least-squares fitting of integrated ITC heats."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .models import Design, Model

R = 8.314462618e-3  # kJ/(mol K)


@dataclass
class FitResult:
    model: Model
    design: Design
    heats: np.ndarray               # uJ, all injections
    included: np.ndarray            # bool mask of injections used in the fit
    values: dict[str, float]
    errors: dict[str, float]        # 1 SD (nan for fixed parameters)
    fixed: dict[str, float]
    fitted: np.ndarray              # uJ, model heats for all injections
    cov: np.ndarray
    free: list[str]
    success: bool
    message: str
    ci95: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def residuals(self) -> np.ndarray:
        return self.heats - self.fitted

    @property
    def dof(self) -> int:
        return max(int(self.included.sum()) - len(self.free), 1)

    @property
    def rmsd(self) -> float:
        r = self.residuals[self.included]
        return float(np.sqrt(np.mean(r ** 2)))

    @property
    def r_squared(self) -> float:
        y = self.heats[self.included]
        ss_res = np.sum(self.residuals[self.included] ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    def thermodynamics(self) -> dict[str, float]:
        """Derived quantities (Ka, Kd, dG, -TdS, c) for each binding site."""
        T = self.design.temperature + 273.15
        out: dict[str, float] = {"T_K": T}
        for suffix in ("", "1", "2"):
            lk = f"logKa{suffix}"
            if lk not in self.values:
                continue
            log_ka, s_lk = self.values[lk], self.errors.get(lk, np.nan)
            ka = 10 ** log_ka
            dg = -R * T * np.log(ka)
            dh = self.values[f"dH{suffix}"]
            n = self.values[f"n{suffix}"]
            out[f"Ka{suffix}_M-1"] = ka
            out[f"Ka{suffix}_err"] = ka * np.log(10) * s_lk
            out[f"Kd{suffix}_uM"] = 1e6 / ka
            out[f"Kd{suffix}_err"] = 1e6 / ka * np.log(10) * s_lk
            out[f"dG{suffix}_kJmol"] = dg
            out[f"dG{suffix}_err"] = R * T * np.log(10) * s_lk
            out[f"-TdS{suffix}_kJmol"] = dg - dh
            out[f"c{suffix}"] = n * ka * self.design.cell_conc * 1e-3
        return out


def _vec(model: Model, free: list[str], fixed: dict[str, float], x) -> dict[str, float]:
    p = dict(fixed)
    p.update(zip(free, x))
    return p


def _linear_init(model: Model, d: Design, y: np.ndarray, mask: np.ndarray,
                 nonlin: dict[str, float], fixed: dict[str, float]):
    """Solve the linear parameters exactly for given nonlinear ones."""
    lin = [q.name for q in model.params if q.linear and q.name not in fixed]
    base = {q.name: 0.0 for q in model.params if q.linear}
    base.update({k: v for k, v in fixed.items()})
    base.update(nonlin)
    offset = model.heats(base, d)
    if not lin:
        r = (y - offset)[mask]
        return {}, float(r @ r)
    cols = []
    for name in lin:
        p = dict(base)
        p[name] = 1.0
        cols.append(model.heats(p, d) - offset)
    A = np.column_stack(cols)[mask]
    b = (y - offset)[mask]
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = A @ coef - b
    return dict(zip(lin, coef)), float(r @ r)


def initial_guess(model: Model, d: Design, y: np.ndarray, mask: np.ndarray,
                  fixed: dict[str, float], n_best: int = 5) -> list[dict[str, float]]:
    """Grid search over nonlinear parameters with linear ones solved exactly."""
    scored = []
    for g in model.grid(d):
        g = {k: v for k, v in g.items() if k not in fixed}
        lin, ssr = _linear_init(model, d, y, mask, g, fixed)
        if np.isfinite(ssr):
            scored.append((ssr, {**g, **lin}))
    scored.sort(key=lambda s: s[0])
    starts = [s[1] for s in scored[:n_best]]
    return starts or [{}]


def fit(model: Model, design: Design, heats: np.ndarray,
        exclude: list[int] | None = None, fixed: dict[str, float] | None = None,
        start: dict[str, float] | None = None) -> FitResult:
    """Fit ``model`` to integrated heats (uJ).

    ``exclude`` lists 1-based injection numbers to leave out (commonly [1]).
    ``fixed`` maps parameter names to values held constant.
    """
    heats = np.asarray(heats, float)
    fixed = dict(fixed or {})
    unknown = set(fixed) - set(model.param_names)
    if unknown:
        raise ValueError(f"cannot fix unknown parameter(s) {sorted(unknown)} for model {model.name}")
    mask = np.ones(len(heats), bool)
    for i in exclude or []:
        if 1 <= i <= len(heats):
            mask[i - 1] = False

    specs = {q.name: q for q in model.params}
    free = [q.name for q in model.params if q.name not in fixed]
    lo = np.array([specs[k].lower for k in free])
    hi = np.array([specs[k].upper for k in free])

    starts = [start] if start else initial_guess(model, design, heats, mask, fixed)

    def resid(x):
        return (model.heats(_vec(model, free, fixed, x), design) - heats)[mask]

    best = None
    for s in starts:
        x0 = np.array([s.get(k, specs[k].default) for k in free], float)
        x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
        if not free:
            break
        res = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", method="trf",
                            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=5000)
        if best is None or res.cost < best.cost:
            best = res

    if best is None:   # everything fixed
        x = np.array([])
        cov = np.zeros((0, 0))
        success, message = True, "no free parameters"
    else:
        x = best.x
        J = best.jac
        dof = max(int(mask.sum()) - len(free), 1)
        s2 = 2 * best.cost / dof
        cov = np.linalg.pinv(J.T @ J) * s2
        success, message = bool(best.success), str(best.message)

    values = _vec(model, free, fixed, x)
    values = {k: float(values[k]) for k in model.param_names}
    errors = {k: float("nan") for k in model.param_names}
    for i, k in enumerate(free):
        errors[k] = float(np.sqrt(max(cov[i, i], 0.0)))

    return FitResult(model, design, heats, mask, values, errors, fixed,
                     model.heats(values, design), cov, free, success, message)


def bootstrap(result: FitResult, n: int = 200, seed: int = 0) -> dict[str, tuple[float, float]]:
    """95% confidence intervals by resampling residuals of the included points."""
    rng = np.random.default_rng(seed)
    resid = result.residuals[result.included]
    samples = {k: [] for k in result.free}
    for _ in range(n):
        y = result.fitted.copy()
        y[result.included] += rng.choice(resid, size=resid.size, replace=True)
        excl = [i + 1 for i, m in enumerate(result.included) if not m]
        r = fit(result.model, result.design, y, exclude=excl, fixed=result.fixed,
                start=result.values)
        for k in result.free:
            samples[k].append(r.values[k])
    ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
          for k, v in samples.items()}
    result.ci95 = ci
    return ci
