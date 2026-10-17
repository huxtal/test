"""Binding models for incremental ITC heats.

Units used throughout: volumes in uL, concentrations in mM, enthalpies in
kJ/mol and heats in uJ (uL * mM * kJ/mol = uJ, so no conversion factors).

Cell/syringe concentrations after each injection follow the MicroCal/Origin
overflow-cell treatment, and the incremental heat includes the displaced-volume
correction:

    dQ_i = Q_i + (v_i / V0) * (Q_i + Q_{i-1}) / 2 - Q_{i-1} + dH0 * v_i * X0

where Q_i is the total heat content of the cell after injection i and dH0 is a
constant heat of dilution per mol of injectant.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass
class Design:
    """Experimental design of one titration."""
    volumes: np.ndarray     # uL injected per injection
    cell_volume: float      # uL
    cell_conc: float        # mM, macromolecule ("M") in the cell
    syringe_conc: float     # mM, ligand ("X") in the syringe
    temperature: float = 25.0   # C

    def __post_init__(self):
        self.volumes = np.asarray(self.volumes, float)
        dv = np.cumsum(self.volumes) / self.cell_volume
        self.Mt = self.cell_conc * (1 - dv / 2) / (1 + dv / 2)
        self.Xt = self.syringe_conc * dv * (1 - dv / 2)

    @property
    def molar_ratio(self) -> np.ndarray:
        return self.Xt / self.Mt

    @property
    def injected_mol(self) -> np.ndarray:
        """nmol of ligand delivered by each injection (uL * mM)."""
        return self.volumes * self.syringe_conc

    def incremental(self, Q: np.ndarray, dH0: float) -> np.ndarray:
        Qprev = np.concatenate([[0.0], Q[:-1]])
        r = self.volumes / self.cell_volume
        return Q + r * (Q + Qprev) / 2 - Qprev + dH0 * self.injected_mol


@dataclass
class Param:
    name: str
    default: float
    lower: float
    upper: float
    linear: bool = False    # model heat is linear in this parameter
    unit: str = ""


class Model:
    name = ""
    description = ""
    params: list[Param] = []

    def heats(self, p: dict[str, float], d: Design) -> np.ndarray:
        raise NotImplementedError

    def grid(self, d: Design):
        """Starting values for the nonlinear parameters (dicts)."""
        yield {}

    @property
    def param_names(self) -> list[str]:
        return [q.name for q in self.params]


def _ka_to_kd_mM(log_ka: float) -> float:
    return 1e3 / 10.0 ** log_ka


class Blank(Model):
    name = "blank"
    description = "Constant heat of dilution (control titration)"
    params = [Param("dH0", 0.0, -1e4, 1e4, linear=True, unit="kJ/mol")]

    def heats(self, p, d):
        return d.incremental(np.zeros_like(d.Mt), p["dH0"])


class OneSite(Model):
    name = "onesite"
    description = "One set of n identical, independent sites"
    params = [
        Param("logKa", 6.0, 0.0, 14.0, unit="log10(M^-1)"),
        Param("dH", -20.0, -1e4, 1e4, linear=True, unit="kJ/mol"),
        Param("n", 1.0, 0.01, 20.0),
        Param("dH0", 0.0, -1e4, 1e4, linear=True, unit="kJ/mol"),
    ]

    @staticmethod
    def bound(log_ka: float, n: float, Mt: np.ndarray, Xt: np.ndarray) -> np.ndarray:
        """Concentration of occupied sites (mM) from the exact quadratic."""
        kd = _ka_to_kd_mM(log_ka)
        S = n * Mt
        b = S + Xt + kd
        disc = np.maximum(b * b - 4 * S * Xt, 0.0)
        # Numerically stable root: 2c / (b + sqrt(disc)).
        return 2 * S * Xt / (b + np.sqrt(disc))

    def heats(self, p, d):
        Q = d.cell_volume * p["dH"] * self.bound(p["logKa"], p["n"], d.Mt, d.Xt)
        return d.incremental(Q, p["dH0"])

    def grid(self, d):
        for lk, n in itertools.product(np.arange(2.0, 11.01, 0.25),
                                       np.arange(0.2, 3.01, 0.1)):
            yield {"logKa": lk, "n": n}


def _free_ligand_two_sites(Xt, Mt, k1, k2, n1, n2, iters=80):
    """Solve Xt = X + Mt*(n1 k1 X/(1+k1 X) + n2 k2 X/(1+k2 X)) by bisection."""
    lo = np.zeros_like(Xt)
    hi = Xt.copy()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = mid + Mt * (n1 * k1 * mid / (1 + k1 * mid) + n2 * k2 * mid / (1 + k2 * mid)) - Xt
        lo = np.where(f < 0, mid, lo)
        hi = np.where(f < 0, hi, mid)
    return 0.5 * (lo + hi)


class TwoSites(Model):
    name = "twosites"
    description = "Two independent sets of sites (n1, K1, dH1; n2, K2, dH2)"
    params = [
        Param("logKa1", 7.0, 0.0, 14.0, unit="log10(M^-1)"),
        Param("dH1", -20.0, -1e4, 1e4, linear=True, unit="kJ/mol"),
        Param("n1", 1.0, 0.01, 20.0),
        Param("logKa2", 5.0, 0.0, 14.0, unit="log10(M^-1)"),
        Param("dH2", -20.0, -1e4, 1e4, linear=True, unit="kJ/mol"),
        Param("n2", 1.0, 0.01, 20.0),
        Param("dH0", 0.0, -1e4, 1e4, linear=True, unit="kJ/mol"),
    ]

    def heats(self, p, d):
        k1 = 1.0 / _ka_to_kd_mM(p["logKa1"])
        k2 = 1.0 / _ka_to_kd_mM(p["logKa2"])
        X = _free_ligand_two_sites(d.Xt, d.Mt, k1, k2, p["n1"], p["n2"])
        th1 = k1 * X / (1 + k1 * X)
        th2 = k2 * X / (1 + k2 * X)
        Q = d.cell_volume * d.Mt * (p["n1"] * p["dH1"] * th1 + p["n2"] * p["dH2"] * th2)
        return d.incremental(Q, p["dH0"])

    def grid(self, d):
        for lk1 in np.arange(4.0, 10.01, 0.5):
            for lk2 in np.arange(2.0, lk1 - 0.99, 0.5):
                for n1, n2 in itertools.product([0.3, 0.5, 1.0, 1.5], repeat=2):
                    yield {"logKa1": lk1, "n1": n1, "logKa2": lk2, "n2": n2}


MODELS: dict[str, type[Model]] = {m.name: m for m in (OneSite, TwoSites, Blank)}


def get_model(name: str) -> Model:
    try:
        return MODELS[name.lower()]()
    except KeyError:
        raise ValueError(f"unknown model {name!r}; choose from {', '.join(MODELS)}") from None


def smooth_isotherm(model: Model, p: dict[str, float], d: Design,
                    n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Continuous fit curve: molar ratio vs. heat per mol injectant (kJ/mol).

    Evaluates the same finite-injection model at ``n`` cumulative volumes,
    each for an injection of the median volume, so the curve passes through
    the fitted values of equal-volume injections.  The dilution offset is left
    out (dH0 = 0).
    """
    v = float(np.median(d.volumes))
    total = float(np.sum(d.volumes))
    q = dict(p)
    if "dH0" in q:
        q["dH0"] = 0.0
    xs, ys = [], []
    for vc in np.linspace(v, total, n):
        sub = Design(np.array([vc - v, v]) if vc > v else np.array([v]),
                     d.cell_volume, d.cell_conc, d.syringe_conc, d.temperature)
        dq = model.heats(q, sub)[-1]
        xs.append(sub.molar_ratio[-1])
        ys.append(dq / (v * d.syringe_conc))
    return np.asarray(xs), np.asarray(ys)
