"""Publication-style ITC figure: thermogram, binding isotherm and residuals."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .fit import FitResult  # noqa: E402
from .integrate import Integration  # noqa: E402
from .nitc import ITCRun  # noqa: E402

KCAL = 4.184


def summary_lines(result: FitResult, energy: str = "kJ") -> list[str]:
    f = KCAL if energy == "kcal" else 1.0
    eu = f"{energy}/mol"
    th = result.thermodynamics()
    v, e = result.values, result.errors
    lines = []
    for sfx in ("", "1", "2"):
        if f"logKa{sfx}" not in v:
            continue
        tag = f" ({sfx})" if sfx else ""
        lines += [
            f"n{tag} = {v['n' + sfx]:.3f} ± {e['n' + sfx]:.3f}",
            f"Kd{tag} = {_fmt_kd(th['Kd' + sfx + '_uM'], th['Kd' + sfx + '_err'])}",
            f"ΔH{tag} = {v['dH' + sfx] / f:.2f} ± {e['dH' + sfx] / f:.2f} {eu}",
            f"ΔG{tag} = {th['dG' + sfx + '_kJmol'] / f:.2f} {eu}",
            f"-TΔS{tag} = {th['-TdS' + sfx + '_kJmol'] / f:.2f} {eu}",
        ]
    if "dH0" in v:
        lines.append(f"offset = {v['dH0'] / f:.2f} ± {e['dH0'] / f:.2f} {eu}")
    lines.append(f"RMSD = {result.rmsd / f:.3g} {'µcal' if energy == 'kcal' else 'µJ'}")
    return lines


def _fmt_kd(kd_um: float, err_um: float) -> str:
    for scale, unit in ((1e-3, "nM"), (1.0, "µM"), (1e3, "mM")):
        if kd_um < 1000 * scale or unit == "mM":
            k, s = kd_um / scale, err_um / scale
            return f"{k:.3g} ± {s:.2g} {unit}" if np.isfinite(s) else f"{k:.3g} {unit} (fixed)"
    return ""


def plot_fit(run: ITCRun, integ: Integration, result: FitResult, path,
             energy: str = "kJ", title: str | None = None,
             exotherm_up: bool | None = None, exotherm_display: str = "down",
             blank: np.ndarray | None = None):
    """Draw thermogram, isotherm and residuals.

    ``exotherm_up`` says whether exothermic peaks are positive in the raw
    heat rate (default: the file's flag); ``exotherm_display`` ("up"/"down")
    chooses how they are drawn.  ``blank`` is the per-injection heat (uJ)
    subtracted from the sample, shown on the isotherm for reference.
    """
    f = KCAL if energy == "kcal" else 1.0
    raw_up = run.exotherm_up if exotherm_up is None else exotherm_up
    flip = 1.0 if raw_up == (exotherm_display == "up") else -1.0
    rate_unit = "µcal/s" if energy == "kcal" else "µJ/s"
    d = result.design
    nmol = d.injected_mol
    y = result.heats / nmol / f
    yfit = result.fitted / nmol / f
    res = result.residuals / nmol / f
    x = d.molar_ratio
    inc = result.included

    fig = plt.figure(figsize=(5.5, 8.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.25, 0.4], hspace=0.08)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2], sharex=ax1)

    tmin = run.time / 60.0
    g = flip / f
    ax0.plot(tmin, run.heat_rate * g, lw=0.7, color="black")
    ax0.plot(tmin, integ.baseline * g, lw=0.8, color="tab:red", alpha=0.8)
    ax0.plot(integ.node_x / 60.0, integ.node_y * g, "o", ms=2.5, color="tab:red")
    ax0.set_xlabel("Time (min)")
    ax0.set_ylabel(f"Heat rate ({rate_unit})\nexotherm {exotherm_display}")
    ax0.xaxis.set_label_position("top")
    ax0.xaxis.tick_top()
    ax0.set_title(title or run.name, fontsize=10, pad=28)

    ax1.plot(x[inc], y[inc], "s", ms=5, color="black", label="data")
    if (~inc).any():
        ax1.plot(x[~inc], y[~inc], "s", ms=5, mfc="none", color="gray", label="excluded")
    if blank is not None:
        ax1.plot(x, blank / nmol / f, "^", ms=4, color="tab:blue", alpha=0.6,
                 label="blank (subtracted)")
    ax1.plot(x, yfit, "-", color="tab:red", lw=1.3, label=f"fit ({result.model.name})")
    ax1.set_ylabel(f"ΔH ({energy}/mol of injectant)")
    # Put the parameter box in the empty corner: lower right for a rising
    # (exothermic) isotherm, upper right for a falling (endothermic) one.
    rising = yfit[-1] > yfit[0]
    ax1.legend(fontsize=8, frameon=False, loc="center right")
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.text(0.97, 0.03 if rising else 0.97, "\n".join(summary_lines(result, energy)),
             transform=ax1.transAxes, va="bottom" if rising else "top", ha="right",
             fontsize=7.5, family="DejaVu Sans",
             bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))

    ax2.axhline(0, color="0.5", lw=0.8)
    ax2.plot(x[inc], res[inc], "s", ms=3.5, color="black")
    # Scale to the fitted points; excluded injections are clipped to the edge.
    lim = 1.25 * max(np.abs(res[inc]).max(), 1e-12)
    ax2.set_ylim(-lim, lim)
    if (~inc).any():
        ax2.plot(x[~inc], np.clip(res[~inc], -lim, lim), "s", ms=3.5, mfc="none", color="gray")
    ax2.set_xlabel("Molar ratio")
    ax2.set_ylabel("Resid.")

    for ax in (ax0, ax1, ax2):
        ax.tick_params(direction="in", top=ax is not ax0, right=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
