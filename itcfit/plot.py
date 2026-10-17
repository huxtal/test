"""Publication-style ITC figure: thermogram, binding isotherm and residuals."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from .fit import FitResult  # noqa: E402
from .integrate import Integration  # noqa: E402
from .models import smooth_isotherm  # noqa: E402
from .nitc import ITCRun  # noqa: E402

KCAL = 4.184


def _pm(v: float, e: float, fmt: str) -> str:
    return f"{v:{fmt}} (± {e:{fmt}})" if np.isfinite(e) else f"{v:{fmt}} (fixed)"


def _fmt_kd(kd_um: float, err_um: float) -> str:
    for scale, unit in ((1e-3, "nM"), (1.0, "µM"), (1e3, "mM")):
        if kd_um < 1000 * scale or unit == "mM":
            return f"{_pm(kd_um / scale, err_um / scale, '.2g')} {unit}"
    return ""


def legend_lines(result: FitResult, energy: str = "kJ") -> list[str]:
    """The short Kd / n / dH / dS block drawn on the isotherm."""
    f = KCAL if energy == "kcal" else 1.0
    s_unit = "cal/mol·K" if energy == "kcal" else "J/mol·K"
    th = result.thermodynamics()
    v, e = result.values, result.errors
    lines = []
    for sfx in ("", "1", "2"):
        if f"logKa{sfx}" not in v:
            continue
        tag = sfx and f"$_{sfx}$"
        lines += [
            f"Kd{tag} = {_fmt_kd(th['Kd' + sfx + '_uM'], th['Kd' + sfx + '_err'])}",
            f" n{tag}  = {_pm(v['n' + sfx], e['n' + sfx], '.2f')}",
            f"ΔH{tag} = {_pm(v['dH' + sfx] / f, e['dH' + sfx] / f, '.1f')} {energy}/mol",
            f"ΔS{tag} = {th['dS' + sfx + '_JmolK'] / f:.1f} {s_unit}",
        ]
    if not lines and "dH0" in v:
        lines.append(f"ΔH(dilution) = {_pm(v['dH0'] / f, e['dH0'] / f, '.2f')} {energy}/mol")
    return lines


def footer_lines(run: ITCRun, result: FitResult, energy: str, has_blank: bool,
                 blank_name: str | None, blank_mode: str | None,
                 sample_label: str, blank_label: str) -> list[str]:
    f = KCAL if energy == "kcal" else 1.0
    eu = f"{energy}/mol"
    hu = "µcal" if energy == "kcal" else "µJ"
    th = result.thermodynamics()
    v, e = result.values, result.errors
    excl = [str(i + 1) for i, m in enumerate(result.included) if not m]

    is_blank = result.model.name == "blank"
    traces = f"Green = {blank_label if is_blank else sample_label} ({run.name})"
    if has_blank:
        traces = f"Blue = {blank_label} ({blank_name}); " + traces
    lines = [traces]

    sub = []
    if has_blank:
        sub.append(f"{blank_mode} blank subtracted")
    if "dH0" in v and not is_blank and not (np.isnan(e["dH0"]) and v["dH0"] == 0):
        sub.append(f"fitted offset {_pm(v['dH0'] / f, e['dH0'] / f, '.2f')} {eu} removed from plotted points")
    if sub:
        lines.append("Isotherm: " + "; ".join(sub) + ".")

    lines.append(
        f"Model: {result.model.description}. Cell {run.cell_conc * 1e3:g} µM, "
        f"syringe {run.syringe_conc * 1e3:g} µM, V = {run.cell_volume:g} µL, "
        f"{run.temperature:g} °C. Excluded injection(s): {', '.join(excl) or 'none'}.")

    parts = []
    for sfx in ("", "1", "2"):
        if f"dG{sfx}_kJmol" in th:
            parts += [f"ΔG{sfx} = {th['dG' + sfx + '_kJmol'] / f:.2f} {eu}",
                      f"−TΔS{sfx} = {th['-TdS' + sfx + '_kJmol'] / f:.2f} {eu}",
                      f"c{sfx} = {th['c' + sfx]:.0f}"]
    parts.append(f"RMSD = {result.rmsd / f:.3g} {hu}")
    if not is_blank:
        parts.append(f"R² = {result.r_squared:.4f}")
    lines.append(", ".join(parts) + ".")
    if result.ci95:
        ci = []
        for k, (lo, hi) in result.ci95.items():
            if k.startswith("logKa"):
                ci.append(f"Kd{k[5:]} {1e9 / 10 ** hi:.3g}–{1e9 / 10 ** lo:.3g} nM")
            elif k.startswith("dH0"):
                continue
            elif k.startswith("dH"):
                ci.append(f"ΔH{k[2:]} {lo / f:.2f} to {hi / f:.2f} {eu}")
            else:
                ci.append(f"{k} {lo:.3f}–{hi:.3f}")
        lines.append("Bootstrap 95% CI: " + "; ".join(ci) + ".")
    return lines


SAMPLE_COLOR = "#008000"
BLANK_COLOR = "#0000ff"


def plot_fit(run: ITCRun, integ: Integration, result: FitResult, path,
             energy: str = "kJ", title: str | None = None,
             exotherm_up: bool | None = None, exotherm_display: str = "down",
             blank_run: ITCRun | None = None, blank_integ: Integration | None = None,
             blank_mode: str | None = None, sample_label: str = "ligand into receptor",
             blank_label: str = "ligand into buffer"):
    """Thermogram (baseline-subtracted), isotherm with smooth fit, residuals, footer.

    ``exotherm_up`` says whether exothermic peaks are positive in the raw
    heat rate (default: the file's flag); ``exotherm_display`` ("up"/"down")
    chooses how they are drawn.  A blank run, if given, is overlaid in blue.
    """
    f = KCAL if energy == "kcal" else 1.0
    rate_unit = "µcal/s" if energy == "kcal" else "µJ/s"

    def display_sign(r: ITCRun, up: bool | None) -> float:
        raw_up = r.exotherm_up if up is None else up
        return 1.0 if raw_up == (exotherm_display == "up") else -1.0

    d = result.design
    nmol = d.injected_mol
    offset = result.values.get("dH0", 0.0) * nmol if result.model.name != "blank" else 0.0
    y = (result.heats - offset) / nmol / f
    res = result.residuals / nmol / f
    x = d.molar_ratio
    inc = result.included

    with plt.rc_context({"font.size": 13, "axes.linewidth": 1.0,
                         "xtick.direction": "out", "ytick.direction": "out"}):
        fig = plt.figure(figsize=(6.5, 10.5))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.32], hspace=0.0)
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2], sharex=ax1)

        # --- thermogram, baseline subtracted --------------------------------
        traces = []
        if blank_run is not None and blank_integ is not None:
            sb = display_sign(blank_run, exotherm_up) / f
            traces.append((blank_run.time / 60,
                           (blank_run.heat_rate - blank_integ.baseline) * sb, BLANK_COLOR, 0.7))
        s0 = display_sign(run, exotherm_up) / f
        traces.append((run.time / 60, (run.heat_rate - integ.baseline) * s0, SAMPLE_COLOR, 0.9))
        for tx, ty, color, lw in traces:
            ax0.plot(tx, ty, lw=lw, color=color)
        ax0.set_xlabel("Time (min)")
        ax0.set_ylabel(f"Heat Rate ({rate_unit})")
        ax0.xaxis.set_label_position("top")
        ax0.xaxis.tick_top()
        # Sample name in whichever right-hand corner the late (small) peaks leave free.
        lo, hi = ax0.get_ylim()
        tail = np.concatenate([ty[tx > 0.7 * tx.max()] for tx, ty, _, _ in traces])
        top = (hi - tail.max()) > (tail.min() - lo)
        ax0.text(0.97, 0.95 if top else 0.05, title or run.name, transform=ax0.transAxes,
                 ha="right", va="top" if top else "bottom", fontsize=15)

        # --- isotherm --------------------------------------------------------
        ax1.axhline(0, color="0.3", lw=0.8, ls=(0, (3, 3)))
        if result.model.name != "blank":
            xs, ys = smooth_isotherm(result.model, result.values, d)
            ax1.plot(xs, ys / f, color="red", lw=1.4, zorder=2)
        else:
            ax1.plot(x, (result.fitted - offset) / nmol / f, color="red", lw=1.4)
        ax1.plot(x[inc], y[inc], "o", ms=4.5, color=SAMPLE_COLOR, zorder=3)
        if (~inc).any():
            ax1.plot(x[~inc], y[~inc], "o", ms=4.5, mfc="none", color=SAMPLE_COLOR, zorder=3)
        ax1.set_ylabel(f"Enthalpy ({energy}/mole)")
        ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda val, _: f"{val:.1f}".replace("-", "\u2212")))
        plt.setp(ax1.get_xticklabels(), visible=False)
        rising = y[inc][-1] > y[inc][0] if inc.sum() > 1 else True
        ax1.text(0.97, 0.05 if rising else 0.95, "\n".join(legend_lines(result, energy)),
                 transform=ax1.transAxes, ha="right", va="bottom" if rising else "top",
                 multialignment="left", fontsize=11.5, linespacing=1.35,
                 bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.8))

        # --- residuals ---------------------------------------------------------
        ax2.axhline(0, color="0.5", lw=0.8)
        ax2.plot(x[inc], res[inc], "s", ms=3.5, color="black")
        lim = 1.25 * max(np.abs(res[inc]).max(), 1e-12)
        ax2.set_ylim(-lim, lim)
        if (~inc).any():
            ax2.plot(x[~inc], np.clip(res[~inc], -lim, lim), "s", ms=3.5, mfc="none",
                     color="gray")
        ax2.set_xlabel("Mole Ratio")
        ax2.set_ylabel("Resid.")
        for ax in (ax0, ax1, ax2):
            ax.tick_params(right=False)
        fig.align_ylabels([ax0, ax1, ax2])

        # --- footer --------------------------------------------------------------
        import textwrap
        foot = footer_lines(run, result, energy, blank_run is not None,
                            blank_run.name if blank_run is not None else None, blank_mode,
                            sample_label, blank_label)
        wrapped = "\n".join(textwrap.fill(line, 100) for line in foot)
        box = ax2.get_position()
        fig.text(0.02, box.y0 - 0.065, wrapped, ha="left", va="top", fontsize=8.5,
                 color="0.15", linespacing=1.4)

        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)
