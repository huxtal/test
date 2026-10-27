"""Figures: annotated gel, lane profiles, standard curve."""

from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from .quant import GelResult  # noqa: E402

BAND_COLOR = "#d62728"
TARGET_COLOR = "#1f77b4"


def _label(res: GelResult, lane_no: int, names: list[str] | None) -> str:
    if names and lane_no <= len(names) and names[lane_no - 1]:
        return names[lane_no - 1]
    return "M" if lane_no in res.ladder_lanes else str(lane_no)


def _footer(fig, lines: list[str], y: float):
    text = "\n".join(textwrap.fill(s, 125) for s in lines)
    fig.text(0.01, y, text, ha="left", va="top", fontsize=7.5, color="0.2", linespacing=1.35)


def plot_gel(res: GelResult, path, lane_names: list[str] | None = None,
             label_min_percent: float = 5.0):
    """The analysed (cropped) gel with lanes, bands and sizes marked.

    Bands holding at least ``label_min_percent`` of their lane (and every
    target band) are labelled with MW and % of lane.
    """
    gel = res.gel
    img = gel.raw
    h, w = img.shape
    ox, oy = (gel.crop[0], gel.crop[1]) if gel.crop else (0, 0)
    fig_w = 8.0
    fig_h = fig_w * h / w + 0.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    lo, hi = np.percentile(img, [0.5, 99.5])
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, extent=(ox, ox + w, oy + h, oy),
              interpolation="lanczos")

    targets = {b.lane: b for b in res.targets.values() if b is not None}
    for p in res.profiles:
        ln = p.lane
        is_ladder = ln.number in res.ladder_lanes
        faint = not is_ladder and res.low_signal(ln.number)
        color = "0.55" if faint else BAND_COLOR
        ax.add_patch(plt.Rectangle((ox + ln.x0, oy), ln.width, h, fill=False,
                                   ec="#ff9f1c" if is_ladder else "#2ca02c", lw=0.6,
                                   ls="--" if is_ladder else "-", alpha=0.8))
        ax.text(ox + ln.center, oy - 0.01 * h, _label(res, ln.number, lane_names),
                ha="center", va="bottom", fontsize=8, fontweight="bold")
        for b in p.bands:
            is_t = targets.get(ln.number) is b
            ax.plot([ox + ln.x0, ox + ln.x1], [oy + b.y] * 2,
                    color=TARGET_COLOR if is_t else color, lw=1.0 if is_t else 0.6)
            if is_ladder:
                if ln.number in res.ladder_cals:
                    cal = res.ladder_cals[ln.number]
                    k = np.argmin(np.abs(cal.y - b.y))
                    if abs(cal.y[k] - b.y) < 1:
                        ax.text(ox + ln.x0 - 2, oy + b.y, f"{cal.mw[k]:g}", ha="right",
                                va="center", fontsize=5.5, color="#ff7f0e")
            elif faint:
                continue
            elif (b.percent >= label_min_percent or is_t) and np.isfinite(b.mw):
                ax.text(ox + ln.x1 + 1, oy + b.y, f"{b.mw:.0f}\n{b.percent:.0f}%",
                        ha="left", va="center", fontsize=4.8, linespacing=0.9,
                        color=TARGET_COLOR if is_t else BAND_COLOR)
            elif b.percent >= label_min_percent:
                ax.text(ox + ln.x1 + 1, oy + b.y, f"{b.percent:.0f}%", ha="left",
                        va="center", fontsize=4.8, color=BAND_COLOR)
        if faint:
            ax.text(ox + ln.center, oy + 0.02 * h, "low\nsignal", ha="center", va="top",
                    fontsize=5, color="0.4", linespacing=0.9)
    ax.set_xlim(ox, ox + w)
    ax.set_ylim(oy + h, oy - 0.04 * h)
    ax.tick_params(labelsize=6)
    ax.set_title(gel.name, fontsize=10, pad=4)

    foot = [f"Channel analysed: {gel.channel}; {'inverted (dark bands)' if gel.inverted else 'bright bands'}"
            + (f"; crop x {gel.crop[0]}–{gel.crop[2]}, y {gel.crop[1]}–{gel.crop[3]} px" if gel.crop else "")
            + (f"; rotated {gel.angle:+.2f}°" if gel.angle else "") + ". "
            "Labels: MW (kDa) / % of lane; blue = target band; orange = ladder sizes; "
            "grey = low-signal lane (strongest band < 5% of the gel's strongest; % not meaningful)."]
    foot += [f"! {w_}" for w_ in res.warnings]
    _footer(fig, foot, 0.0)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def plot_profiles(res: GelResult, path, lane_names: list[str] | None = None):
    n = len(res.profiles)
    ncol = min(5, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.2 * nrow), squeeze=False,
                             sharey=False)
    oy = res.gel.crop[1] if res.gel.crop else 0
    targets = {b.lane: b for b in res.targets.values() if b is not None}
    for k, ax in enumerate(axes.flat):
        if k >= n:
            ax.axis("off")
            continue
        p = res.profiles[k]
        rows = np.arange(len(p.raw)) + oy
        ax.plot(rows, p.raw, color="0.65", lw=0.6)
        ax.plot(rows, p.background, color="0.4", lw=0.6, ls="--")
        ax.plot(rows, p.corrected, color="black", lw=0.8)
        for b in p.bands:
            is_t = targets.get(p.lane.number) is b
            sl = slice(b.y0, b.y1)
            ax.fill_between(rows[sl], 0, p.corrected[sl], alpha=0.45, lw=0,
                            color=TARGET_COLOR if is_t else BAND_COLOR)
            if b.percent >= 5 or is_t:
                txt = f"{b.mw:.0f}" if np.isfinite(b.mw) else ""
                ax.text(oy + b.y, p.corrected[int(b.y)], f"{txt}\n{b.percent:.0f}%",
                        ha="center", va="bottom", fontsize=5, linespacing=0.9)
        ax.set_title(f"lane {p.lane.number}" + (f" ({_label(res, p.lane.number, lane_names)})"
                     if lane_names or p.lane.number in res.ladder_lanes else ""), fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_xlim(rows[0], rows[-1])
        top = max(p.raw.max(), 1e-6)
        ax.set_ylim(-0.03 * top, 1.18 * top)
    fig.supxlabel("row (px, top → bottom of gel)", fontsize=8)
    fig.supylabel("signal (a.u.)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_standard(res: GelResult, path, unit: str = "µg"):
    sc = res.standard
    fig, ax = plt.subplots(figsize=(4, 3.2))
    ax.plot(sc.amounts, sc.volumes, "o", color="black", label="standards")
    amts = [float(sc.amount(b.volume)) for b in res.targets.values() if b is not None]
    xs = np.linspace(0, max([sc.amounts.max(), *amts, 1e-9]) * 1.1, 50)
    ax.plot(xs, sc.slope * xs + sc.intercept, color="tab:red", lw=1,
            label=f"fit, R² = {sc.r_squared:.3f}")
    for lane_no, b in res.targets.items():
        if b is not None:
            amt = float(sc.amount(b.volume))
            ax.plot(amt, b.volume, "s", ms=4, mfc="none", color=TARGET_COLOR)
            ax.annotate(str(lane_no), (amt, b.volume), fontsize=6, xytext=(3, 2),
                        textcoords="offset points", color=TARGET_COLOR)
    ax.set_xlabel(f"amount loaded ({unit})")
    ax.set_ylabel("band volume (a.u.)")
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_grid(gel, path):
    """Whole image with a pixel grid, for choosing --crop and lane positions."""
    img = gel.raw
    h, w = img.shape
    fig, ax = plt.subplots(figsize=(10, 10 * h / w))
    lo, hi = np.percentile(img, [0.5, 99.5])
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
    step = 50 if max(h, w) < 2000 else 100
    ax.set_xticks(np.arange(0, w, step))
    ax.set_yticks(np.arange(0, h, step))
    ax.tick_params(labelsize=6)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.grid(color="red", lw=0.3, alpha=0.6)
    ax.set_title(f"{gel.name}: choose --crop x0,y0,x1,y1 to keep only the resolving gel",
                 fontsize=9)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_ladder_mass(res: GelResult, path):
    """Per-band ladder response (volume per ug) vs size: shows how far apart
    proteins stain, i.e. how reliable ladder-based amounts are."""
    lm = res.ladder_mass
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    markers = "osD^v"
    for k, ll in enumerate(sorted({p[0] for p in lm.points})):
        pts = [p for p in lm.points if p[0] == ll]
        ax.plot([p[1] for p in pts], [p[3] / p[2] for p in pts], markers[k % 5], ms=5,
                mfc="none", label=f"ladder lane {ll}")
    ax.axhline(lm.factor, color="tab:red", lw=1, label="geometric mean")
    ax.axhspan(lm.factor / lm.gsd, lm.factor * lm.gsd, color="tab:red", alpha=0.12,
               label=f"x/÷{lm.gsd:.1f} (1 geom. SD)")
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 30, 50, 100, 200])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("ladder band (kDa)")
    ax.set_ylabel("response (volume per µg)")
    ax.set_title(f"ladder response: CV {lm.cv_percent:.0f}%", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
