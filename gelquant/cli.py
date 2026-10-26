"""Command-line SDS-PAGE band quantification."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from .image import apply_rotation, deskew_angle, load_gel
from .ladder import LADDERS, REFERENCE_BANDS
from .plot import plot_gel, plot_grid, plot_profiles, plot_standard
from .quant import analyse_gel

IMAGE_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def _ints(text: str | None) -> list[int]:
    return [int(v) for v in text.split(",") if v.strip()] if text else []


def _floats(text: str | None) -> list[float]:
    return [float(v) for v in text.split(",") if v.strip()] if text else []


def _standards(text: str | None) -> dict[int, float]:
    out = {}
    for item in (text or "").split(","):
        if item.strip():
            lane, _, amt = item.partition("=")
            if not amt:
                raise SystemExit(f"--standards expects LANE=AMOUNT pairs, got {item!r}")
            out[int(lane)] = float(amt)
    return out


def _collect(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files += sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXT)
        elif p.exists():
            files.append(p)
        else:
            files += sorted(Path().glob(item))
    return files


def process(path: Path, args, outdir: Path) -> list[dict]:
    crop = tuple(_ints(args.crop)) if args.crop else None
    if crop is not None and len(crop) != 4:
        raise SystemExit("--crop expects x0,y0,x1,y1")
    invert = {"auto": None, "yes": True, "no": False}[args.invert]
    if args.grid:
        gel = load_gel(path, invert=invert, channel=args.channel)
        out = outdir / f"{gel.name}_grid.png"
        plot_grid(gel, out)
        print(f"[GRID] {out}")
        return []

    gel = load_gel(path, invert=invert, channel=args.channel, crop=crop)
    if args.rotate:
        apply_rotation(gel, args.rotate)
    elif args.deskew:
        apply_rotation(gel, deskew_angle(gel.signal))

    sizes = _floats(args.ladder_mw) or LADDERS[args.ladder]
    reference = [] if args.ladder_mw else REFERENCE_BANDS.get(args.ladder, [])
    ladder_lanes = _ints(args.ladder_lane)
    centers = _floats(args.lane_centers)
    if centers and crop:
        centers = [c - crop[0] for c in centers]
    standards = _standards(args.standards)
    res = analyse_gel(
        gel, n_lanes=args.lanes, lane_centers=centers or None,
        lane_width_frac=args.lane_width, ball_radius=args.ball,
        min_prominence=args.min_prominence, gel_floor=args.floor,
        ladder_lanes=ladder_lanes, ladder_sizes=sizes, ladder_reference=reference,
        target_mw=args.target_mw, target_tol=args.target_tol,
        standards=standards, standard_mw=args.standard_mw)
    names = [s.strip() for s in args.lane_names.split(",")] if args.lane_names else None

    stem = outdir / gel.name
    ox, oy = (crop[0], crop[1]) if crop else (0, 0)
    ref_lane = args.reference_lane
    ref_band = res.targets.get(ref_lane) if ref_lane else None

    with open(f"{stem}_bands.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lane", "lane_name", "band", "y_px", "mw_kDa", "mw_extrapolated",
                    "peak_height", "volume", "percent_of_lane", "is_target"]
                   + (["amount_ug"] if res.standard else []))
        for p in res.profiles:
            ln = p.lane.number
            cal = res.lane_cals.get(ln)
            for b in p.bands:
                row = [ln, names[ln - 1] if names and ln <= len(names) else "",
                       b.index, round(b.y + oy, 1),
                       f"{b.mw:.4g}" if np.isfinite(b.mw) else "",
                       int(bool(cal.extrapolated(b.y))) if cal is not None else "",
                       f"{b.height:.5g}", f"{b.volume:.5g}", f"{b.percent:.3g}",
                       int(res.targets.get(ln) is b)]
                if res.standard:
                    row.append(f"{float(res.standard.amount(b.volume)):.4g}")
                w.writerow(row)

    lane_rows = []
    for p in res.profiles:
        ln = p.lane.number
        main = max(p.bands, key=lambda b: b.volume, default=None)
        t = res.targets.get(ln)
        row = {
            "file": gel.name, "lane": ln,
            "lane_name": names[ln - 1] if names and ln <= len(names) else "",
            "role": "ladder" if ln in res.ladder_lanes else
                    "standard" if ln in standards else "sample",
            "center_x_px": round(p.lane.center + ox, 1),
            "n_bands": len(p.bands),
            "low_signal": int(ln not in res.ladder_lanes and res.low_signal(ln)),
            "total_volume": f"{p.total:.5g}",
            "main_band_mw_kDa": f"{main.mw:.4g}" if main and np.isfinite(main.mw) else "",
            "main_band_percent": f"{main.percent:.3g}" if main else "",
        }
        if res.target_mw is not None:
            row.update({
                "target_mw_kDa": f"{t.mw:.4g}" if t else "",
                "target_volume": f"{t.volume:.5g}" if t else "",
                "target_percent_of_lane": f"{t.percent:.3g}" if t else "",
            })
            if ref_band is not None:
                row["target_relative_to_lane_%d" % ref_lane] = \
                    f"{t.volume / ref_band.volume:.3g}" if t else ""
            if res.standard is not None:
                row["target_amount_ug"] = f"{float(res.standard.amount(t.volume)):.4g}" if t else ""
        lane_rows.append(row)
    keys = list(dict.fromkeys(k for r in lane_rows for k in r))
    with open(f"{stem}_lanes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(lane_rows)

    plot_gel(res, f"{stem}_gel.{args.format}", names, args.label_min)
    plot_profiles(res, f"{stem}_profiles.{args.format}", names)
    if res.standard:
        plot_standard(res, f"{stem}_standard.{args.format}")

    report = {
        "file": str(path), "channel": gel.channel, "inverted": gel.inverted,
        "crop": crop, "rotation_deg": gel.angle,
        "lanes": [round(ln.center + ox, 1) for ln in res.lanes],
        "lane_width_px": round(res.lanes[0].width, 1),
        "ladder": args.ladder if not args.ladder_mw else "custom",
        "ladder_lanes": res.ladder_lanes,
        "ladder_bands": {ll: [[round(float(y) + oy, 1), float(m)] for y, m in zip(c.y, c.mw)]
                         for ll, c in res.ladder_cals.items()},
        "target_mw": res.target_mw,
        "standard_curve": None if not res.standard else {
            "lanes": res.standard.lanes, "amounts_ug": res.standard.amounts.tolist(),
            "volumes": res.standard.volumes.tolist(), "slope": res.standard.slope,
            "intercept": res.standard.intercept, "r_squared": res.standard.r_squared},
        "warnings": res.warnings,
    }
    with open(f"{stem}_report.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"[ OK ] {path.name}: {len(res.lanes)} lanes, "
          f"{sum(len(p.bands) for p in res.sample_profiles)} sample bands"
          + (f", ladder lane(s) {','.join(map(str, res.ladder_lanes))}" if res.ladder_lanes else ""))
    for p in res.sample_profiles:
        main = max(p.bands, key=lambda b: b.volume, default=None)
        if main is None:
            continue
        if p.lane.number in standards:
            print(f"       lane {p.lane.number:>2}: standard, {standards[p.lane.number]:g} ug "
                  f"({main.mw:.1f} kDa band)")
            continue
        if res.low_signal(p.lane.number):
            print(f"       lane {p.lane.number:>2}: low signal (no quantifiable bands)")
            continue
        t = res.targets.get(p.lane.number)
        msg = (f"       lane {p.lane.number:>2}: main band {main.mw:6.1f} kDa "
               f"({main.percent:4.1f}% of lane)")
        if res.target_mw is not None:
            msg += f"; target {t.mw:.1f} kDa {t.percent:.1f}%" if t else "; target not found"
            if t is not None and res.standard is not None:
                msg += f" = {float(res.standard.amount(t.volume)):.3g} ug"
        print(msg)
    for w_ in res.warnings:
        print(f"       ! {w_}")
    return lane_rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="quant_gel",
        description="Detect lanes and bands in SDS-PAGE images and quantify them.")
    p.add_argument("inputs", nargs="+", help="gel images (TIFF/PNG/JPG), folders or globs")
    p.add_argument("--grid", action="store_true",
                   help="only write the image with a pixel grid (to choose --crop) and stop")
    p.add_argument("--crop", metavar="X0,Y0,X1,Y1",
                   help="analyse only this box (exclude wells, stacking gel, dye front, edges)")
    p.add_argument("--channel", default="auto", choices=["auto", "r", "g", "b", "gray"],
                   help="colour images: plane to analyse (auto = widest range; red for Coomassie)")
    p.add_argument("--invert", default="auto", choices=["auto", "yes", "no"],
                   help="dark bands on light background? (auto)")
    p.add_argument("--rotate", type=float, default=0.0, help="rotate by DEG before analysis")
    p.add_argument("--deskew", action="store_true", help="automatically straighten the lanes")
    p.add_argument("-n", "--lanes", type=int, help="number of lanes (wells), incl. empty ones")
    p.add_argument("--lane-centers", metavar="X,...", help="lane centres in image pixels")
    p.add_argument("--lane-width", type=float, default=0.6,
                   help="fraction of the lane spacing used for the profile (default 0.6)")
    p.add_argument("--lane-names", metavar="A,B,...", help="names for lanes, left to right")
    p.add_argument("--ladder", default="benchmark", choices=sorted(LADDERS),
                   help="ladder preset (default: benchmark = Invitrogen 10747-012)")
    p.add_argument("--ladder-mw", metavar="KDA,...", help="custom ladder sizes (kDa)")
    p.add_argument("--ladder-lane", default="1", metavar="N[,M]",
                   help="ladder lane(s); two or more correct for tilt/smile (default 1; "
                        "'0' or '' for none)")
    p.add_argument("--target-mw", type=float, help="protein of interest (kDa)")
    p.add_argument("--target-tol", type=float, default=0.1,
                   help="relative MW window for the target (default 0.1 = ±10%%)")
    p.add_argument("--reference-lane", type=int,
                   help="report the target band relative to this lane")
    p.add_argument("--standards", metavar="LANE=UG,...",
                   help="standard lanes and amounts loaded, e.g. 2=0.25,3=0.5,4=1")
    p.add_argument("--standard-mw", type=float, help="MW of the standard protein (e.g. 66.4 BSA)")
    p.add_argument("--bg-radius", "--ball", dest="ball", type=float,
                   help="background window half-width in px (default: height/20); features "
                        "narrower than twice this are kept as bands: raise it for very thick "
                        "(overloaded) bands, lower it for uneven staining")
    p.add_argument("--min-prominence", type=float, default=0.02,
                   help="minimum band prominence relative to the lane's strongest (0.02)")
    p.add_argument("--floor", type=float, default=0.03,
                   help="minimum band height relative to the gel's strongest band (0.03)")
    p.add_argument("--label-min", type=float, default=5.0,
                   help="label bands with at least this %% of their lane (5)")
    p.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    p.add_argument("-o", "--outdir", help="output folder (default: <image dir>/gelquant_results)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.ladder_lane in ("0", ""):
        args.ladder_lane = None
    files = _collect(args.inputs)
    if not files:
        print("no images found", file=sys.stderr)
        return 1
    failed = 0
    summary: dict[Path, list[dict]] = {}
    for path in files:
        outdir = Path(args.outdir) if args.outdir else path.parent / "gelquant_results"
        outdir.mkdir(parents=True, exist_ok=True)
        try:
            rows = process(path, args, outdir)
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {path}: {exc}", file=sys.stderr)
            continue
        summary.setdefault(outdir, []).extend(rows)
    for outdir, rows in summary.items():
        if rows and len(files) > 1:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with open(outdir / "summary_lanes.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
        print(f"results written to {outdir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
