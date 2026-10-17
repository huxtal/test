"""Command-line batch fitting of Affinity ITC .nitc files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from .fit import bootstrap, fit
from .integrate import integrate
from .models import MODELS, Design, get_model
from .nitc import read_nitc
from .plot import KCAL, plot_fit


def _collect(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files += sorted(q for q in p.rglob("*") if q.suffix.lower() == ".nitc")
        elif p.exists():
            files.append(p)
        else:
            files += sorted(Path().glob(item))
    return files


def _parse_fix(items: list[str]) -> dict[str, float]:
    out = {}
    for item in items:
        k, _, v = item.partition("=")
        if not v:
            raise SystemExit(f"--fix expects NAME=VALUE, got {item!r}")
        out[k.strip()] = float(v)
    return out


def _warnings(result) -> list[str]:
    warns = []
    th = result.thermodynamics()
    for sfx in ("", "1", "2"):
        c = th.get(f"c{sfx}")
        if c is None:
            continue
        if c > 1000:
            warns.append(f"c{sfx} = {c:.0f} > 1000: curve is too steep, Ka{sfx} is poorly "
                         "determined (consider lower concentrations or displacement ITC)")
        elif c < 1 and f"n{sfx}" not in result.fixed:
            warns.append(f"c{sfx} = {c:.2g} < 1: low-c titration, fix n{sfx} "
                         "(e.g. --fix n=1) for a meaningful Ka")
    for k in result.free:
        v, e = result.values[k], result.errors[k]
        if np.isfinite(e) and abs(e) > max(abs(v), 1e-9) and not k.startswith("dH0"):
            warns.append(f"{k} is not well determined ({v:.3g} ± {e:.2g})")
    if not result.success:
        warns.append(f"optimizer did not converge: {result.message}")
    return warns


def process(path: Path, args, outdir: Path) -> dict:
    run = read_nitc(path)
    if args.cell_conc is not None:
        run.cell_conc = args.cell_conc
    if args.syringe_conc is not None:
        run.syringe_conc = args.syringe_conc
    if args.cell_volume is not None:
        run.cell_volume = args.cell_volume
    exo_up = run.exotherm_up ^ args.flip_sign

    integ = integrate(run, baseline=args.baseline, window=args.window, exotherm_up=exo_up)
    design = Design(run.inj_volume, run.cell_volume, run.cell_conc, run.syringe_conc,
                    run.temperature)
    model = get_model(args.model)
    result = fit(model, design, integ.heat, exclude=args.exclude, fixed=_parse_fix(args.fix))
    if args.bootstrap:
        bootstrap(result, n=args.bootstrap)

    stem = outdir / run.name
    f = KCAL if args.energy == "kcal" else 1.0
    eu = f"{args.energy}_mol"

    with open(f"{stem}_injections.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["injection", "start_s", "end_s", "volume_uL", "Mt_mM", "Xt_mM",
                    "molar_ratio", "heat_uJ", "heat_noise_uJ", f"dH_{eu}_injectant",
                    f"fit_{eu}_injectant", "included"])
        nmol = design.injected_mol
        for i in range(run.n_injections):
            w.writerow([i + 1, run.inj_start[i], run.inj_end[i], run.inj_volume[i],
                        f"{design.Mt[i]:.6g}", f"{design.Xt[i]:.6g}",
                        f"{design.molar_ratio[i]:.6g}", f"{integ.heat[i]:.6g}",
                        f"{integ.noise[i]:.3g}", f"{integ.heat[i] / nmol[i] / f:.6g}",
                        f"{result.fitted[i] / nmol[i] / f:.6g}", int(result.included[i])])

    if args.thermogram:
        np.savetxt(f"{stem}_thermogram.csv",
                   np.column_stack([run.time, run.heat_rate, integ.baseline]),
                   delimiter=",", header="time_s,heat_rate_uJ_s,baseline_uJ_s", comments="")

    plot_fit(run, integ, result, f"{stem}_fit.{args.format}", energy=args.energy)

    th = {k: float(v) for k, v in result.thermodynamics().items()}
    warns = _warnings(result)
    report = {
        "file": str(path),
        "model": model.name,
        "cell_conc_mM": run.cell_conc,
        "syringe_conc_mM": run.syringe_conc,
        "cell_volume_uL": run.cell_volume,
        "temperature_C": run.temperature,
        "n_injections": run.n_injections,
        "excluded": [i + 1 for i in range(run.n_injections) if not result.included[i]],
        "baseline": args.baseline,
        "exotherm_up": exo_up,
        "parameters": result.values,
        "errors_1sd": result.errors,
        "fixed": result.fixed,
        "ci95_bootstrap": result.ci95,
        "derived": th,
        "rmsd_uJ": result.rmsd,
        "r_squared": result.r_squared,
        "warnings": warns,
    }
    with open(f"{stem}_fit.json", "w") as fh:
        json.dump(report, fh, indent=2)

    row = {"file": run.name, "model": model.name, "T_C": run.temperature,
           "cell_mM": run.cell_conc, "syringe_mM": run.syringe_conc}
    for k in model.param_names:
        scale = f if k.startswith("dH") else 1.0
        name = f"{k}_{eu}" if k.startswith("dH") else k
        row[name] = result.values[k] / scale
        row[f"{name}_err"] = result.errors[k] / scale
    for k, v in th.items():
        if k.startswith(("dG", "-TdS")):
            row[k.replace("kJmol", eu)] = v / f
        elif k.startswith(("Kd", "c")):
            row[k] = v
    row["rmsd_uJ"] = result.rmsd
    row["r_squared"] = result.r_squared
    row["warnings"] = "; ".join(warns)
    return row


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fit_itc",
        description="Integrate and fit TA Instruments Affinity ITC (.nitc) titrations.")
    p.add_argument("inputs", nargs="+", help=".nitc files, directories or glob patterns")
    p.add_argument("-m", "--model", default="onesite", choices=sorted(MODELS),
                   help="binding model (default: onesite)")
    p.add_argument("-x", "--exclude", type=int, nargs="*", default=[1],
                   help="1-based injections to exclude from the fit (default: 1)")
    p.add_argument("--fix", action="append", default=[], metavar="NAME=VALUE",
                   help="hold a parameter fixed, e.g. --fix n=1 --fix dH0=0 "
                        "(logKa in log10 M^-1, dH in kJ/mol)")
    p.add_argument("--baseline", choices=["auto", "file"], default="auto",
                   help="auto: median of the last --window s before each injection; "
                        "file: baseline nodes stored by ITCRun")
    p.add_argument("--window", type=float, default=20.0,
                   help="baseline averaging window in s (default: 20)")
    p.add_argument("--cell-conc", type=float, help="override cell concentration (mM)")
    p.add_argument("--syringe-conc", type=float, help="override syringe concentration (mM)")
    p.add_argument("--cell-volume", type=float, help="override cell volume (uL)")
    p.add_argument("--flip-sign", action="store_true",
                   help="invert the heat sign if the file's ExothermUp flag is wrong")
    p.add_argument("--bootstrap", type=int, default=0, metavar="N",
                   help="add 95%% CIs from N residual-bootstrap refits")
    p.add_argument("--energy", choices=["kJ", "kcal"], default="kJ",
                   help="energy unit for plots and summary (default: kJ)")
    p.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                   help="figure format (default: png)")
    p.add_argument("--thermogram", action="store_true",
                   help="also export time / heat rate / baseline as CSV")
    p.add_argument("-o", "--outdir", help="output directory (default: <input dir>/itcfit_results)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    files = _collect(args.inputs)
    if not files:
        print("no .nitc files found", file=sys.stderr)
        return 1

    rows, failed = [], 0
    for path in files:
        outdir = Path(args.outdir) if args.outdir else path.parent / "itcfit_results"
        outdir.mkdir(parents=True, exist_ok=True)
        try:
            row = process(path, args, outdir)
        except Exception as exc:  # keep going through a batch
            failed += 1
            print(f"[FAIL] {path}: {exc}", file=sys.stderr)
            continue
        rows.append((outdir, row))
        kd = row.get("Kd_uM", row.get("Kd1_uM"))
        dh_key = next((k for k in row if k.startswith("dH") and not k.endswith("_err")), None)
        msg = f"[ OK ] {path.name}: "
        if kd is not None:
            msg += f"Kd = {kd:.3g} uM, n = {row.get('n', row.get('n1')):.3f}, "
        msg += f"{dh_key} = {row[dh_key]:.2f}, RMSD = {row['rmsd_uJ']:.3g} uJ"
        print(msg)
        if row["warnings"]:
            for w in row["warnings"].split("; "):
                print(f"       ! {w}")

    by_dir: dict[Path, list[dict]] = {}
    for outdir, row in rows:
        by_dir.setdefault(outdir, []).append(row)
    for outdir, group in by_dir.items():
        keys = list(dict.fromkeys(k for r in group for k in r))
        with open(outdir / "summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(group)
        print(f"results written to {outdir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
