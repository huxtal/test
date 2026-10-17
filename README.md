# itcfit: automated fitting of TA Instruments Affinity ITC data

Reads `.nitc` files written by **ITCRun** (Affinity ITC / Nano ITC) directly, so there's no
export from NanoAnalyze. It then integrates the injection peaks, fits a binding model and
writes figures and tables ready for a manuscript. It handles one file or a whole folder at once.

```
python fit_itc.py run1.nitc                      # one file
python fit_itc.py /data/ITC/2026-10/             # every .nitc under a folder (batch)
python fit_itc.py *.nitc --fix n=1 --energy kcal --format pdf
python fit_itc.py blank.nitc -m blank            # control titration (heat of dilution)
python fit_itc.py run1.nitc -m twosites --bootstrap 200
```

On Windows: `py fit_itc.py C:\ITC\run1.nitc`. Install the dependencies once with
`pip install -r requirements.txt` (numpy, scipy, matplotlib).

## Outputs (default: `itcfit_results/` next to each input file)

| file | content |
|---|---|
| `<run>_fit.png` | thermogram + baseline, isotherm + fit + parameter box, residuals |
| `<run>_injections.csv` | per-injection volume, Mt, Xt, molar ratio, heat (µJ), normalized ΔH, fit |
| `<run>_fit.json` | all parameters, 1 SD errors, bootstrap CIs, ΔG, −TΔS, c-value, warnings |
| `<run>_thermogram.csv` | time, heat rate, baseline (with `--thermogram`) |
| `summary.csv` | one row per file, for comparing a whole batch |

## How it works

1. **Reading the file.** A `.nitc` file is a gzip-compressed .NET `BinaryFormatter`
   (MS-NRBF) serialization of a `DataSet`. `itcfit/nrbf.py` is a small, dependency-free
   NRBF decoder. `itcfit/nitc.py` maps the tables onto an `ITCRun` object:
   `DataPointsTable` (Time, HeatRate µJ/s, …), `InjectionsTable` (start/end, volume µL),
   `BaselineNodesTable`, and `KVP` (cell/syringe concentration in mM, cell volume in µL,
   temperature, `ExothermUp`, `heatRateOffset`, …). `read_tables()` gives you every column if
   you need the other channels (temperatures, DACs, pre-titration data).
2. **Baseline and integration.** By default (`--baseline auto`), there is one baseline node
   just before each injection: the median of the last `--window` seconds (default 20 s).
   The nodes are joined by straight lines, and each injection is integrated with the trapezoid
   rule. `--baseline file` uses the nodes that ITCRun saved instead. Heats are reported as
   heat absorbed by the sample, so exothermic binding gives ΔH < 0. The sign comes from the
   file's `ExothermUp` flag; `--flip-sign` reverses it.
3. **Models** (`itcfit/models.py`). Total concentrations follow the MicroCal/Origin
   overflow-cell treatment, and incremental heats include the displaced-volume correction.
   - `onesite`: n identical, independent sites (exact quadratic). Parameters are logKa, ΔH, n,
     and an offset ΔH0 (a constant heat of dilution per mol of injectant).
   - `twosites`: two independent sets of sites (n1, K1, ΔH1; n2, K2, ΔH2) plus the offset.
   - `blank`: a constant heat of dilution only.
4. **Fitting** (`itcfit/fit.py`). The fit is unattended:
   - Starting values come from a grid over the nonlinear parameters (logKa, n). For each grid
     point, ΔH and the offset are solved exactly by linear least squares.
   - The best starts are then refined with bounded `scipy.optimize.least_squares` (TRF).
   - Errors are 1 SD from the covariance matrix. `--bootstrap N` adds 95 % CIs from
     resampling the residuals.
   - The first injection is excluded by default (`-x` changes this; `-x` alone keeps all).
   - Any parameter can be held fixed, e.g. `--fix n=1 --fix dH0=0`.
5. **Diagnostics.** The program warns when c = n·Ka·[M] is above 1000 (Ka poorly determined)
   or below 1 with n free, and when any parameter's error is larger than the parameter itself.

Units used internally: µL × mM × kJ/mol = µJ, so no conversion factors are needed.

## Using it from Python / Jupyter

```python
from itcfit import read_nitc, integrate, Design, get_model, fit
run = read_nitc("run1.nitc")
heats = integrate(run).heat                     # µJ per injection
d = Design(run.inj_volume, run.cell_volume, run.cell_conc, run.syringe_conc, run.temperature)
res = fit(get_model("onesite"), d, heats, exclude=[1])
print(res.values, res.errors, res.thermodynamics())
```

## Tests

`python -m pytest tests`. Set `ITCFIT_SAMPLE_NITC=/path/to/file.nitc` to also run the
real-file test.

Tested with ITCRun 3.7 files (Affinity ITC, `FileVersion` in the KVP table). The `.nitc`
format was decoded independently from the NRBF spec. The open-source
[ViaBind](https://github.com/GuangluWu/ViaBind_itc) project (MIT) provided the hint that the
payload is gzip-compressed, and its example file was used for validation.
