# gelquant: SDS-PAGE band quantification

Finds lanes and bands in a gel image and measures each band's integrated intensity (its
"volume"). From that it reports:
- molecular weight from the ladder;
- **% of lane** (purity);
- optionally, a target protein tracked across lanes and **µg** from a standard curve.

Each image gets an annotated gel figure, a lane-profile figure and CSV tables.

## Quick start

```
# 1. find the crop box: writes <image>_grid.png with a pixel grid
python quant_gel.py gel.tif --grid

# 2. analyse (BenchMark ladder in lanes 1 and 7, 15 wells)
python quant_gel.py gel.tif --crop 45,280,1080,975 --lanes 15 --ladder-lane 1,7

# protein of interest at ~36 kDa, amounts against BSA standards in lanes 2-5
python quant_gel.py gel.tif --crop ... --lanes 15 --ladder-lane 1 --target-mw 36 \
    --standards 2=0.25,3=0.5,4=1,5=2 --standard-mw 66.4

# rough amounts from the ladder itself (5 uL BenchMark ~ 0.5 ug per band)
python quant_gel.py gel.tif --crop ... --lanes 15 --ladder-lane 1,7 --target-mw 36 --ladder-load 5

# whole folder of gels with the same layout, lane names on the figure
python quant_gel.py D:\gels\2026-01 --lanes 10 --lane-names M,Lys,FT,W1,E1,E2,E3,E4,E5,E6
```

## Getting good numbers

- **Image source.** Use the imager's raw export, ideally a 16-bit grayscale TIFF:
  - ChemiDoc / Image Lab: *File → Export → Export for Analysis*.
  - iBright, Azure, Amersham, LI-COR: the raw TIFF.

  Photos, scans and JPEGs work (the tool picks the most informative colour channel: red for
  Coomassie). Their contrast curves aren't linear, though, so percentages from them are
  semi-quantitative, and the report says so.
- **Crop to the resolving gel.** Leave out the wells, stacking gel, dye front and plastic edges.
  They are full-width dark features and would otherwise read as huge bands.
- **Avoid saturation.** Black, flat-topped bands are underestimated, and the tool warns when
  pixels are saturated. Load less, or image with a shorter exposure.
- **Two ladders** (e.g. `--ladder-lane 1,7`): each lane then gets its own calibration,
  interpolated between the ladders, which corrects for a tilted or smiling gel.
- **Standard curve:** the amounts must bracket your samples, and R² below 0.98 raises a warning.

## Amounts from the ladder (`--ladder-load`)

The BenchMark user guide gives about 0.1 µg/µL of each protein, so 5 µL is about 0.5 µg per
band. With `--ladder-load 5` (or `--ladder-ug 0.5` for any ladder), every ladder band becomes
a calibration point:
- Each band's response (volume per µg) is measured.
- Their geometric mean converts the volume of any band to µg.
- Their spread (geometric SD) is reported as a fold-uncertainty: `ladder_ug_low` /
  `ladder_ug_high` in the CSVs, plus the `<gel>_ladder_mass.png` figure.

The intensified 50 and 20 kDa reference bands are left out, because their amount isn't stated;
give `--ladder-ref-ug` if you know it.

**Treat these as estimates.** On the example gel the ladder bands' responses varied by 60% CV,
about ×/÷1.8. The variation was reproducible between the two ladder lanes, so it is protein-
specific staining, not noise; for example, bands above 100 kDa responded about half as much.
Every band holds the same amount, so the ladder is a one-point calibration, and samples far
above 0.5 µg are extrapolated. The tool warns when a target exceeds 3× the per-band amount.
For accurate amounts, run a BSA (or purified-protein) dilution series and use `--standards`.

## How it works

1. **Image** (`image.py`). Reads TIFF (8/16-bit), PNG or JPG. Colour images use the channel
   with the widest range. Dark-on-light gels (Coomassie, silver) are inverted automatically,
   so bands become positive signal. Optional `--rotate` / `--deskew`.
2. **Lanes** (`lanes.py`).
   - Each row's background is removed, then a column profile is built (99th percentile).
     Lanes are its peaks, centred on the half-maximum edges because lanes are flat-topped.
   - With `--lanes N`, empty or faint lanes are placed on the regular comb grid that best
     fits the lanes found.
   - `--lane-centers` sets lanes by hand.
   - The profile is the mean over the central 60% of each lane (`--lane-width`).
3. **Background** (`bands.py`). A grey-scale opening (minimum then maximum filter over
   2 × `--bg-radius` rows, default gel height / 20), smoothed. It follows uneven staining and
   gradients but not bands, and it doesn't depend on intensity, so faint and strong bands are
   treated alike.
4. **Bands.**
   - Peaks must reach 2% of the lane's strongest band (`--min-prominence`), 3% of the gel's
     strongest band (`--floor`, which keeps empty lanes empty) and 4 × the noise.
   - Each band is integrated out to the nearest of: the valley to its neighbour, 2% of its
     peak, or 2.5 × its FWHM.
   - Lanes whose strongest band is under 5% of the gel's strongest are flagged *low signal*.
5. **Ladder** (`ladder.py`).
   - Presets: `benchmark` (Invitrogen 10747-012, default), `pageruler-plus`,
     `pageruler-prestained`, `pageruler-unstained`, `precision-plus`; or `--ladder-mw` for a
     custom list.
   - The tallest peaks in the ladder lane are the ladder bands (background bumps are low and
     broad), assigned top to bottom.
   - If bands are missing, the intensified reference bands (BenchMark 50 and 20 kDa) decide
     which sizes are absent.
   - MW is interpolated point to point (monotone PCHIP in log MW), because real gels aren't
     log-linear. For example, BenchMark's 20 kDa band runs close to 25 kDa and far from 15 kDa.
6. **Quantities** (`quant.py`):
   - % of lane;
   - target band: closest to `--target-mw` within `--target-tol` (±10%);
   - target relative to `--reference-lane`;
   - linear standard curve, giving µg.

## Outputs (`gelquant_results/` next to each image)

| file | content |
|---|---|
| `<gel>_gel.png` | gel with lane boxes, ladder sizes, each band marked; main bands labelled MW / % |
| `<gel>_profiles.png` | every lane's profile: raw, background (dashed), integrated bands shaded |
| `<gel>_bands.csv` | one row per band: lane, position, MW, extrapolated?, height, volume, % of lane, target?, µg (standards) and ladder µg with range |
| `<gel>_lanes.csv` | one row per lane: role, main band MW and %, target MW / % / relative / µg, low-signal flag |
| `<gel>_standard.png` | standard curve with the target bands (with `--standards`) |
| `<gel>_ladder_mass.png` | per-band ladder response vs size (with `--ladder-load` / `--ladder-ug`) |
| `<gel>_report.json` | settings, lane positions, ladder assignment, standard curve, warnings |
| `summary_lanes.csv` | all lanes of all gels (when several images are processed) |

## Python

```python
from gelquant import load_gel, analyse_gel, LADDERS, REFERENCE_BANDS
gel = load_gel("gel.tif", crop=(45, 280, 1080, 975))
res = analyse_gel(gel, n_lanes=15, ladder_lanes=[1, 7], ladder_sizes=LADDERS["benchmark"],
                  ladder_reference=REFERENCE_BANDS["benchmark"], target_mw=36)
for p in res.sample_profiles:
    print(p.lane.number, [(round(b.mw, 1), round(b.percent, 1)) for b in p.bands])
```

`gelquant.synthetic.make_gel` draws test gels with known band amounts; `tests/test_gel.py`
uses it to check lane positions, MW, purity and standard-curve recovery.
