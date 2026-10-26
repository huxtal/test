import numpy as np
import pytest

skimage = pytest.importorskip("skimage")
from skimage import io  # noqa: E402

from gelquant import LADDERS, REFERENCE_BANDS, analyse_gel, load_gel  # noqa: E402
from gelquant.synthetic import make_gel  # noqa: E402

BENCH = LADDERS["benchmark"]


@pytest.fixture(scope="module")
def synth_gel(tmp_path_factory):
    ladder = [(m, 3.0 if m in (50, 20) else 1.5) for m in BENCH]
    lanes = [ladder]
    lanes += [[(66.4, a)] for a in (0.5, 1, 2, 4)]                  # BSA standards
    lanes += [[(42, 3), (70, 0.3), (25, 0.2)], [(42, 6), (55, 0.6)], [], [(42, 2), (30, 2)]]
    img = make_gel(lanes, ref_mw=300, slope=420, top=30, height=640)
    path = tmp_path_factory.mktemp("gel") / "synth.tif"
    io.imsave(str(path), img, check_contrast=False)
    return path


def _analyse(path):
    return analyse_gel(load_gel(path), n_lanes=9, ladder_lanes=[1], ladder_sizes=BENCH,
                       ladder_reference=REFERENCE_BANDS["benchmark"], target_mw=42,
                       standards={2: 0.5, 3: 1, 4: 2, 5: 4}, standard_mw=66.4)


def test_lanes_and_ladder(synth_gel):
    r = _analyse(synth_gel)
    centers = [ln.center for ln in r.lanes]
    np.testing.assert_allclose(centers, 80 * np.arange(1, 10), atol=2)
    cal = r.ladder_cals[1]
    assert cal.reference_ok
    assert r.low_signal(8) and not r.low_signal(6)


def test_molecular_weights(synth_gel):
    r = _analyse(synth_gel)
    mws = sorted(b.mw for b in r.profile(9).bands)
    assert mws[0] == pytest.approx(30, rel=0.03)
    assert mws[1] == pytest.approx(42, rel=0.03)


def test_purity_and_standard_curve(synth_gel):
    r = _analyse(synth_gel)
    lane6 = sorted(r.profile(6).bands, key=lambda b: -b.volume)
    assert lane6[0].percent == pytest.approx(100 * 3 / 3.5, abs=3)
    assert r.standard.r_squared > 0.999
    for lane, amount in ((6, 3), (7, 6), (9, 2)):
        got = float(r.standard.amount(r.targets[lane].volume))
        assert got == pytest.approx(amount, rel=0.06)


def test_ladder_with_extra_and_missing_bands():
    from gelquant.bands import Band
    from gelquant.ladder import calibrate
    ys = 30 + 420 * np.log10(300 / np.array(BENCH))
    bands = [Band(1, i, y, 0, 0, 0.5 if m in (50, 20) else 0.3, 1.0)
             for i, (y, m) in enumerate(zip(ys, BENCH))]
    bands.append(Band(1, 99, 15.0, 0, 0, 0.05, 1.0))            # faint artefact above
    cal = calibrate(bands, BENCH, reference=[50, 20])
    np.testing.assert_allclose(cal.mw, BENCH)
    missing = [b for b in bands if b.y != ys[3] and b.index != 99]   # drop the 100 kDa band
    cal = calibrate(missing, BENCH, reference=[50, 20])
    assert cal.unmatched == 1 and 100 not in cal.mw.tolist()
    assert float(cal.predict(ys[3])) == pytest.approx(100, rel=0.05)
