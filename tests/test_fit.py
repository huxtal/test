import os
from pathlib import Path

import numpy as np
import pytest

from itcfit import Design, fit, get_model, integrate, read_nitc


def _design(**kw):
    args = dict(volumes=np.full(20, 2.5), cell_volume=170.0, cell_conc=0.05,
                syringe_conc=0.6, temperature=25.0)
    args.update(kw)
    return Design(**args)


def test_onesite_recovers_parameters():
    d = _design()
    model = get_model("onesite")
    truth = {"logKa": 6.3, "dH": -42.0, "n": 0.9, "dH0": -1.5}
    rng = np.random.default_rng(1)
    y = model.heats(truth, d) + rng.normal(0, 0.05, 20)
    r = fit(model, d, y, exclude=[1])
    assert r.values["logKa"] == pytest.approx(6.3, abs=0.05)
    assert r.values["dH"] == pytest.approx(-42.0, rel=0.01)
    assert r.values["n"] == pytest.approx(0.9, rel=0.01)
    assert r.r_squared > 0.999


def test_fixed_parameter_is_held():
    d = _design()
    model = get_model("onesite")
    y = model.heats({"logKa": 5.0, "dH": -30.0, "n": 1.0, "dH0": 0.0}, d)
    r = fit(model, d, y, fixed={"n": 1.0, "dH0": 0.0})
    assert r.values["n"] == 1.0 and np.isnan(r.errors["n"])
    assert r.values["logKa"] == pytest.approx(5.0, abs=1e-3)


def test_twosites_reduces_to_onesite_when_site2_empty():
    d = _design()
    p = {"logKa1": 6.0, "dH1": -40.0, "n1": 1.0, "logKa2": 4.0, "dH2": 0.0, "n2": 0.0, "dH0": 0.0}
    one = get_model("onesite").heats({"logKa": 6.0, "dH": -40.0, "n": 1.0, "dH0": 0.0}, d)
    two = get_model("twosites").heats(p, d)
    np.testing.assert_allclose(two, one, rtol=1e-6, atol=1e-9)


def test_units_full_saturation():
    # Very tight binding, first tiny injection: all ligand binds, so the
    # normalized heat approaches dH (uL * mM * kJ/mol = uJ).
    d = _design(volumes=np.array([0.1, 0.1]))
    q = get_model("onesite").heats({"logKa": 12, "dH": -50.0, "n": 1.0, "dH0": 0.0}, d)
    assert q[0] / d.injected_mol[0] == pytest.approx(-50.0, rel=2e-3)


SAMPLE = os.environ.get("ITCFIT_SAMPLE_NITC")


@pytest.mark.skipif(not SAMPLE, reason="set ITCFIT_SAMPLE_NITC to a .nitc file")
def test_read_and_fit_sample():
    run = read_nitc(Path(SAMPLE))
    assert run.n_injections > 0 and len(run.time) == len(run.heat_rate)
    integ = integrate(run)
    d = Design(run.inj_volume, run.cell_volume, run.cell_conc, run.syringe_conc, run.temperature)
    r = fit(get_model("onesite"), d, integ.heat, exclude=[1])
    assert r.success and r.r_squared > 0.9


def test_blank_correction_modes():
    from itcfit.integrate import blank_correction
    blank = np.array([-10.0, -2.0, -2.2, -1.8, -2.0])     # first injection is an outlier
    vb = np.full(5, 2.0)
    vs = np.array([1.0, 2.5, 2.5])
    mean = blank_correction(blank, vb, vs, mode="mean", exclude=[1])
    np.testing.assert_allclose(mean, -1.0 * vs)            # -1.0 uJ/uL average
    pw = blank_correction(blank, vb, vs, mode="pointwise")
    np.testing.assert_allclose(pw, [-5.0, -2.5, -2.75])


def test_smooth_curve_passes_through_fitted_points():
    from itcfit.models import smooth_isotherm
    d = _design(volumes=np.full(20, 2.0), cell_volume=185.0, cell_conc=0.005, syringe_conc=0.05)
    m = get_model("onesite")
    p = {"logKa": 7.9, "dH": -70.0, "n": 1.04, "dH0": -9.0}
    x, y = smooth_isotherm(m, p, d)
    pts = m.heats({**p, "dH0": 0.0}, d) / d.injected_mol
    np.testing.assert_allclose(np.interp(d.molar_ratio, x, y), pts, rtol=1e-6)
