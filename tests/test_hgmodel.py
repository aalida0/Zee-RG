import os
os.environ.setdefault("ZEE_INPUTS", "hg")
import numpy as np
import pytest
from zee import hgmodel as H

X = np.load(os.path.join(os.path.dirname(__file__), "data", "best_NO.npy"))


def test_regression_best_point():
    """Change detector, NOT a validation: values after the referee-v5 repair (C01 universality, C02/C03 complete dipoles).
    Previous values (3.1044, 2.5532e-13, 2.1897e-8) came from the inverted universality term and leading dipoles.
    The physics is validated in test_dipfull.py and test_couplings.py; update these numbers only with a documented reason."""
    if H.MODERN:
        pytest.skip("regression values are for the HG input set")
    e = H.evaluate(X, "NO")
    assert abs(e["chi2"] / 12.17126 - 1) < 1e-5
    assert abs(e["obs"]["mu_e_g"] / 2.55549e-13 - 1) < 1e-5
    assert abs(e["obs"]["tau_mu_g"] / 2.35736e-8 - 1) < 1e-5


def test_lambda4_zero_is_hg_slice():
    a = H.evaluate(X, "NO"); b = H.evaluate(np.append(X, 0.0), "NO")
    assert a["chi2"] == b["chi2"]


def test_lambda4_shifts_neutral_masses_only():
    s0 = H.scalar_sector(X); s1 = H.scalar_sector(np.append(X, 2.0))
    assert np.isclose(s1["mA"] ** 2 - s0["mA"] ** 2, 0.5 * H.V**2 * 2.0, rtol=1e-6)
    assert np.isclose(s1["m1"], s0["m1"]) and np.isclose(s1["m2"], s0["m2"])


def test_alignment_limit_mass_relations():
    x = X.copy(); x[17] = 0.0              # lambda6 = 0: exact alignment, lambda1 = m_h^2/v^2
    s = H.scalar_sector(x)
    assert np.isclose(s["sba"], 1.0, atol=1e-9)
    assert np.isclose(s["lam1"], H.MH**2 / H.V**2, rtol=1e-9)
    assert np.isclose(s["mH"] ** 2 - s["mA"] ** 2, x[16] * H.V**2, rtol=1e-6)   # M22 = mA^2 + lambda5 v^2


def test_conversion_component_normalization():
    """Cr(Al) = 0.51 x monopole(Au, HG 3.24) + 2.67e-3 Br(mu->e gamma) + scalar term; relative tolerance only"""
    o = H.evaluate(X, "NO")["obs"]
    ref = 0.51 * o["Cr_Au_HG"] + 2.67e-3 * o["mu_e_g"] + o["Cr_Al_S"]
    assert ref > 0 and abs(o["Cr_Al"] / ref - 1) < 1e-12


def test_mu3e_dipole_estimate_ratio():
    o = H.evaluate(X, "NO")["obs"]
    assert 4e-3 < o["mu_3e"] / o["mu_e_g"] < 8e-3     # alpha/3pi (ln(mmu^2/me^2) - 11/4) = 0.0060


def test_light_cp_even_state_must_be_the_125_gev_one():
    import json
    """A point with M22 < m_h^2 (or nearly degenerate) would make the algebraic lam1 root unphysical: it is penalised, not clamped silently."""
    from zee import hgmodel as Hm
    x = np.array(json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'benchmarks_input.json')))['BM1']['x'], float)
    assert Hm.scalar_sector(x)['pen'] == 0.0
    y = x.copy(); y[12] = np.log10(110.0); y[13] = np.log10(110.0); y[16] = -2.0          # light spectrum, M22 below m_h^2
    assert Hm.scalar_sector(y)['pen'] > 0.0
