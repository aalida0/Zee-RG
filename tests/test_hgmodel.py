import os
os.environ.setdefault("ZEE_INPUTS", "hg")
import numpy as np
import pytest
from zee import hgmodel as H

X = np.load(os.path.join(os.path.dirname(__file__), "data", "best_NO.npy"))


def test_regression_best_point():
    # HG-input fit point: chi2 and two LFV observables frozen (catches silent changes of the model)
    if H.MODERN:
        pytest.skip("regression values are for the HG input set")
    e = H.evaluate(X, "NO")
    assert np.isclose(e["chi2"], 3.1044, atol=1e-3)
    assert np.isclose(e["obs"]["mu_e_g"], 2.5532e-13, rtol=1e-3)
    assert np.isclose(e["obs"]["tau_mu_g"], 2.1897e-8, rtol=1e-3)


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


def test_dipole_conversion_term():
    if not H.MODERN:
        pytest.skip("Cr_Al only meaningful with ZEE_INPUTS=modern (term itself is input independent)")
    o = H.evaluate(X, "NO")["obs"]
    assert np.isclose(o["Cr_Al"] - o["Cr_Au"], H.ALPHA / 3 * o["mu_e_g"], rtol=1e-9)


def test_mu3e_dipole_estimate_ratio():
    o = H.evaluate(X, "NO")["obs"]
    assert 4e-3 < o["mu_3e"] / o["mu_e_g"] < 8e-3     # alpha/3pi (ln(mmu^2/me^2) - 11/4) = 0.0060
