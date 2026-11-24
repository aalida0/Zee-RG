import numpy as np
import pytest

from zee import stu as X


def test_uv_finite_delta_independent():
    a = X.stu(400.0, 1500.0, 0.6, 700.0, 650.0, 0.07, delta=0.0)
    b = X.stu(400.0, 1500.0, 0.6, 700.0, 650.0, 0.07, delta=37.0)
    assert np.allclose(a, b, rtol=0, atol=1e-10)


def test_phi_independent_when_charged_degenerate():
    r = [X.stu(500.0, 500.0, p, 700.0, 650.0, 0.05) for p in (0.0, 0.4, 1.1)]
    assert np.allclose(r[0], r[1], atol=1e-10) and np.allclose(r[0], r[2], atol=1e-10)


def test_2hdm_limit_custodial_T_zero():
    # aligned, sphi=0 (doublet is h1+): T = 0 for m+ = mA or m+ = mH
    for mp, mH, mA in [(600.0, 900.0, 600.0), (600.0, 600.0, 900.0)]:
        S, T, U = X.stu(mp, 5000.0, 0.0, mH, mA, 0.0)
        assert abs(T) < 1e-8


def test_2hdm_T_sign_and_closed_form():
    mp, mH, mA = 400.0, 700.0, 300.0
    S, T, U = X.stu(mp, 5000.0, 0.0, mH, mA, 0.0)
    ref = (X.F(mp**2, mH**2) + X.F(mp**2, mA**2) - X.F(mH**2, mA**2)) / (16 * np.pi**2 * X.ALPHA_MZ * X.V**2)
    assert np.isclose(T, ref, rtol=1e-12) and T < 0  # T ~ (m+ - mA)(m+ - mH) < 0 between them


def test_degenerate_everything_gives_small_stu():
    S, T, U = X.stu(500.0, 5000.0, 0.0, 500.0, 500.0, 0.0, mh=125.25)
    assert abs(T) < 1e-6 and abs(S) < 0.02 and abs(U) < 0.02


def test_mixing_changes_T_and_factor_switch():
    a = X.stu(300.0, 800.0, 0.5, 600.0, 600.0, 0.0)[1]
    b = X.stu(300.0, 800.0, 0.5, 600.0, 600.0, 0.0, hg_factor=True)[1]
    assert a != b
    w1, w2 = np.cos(0.5) ** 2, np.sin(0.5) ** 2
    d = (b - a) * 16 * np.pi**2 * X.ALPHA_MZ * X.V**2
    assert np.isclose(d, w1 * w2 * X.F(300.0**2, 800.0**2), rtol=1e-10)


def test_chi2_at_center_zero():
    assert X.chi2_stu(*X.STU_CENTRAL) == pytest.approx(0.0)
