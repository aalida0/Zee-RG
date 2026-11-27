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


# ---- V13 referee P0: the charged-charged coefficient of T (general Zee result -2 w1 w2 F, Grimus et al. 0711.4022 Eq. 70)
def _T_projector(m1, m2, phi, mH, mA, c_ba, mh=125.25):
    """Scalar-loop part of T built from explicit coupling matrices (independent of stu.stu's bookkeeping):
    W loops: charged state a with doublet component u_a and a neutral state n with Higgs-basis component kappa_n (coupling g/2 each, counted once);
    Z loops: neutral pair (H2 -> A, coupling g/(2 c_W), counted once) and charged pairs with the doublet projector P_ab = u_a u_b (T3 = 1/2, the Q s_W^2 part is diagonal),
    where every ordered pair (a,b), a != b, is a separate vertex of the Lagrangian and the loop is counted for both orders.  Gauge-boson loops (c_ba != 0) are not included."""
    s_ba2, c_ba2 = 1.0 - c_ba**2, c_ba**2
    u = np.array([np.cos(phi), -np.sin(phi)]); P = np.outer(u, u); a = np.array([m1**2, m2**2])
    kap2 = {mh**2: c_ba2, mH**2: s_ba2, mA**2: 1.0}
    T = sum(u[i]**2 * k2 * X.F(a[i], m2n) for i in range(2) for m2n, k2 in kap2.items())
    T -= c_ba2 * X.F(mh**2, mA**2) + s_ba2 * X.F(mH**2, mA**2)
    T -= sum(P[i, j]**2 * X.F(a[i], a[j]) for i in range(2) for j in range(2) if i != j)
    return T / (16 * np.pi**2 * X.ALPHA_MZ * X.V**2)


@pytest.mark.parametrize("m1,m2,phi,mH,mA", [(300.0, 800.0, 0.5, 600.0, 580.0), (450.0, 900.0, 0.25, 700.0, 690.0), (612.0, 869.0, 1.3, 652.0, 629.0)])
def test_T_charged_pair_matches_projector_form(m1, m2, phi, mH, mA):
    assert np.isclose(X.stu(m1, m2, phi, mH, mA, 0.0)[1], _T_projector(m1, m2, phi, mH, mA, 0.0), rtol=1e-10, atol=1e-14)
    # with c_ba != 0 the gauge-loop term is independent of (m1, m2, phi): compare differences
    d1 = X.stu(m1, m2, phi, mH, mA, 0.05)[1] - X.stu(m1 * 1.1, m2 * 0.9, phi * 0.7, mH, mA, 0.05)[1]
    d2 = _T_projector(m1, m2, phi, mH, mA, 0.05) - _T_projector(m1 * 1.1, m2 * 0.9, phi * 0.7, mH, mA, 0.05)
    assert np.isclose(d1, d2, rtol=1e-9)


def test_legacy_option_differs_by_exactly_one_pair_term():
    m1, m2, phi = 612.0, 869.0, 0.34
    leg = X.stu(m1, m2, phi, 652.0, 629.0, 0.0009, hg_factor=True)[1]; gen = X.stu(m1, m2, phi, 652.0, 629.0, 0.0009)[1]
    assert np.isclose((leg - gen) * 16 * np.pi**2 * X.ALPHA_MZ * X.V**2, np.cos(phi)**2 * np.sin(phi)**2 * X.F(m1**2, m2**2), rtol=1e-10)


def _prod_stu(env):
    import json, os, subprocess, sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = ("import sys, json; sys.path[:0] = [%r, %r, %r]\nfrom zee import hgmodel as H\nd = json.load(open(%r))['BM1']\n"
            "e = H.evaluate(list(d['x']), 'NO', eta=d['eta'])\nprint(json.dumps([e['chi_stu'], list(map(float, e['stu']))]))"
            % (root, os.path.join(root, 'gate1'), os.path.join(root, 'gate2'), os.path.join(root, 'results', 'benchmarks_input.json')))
    e = {k: v for k, v in os.environ.items() if not k.startswith(('ZEE_STU', 'ZEE_Z95', 'ZEE_SUMM'))}; e.update(ZEE_INPUTS='modern', **env)
    r = subprocess.run([sys.executable, '-W', 'ignore', '-c', code], capture_output=True, text=True, env=e)
    assert r.returncode == 0, r.stderr[-300:]
    return json.loads(r.stdout.strip().split('\n')[-1])


def test_production_likelihood_uses_general_T_old_and_rpp_inputs():
    """The production objective (hgmodel.evaluate) must use the -2 w1 w2 F term by default; the legacy form only through ZEE_STU_LEGACY_HG=1."""
    c_def, v_def = _prod_stu({}); c_leg, v_leg = _prod_stu({'ZEE_STU_LEGACY_HG': '1'})
    assert v_leg[1] - v_def[1] == pytest.approx(0.0633757, abs=2e-6)          # BM1: delta T of the referee table
    assert abs(v_def[1]) < 0.01 and v_leg[1] > 0.06
    c_rpp, v_rpp = _prod_stu({'ZEE_STU': 'rpp2026'})
    assert v_rpp == pytest.approx(v_def) and c_rpp == pytest.approx(X.chi2_stu(*v_rpp)) and c_rpp == pytest.approx(0.8932, abs=1e-3)
    old = np.array(v_def) - np.array([0.05, 0.09, 0.01]); cov = np.array([[1, .90, -.59], [.90, 1, -.83], [-.59, -.83, 1]]) * np.outer([.11, .13, .11], [.11, .13, .11])
    assert c_def == pytest.approx(float(old @ np.linalg.solve(cov, old)), rel=1e-9) and c_def == pytest.approx(4.5368, abs=1e-3)
