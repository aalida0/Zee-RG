import numpy as np
import pytest

from zee import V_EW
from zee import scalars as S
from zee import neutrino as N

ME, MMU, MTAU = 0.51099895e-3, 0.1056583755, 1.77686
M_LEP = np.array([ME, MMU, MTAU])


def rand_f(rng, scale):
    a = (rng.normal(size=3) + 1j * rng.normal(size=3)) * scale
    f = np.zeros((3, 3), complex)
    f[0, 1], f[0, 2], f[1, 2] = a
    return f - f.T


def test_charged_round_trip():
    for m1, m2, mu in [(300.0, 800.0, 100.0), (1500.0, 2000.0, 400.0), (200.0, 5e4, 3e3)]:
        for lighter in (False, True):
            MH, M33, phi = S.charged_from_physical(m1, m2, mu, singlet_lighter=lighter)
            a, b, phi2 = S.charged_spectrum(MH, M33, mu)
            assert np.isclose(a, m1**2, rtol=1e-10) and np.isclose(b, m2**2, rtol=1e-10)
            assert np.isclose(S.sin2phi(a, b, mu), np.sin(2 * phi), rtol=1e-9)
            # eigenvector check: h1 = s h + c H must be the light eigenvector
            M = S.charged_mass_matrix(MH, M33, mu)
            vec = np.array([np.cos(phi), np.sin(phi)])  # (H, h) components of h1
            assert np.allclose(M @ vec, m1**2 * vec, rtol=1e-8, atol=1e-6 * m2**2)


def test_charged_rejects_large_mu():
    with pytest.raises(ValueError):
        S.charged_from_physical(500.0, 510.0, 1e4)


def test_neutral_round_trip():
    mh, mH, mA, c_ba, MHp2 = 125.0, 900.0, 880.0, 0.05, 950.0**2
    q = S.neutral_from_physical(mh, mH, mA, c_ba, MHp2)
    w1, w2 = S.neutral_spectrum(q["lam1"], q["lam5"], q["lam6"], mA**2)
    assert np.isclose(w1, mh**2, rtol=1e-10) and np.isclose(w2, mH**2, rtol=1e-10)
    # eq. 2.14 with the sign convention used in neutral_from_physical
    assert np.isclose(abs(S.alignment_s2(q["lam6"], w1, w2)), abs(2 * c_ba * np.sqrt(1 - c_ba**2)), rtol=1e-9)
    # eq. 2.11 reproduces mA^2 from M_H+^2 and lam4, lam5
    assert np.isclose(MHp2 - 0.5 * V_EW**2 * (q["lam5"] - q["lam4"]), mA**2, rtol=1e-12)


def test_kernel_limits():
    m1, m2 = 500.0, 3000.0
    assert np.isclose(N.loop_kernel(m1, m2, 0.0), np.log(m1**2 / m2**2))
    # adaptive quadrature, split near x=1 where the integrand varies on scale ml^2/m^2
    from scipy.integrate import quad
    pts = [1 - 10.0**-k for k in range(1, 10)]
    for ml in (0.1, 1.777, 400.0, 500.0):
        f = lambda x: np.log((x * ml**2 + (1 - x) * m1**2) / (x * ml**2 + (1 - x) * m2**2))
        num = sum(quad(f, a, b, epsabs=0, epsrel=1e-13, limit=200)[0] for a, b in zip([0] + pts, pts + [1]))
        assert np.isclose(N.loop_kernel(m1, m2, ml), num, rtol=1e-11)


def test_hg_leading_equals_finite_for_heavy_scalars():
    rng = np.random.default_rng(1)
    f = rand_f(rng, 1e-3)
    Y2 = (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))) * 1e-3
    for tb in (0.5, 5.0, 30.0):
        a = N.mnu_hg_leading(f, M_LEP, Y2, tb, 5e3, 2e4, 1e3)
        b = N.mnu_finite(f, M_LEP, Y2, tb, 5e3, 2e4, 1e3)
        assert np.allclose(a, b, rtol=1e-3, atol=1e-3 * np.abs(b).max())


def test_finite_corrections_matter_for_tau():
    rng = np.random.default_rng(2)
    f = rand_f(rng, 1e-3)
    Y2 = (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))) * 1e-3
    a = N.mnu_hg_leading(f, M_LEP, Y2, 3.0, 5.0, 40.0, 10.0)
    b = N.mnu_finite(f, M_LEP, Y2, 3.0, 5.0, 40.0, 10.0)
    assert np.abs(a - b).max() > 1e-2 * np.abs(b).max()  # light scalars: expansion fails


def test_symmetric_and_zero_diagonal_structure():
    rng = np.random.default_rng(3)
    f = rand_f(rng, 1e-3)
    Y2 = np.zeros((3, 3), complex)
    M = N.mnu_finite(f, M_LEP, Y2, 10.0, 400.0, 2000.0, 300.0)
    assert np.allclose(M, M.T)
    # Y2 = 0: M_ii = 2 a_ii = 0 since f_ii = 0 and D diagonal (Zee zero-diagonal texture)
    assert np.allclose(np.diag(M), 0, atol=1e-30)


def test_takagi_reconstruction():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    M = A + A.T
    m, U = N.takagi(M)
    assert np.allclose(U.conj() @ np.diag(m) @ U.conj().T, M, atol=1e-10)
    assert np.allclose(U.conj().T @ U, np.eye(3), atol=1e-12)


def test_oscillation_extraction_round_trip():
    s12, s13, s23, d = np.sqrt(0.307), np.sqrt(0.0220), np.sqrt(0.55), np.radians(230.0)
    c12, c13, c23 = [np.sqrt(1 - x**2) for x in (s12, s13, s23)]
    e = np.exp(1j * d)
    U = np.array([
        [c12 * c13, s12 * c13, s13 / e],
        [-s12 * c23 - c12 * s23 * s13 * e, c12 * c23 - s12 * s23 * s13 * e, s23 * c13],
        [s12 * s23 - c12 * c23 * s13 * e, -c12 * s23 - s12 * c23 * s13 * e, c23 * c13],
    ])
    m = np.array([0.01, np.sqrt(0.01**2 + 7.4e-5), np.sqrt(0.01**2 + 2.5e-3)]) * 1e-9
    M = U.conj() @ np.diag(m) @ U.conj().T
    o = N.oscillation_observables(M, "NO")
    assert np.isclose(o["s12sq"], 0.307, atol=1e-8)
    assert np.isclose(o["s13sq"], 0.0220, atol=1e-8)
    assert np.isclose(o["s23sq"], 0.55, atol=1e-8)
    assert np.isclose(o["delta_deg"], 230.0, atol=1e-5)
    assert np.isclose(o["dm21_eV2"], 7.4e-5, rtol=1e-8)
    assert np.isclose(o["dm3l_eV2"], 2.5e-3, rtol=1e-8)
