"""Oblique parameters S, T, U of the general Zee model (HG appendix A, with corrections).

Structure: doublet-like neutral scalars (h, H, A) with s_{b-a}, c_{b-a}; two charged
scalars h1+, h2+ whose doublet fractions are w1 = cos^2 phi, w2 = sin^2 phi.

Deviations from the printed HG formulas, each derived from the general coupling
structure (Z/W couple to the charged states through the doublet projector P with
P_ii = w_i, P_12 = -sin phi cos phi) and checked by tests/test_stu.py:
  * T: the charged-charged term is -2 s^2 c^2 F(m1+^2, m2+^2) (Grimus et al.; HG's footnote
    says another reference gives this, HG's text has -s^2 c^2). Switch: ``hg_factor=True``.
  * S: charged part = sum_i (w_i^2 - 2 w_i) B22t(Z,i,i) + 2 w1 w2 B22t(Z,1,2). The printed
    formula's diagonal coefficients are garbled in the PDF text; this form is independent
    of phi for degenerate charged masses and reduces to -B22t(Z,m+,m+) of the 2HDM.
  * S+U: the charged-charged term is -2 sum_ij P_ij^2 B22t(W,i,j) (printed: -2 w_i B22t(W,i,i)).
Tests: Delta-independence (UV finiteness), phi-independence at degenerate masses, 2HDM limit.
"""
import numpy as np

ALPHA_MZ = 1.0 / 127.95
MW, MZ = 80.3692, 91.1876
V = 246.21965
_X, _W = np.polynomial.legendre.leggauss(96)
_X, _W = 0.5 * (_X + 1.0), 0.5 * _W


def F(a, b):
    """HG (A.2), a, b = squared masses."""
    if a == b:
        return 0.0
    return 0.5 * (a + b) - a * b / (a - b) * np.log(a / b)


def _intX(q2, a, b, fn):
    X = a * _X + b * (1 - _X) - q2 * _X * (1 - _X)
    if np.any(X <= 0):
        raise ValueError("below threshold: not supported")
    return float(np.sum(_W * fn(X)))


def B22t(q2, a, b, delta=0.0):
    """B22(q2,a,b) - B22(0,a,b), eq. (A.5)/(A.7); a, b squared masses."""
    f = lambda X: X * np.log(X)
    return (0.25 * (delta + 1.0) * (-q2 / 3.0)
            - 0.5 * (_intX(q2, a, b, f) - _intX(0.0, a, b, f)))


def B0t(q2, a, b):
    f = np.log
    return -(_intX(q2, a, b, f) - _intX(0.0, a, b, f))


def stu(m1, m2, phi, mH, mA, c_ba, mh=125.25, delta=0.0, hg_factor=False):
    """Return (S, T, U) for charged masses m1<=m2 (h1+, h2+), mixing phi, neutral masses."""
    s_ba2, c_ba2 = 1.0 - c_ba**2, c_ba**2
    a1, a2 = m1**2, m2**2
    w = (np.cos(phi) ** 2, np.sin(phi) ** 2)
    mh2, mH2, mA2, mW2, mZ2 = mh**2, mH**2, mA**2, MW**2, MZ**2
    # ---- T
    T = 0.0
    for wi, ai in zip(w, (a1, a2)):
        T += wi * (c_ba2 * F(ai, mh2) + s_ba2 * F(ai, mH2) + F(ai, mA2))
    T -= (1.0 if hg_factor else 2.0) * w[0] * w[1] * F(a1, a2)
    T -= c_ba2 * F(mh2, mA2) + s_ba2 * F(mH2, mA2)
    T += 3.0 * c_ba2 * (F(mZ2, mH2) - F(mW2, mH2) - F(mZ2, mh2) + F(mW2, mh2))
    T /= 16.0 * np.pi**2 * ALPHA_MZ * V**2
    # ---- S
    b = lambda q2, x, y: B22t(q2, x, y, delta)
    S = (s_ba2 * b(mZ2, mH2, mA2)
         + c_ba2 * (b(mZ2, mh2, mA2) + b(mZ2, mZ2, mH2) - b(mZ2, mZ2, mh2)
                    - mZ2 * B0t(mZ2, mZ2, mH2) + mZ2 * B0t(mZ2, mZ2, mh2)))
    S += sum((wi**2 - 2 * wi) * b(mZ2, ai, ai) for wi, ai in zip(w, (a1, a2)))
    S += 2 * w[0] * w[1] * b(mZ2, a1, a2)
    S /= np.pi * mZ2
    # ---- S + U
    SU = 0.0
    for wi, ai in zip(w, (a1, a2)):
        SU += wi * (b(mW2, ai, mA2) + s_ba2 * b(mW2, ai, mH2) + c_ba2 * b(mW2, mh2, ai))
    SU += c_ba2 * (b(mW2, mW2, mH2) - b(mW2, mW2, mh2)
                   - mW2 * B0t(mW2, mW2, mH2) + mW2 * B0t(mW2, mW2, mh2))
    SU -= 2.0 * (w[0] ** 2 * b(mW2, a1, a1) + w[1] ** 2 * b(mW2, a2, a2)
                 + 2 * w[0] * w[1] * b(mW2, a1, a2))
    SU /= np.pi * mW2
    return S, T, SU - S


# RPP 2026 (1 June 2026), Table 10.8, global EW fit.
# Free U: S = 0.018(95), T = 0.04(12), U = -0.013(86); rho_ST = 0.91, rho_SU = -0.66, rho_TU = -0.88.
STU_CENTRAL = np.array([0.018, 0.04, -0.013])
STU_SIGMA = np.array([0.095, 0.12, 0.086])
RHO = {"ST": 0.91, "SU": -0.66, "TU": -0.88}
# U = 0 fit: S = 0.008(71), T = 0.021(55), rho_ST = 0.92.
ST_CENTRAL = np.array([0.008, 0.021])
ST_SIGMA = np.array([0.071, 0.055])
RHO_ST_U0 = 0.92


def chi2_stu(S, T, U, rho=RHO):
    c = np.array([[1, rho["ST"], rho["SU"]], [rho["ST"], 1, rho["TU"]], [rho["SU"], rho["TU"], 1]])
    cov = c * np.outer(STU_SIGMA, STU_SIGMA)
    d = np.array([S, T, U]) - STU_CENTRAL
    return float(d @ np.linalg.solve(cov, d))


def chi2_st(S, T):
    """Fit with U fixed to 0 in the data fit; use only when the model's U is negligible."""
    cov = np.array([[1, RHO_ST_U0], [RHO_ST_U0, 1]]) * np.outer(ST_SIGMA, ST_SIGMA)
    d = np.array([S, T]) - ST_CENTRAL
    return float(d @ np.linalg.solve(cov, d))
