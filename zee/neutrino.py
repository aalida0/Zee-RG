"""One-loop neutrino mass matrix of the general Zee model and its diagonalisation.

Two forms are provided and tested against each other:

* ``mnu_hg_leading``: HG eq. (2.30)-(2.31), leading order in m_lepton^2/m_scalar^2.
* ``mnu_finite``: the same structure with the exact lepton-mass dependence of the
  Feynman-parameter loop integral, in the charged-lepton mass basis with
  D = -tan(beta) sqrt(2) m/v + Y2/cos(beta).

Masses are in GeV throughout; multiply by 1e9 for eV. f is in the HG convention
(a SARAH/Zee-Pheno Fzee equals 2 f).
"""
import numpy as np

from . import V_EW


def loop_kernel(m1, m2, ml):
    """K = int_0^1 ln[(x ml^2 + (1-x) m1^2)/(x ml^2 + (1-x) m2^2)] dx, closed form.

    Uses int_0^1 ln(x z + (1-x) a) dx = (a ln a - z ln z)/(a - z) - 1. For ml -> 0 it
    reduces to ln(m1^2/m2^2). The a = z and b = z limits are handled analytically.
    """
    a, b, z = m1**2, m2**2, ml**2

    def g(a_, z_):
        if z_ == 0.0:
            return np.log(a_)
        if abs(a_ - z_) < 1e-9 * max(a_, z_):
            return np.log(z_) + 1.0
        return (a_ * np.log(a_) - z_ * np.log(z_)) / (a_ - z_)

    return g(a, z) - g(b, z)


def s2phi_from(m1, m2, mu, v=V_EW):
    return np.sqrt(2.0) * v * mu / (m2**2 - m1**2)


def mnu_hg_leading(f, m_lep, Y2, tan_beta, m1, m2, mu, v=V_EW):
    """HG eqs. (2.30)-(2.31). f: 3x3 antisymmetric, m_lep: 3 masses [GeV], Y2: 3x3."""
    f = np.asarray(f, dtype=complex)
    Y2 = np.asarray(Y2, dtype=complex)
    mE = np.diag(m_lep).astype(complex)
    sb = tan_beta / np.sqrt(1.0 + tan_beta**2)
    s2 = s2phi_from(m1, m2, mu, v)
    A = s2 * tan_beta / (8.0 * np.sqrt(2.0) * np.pi**2 * v) * np.log(m2**2 / m1**2)
    t1 = f @ mE @ mE + mE @ mE @ f.T
    t2 = f @ mE @ Y2 + Y2.T @ mE @ f.T
    return A * (t1 - v / (np.sqrt(2.0) * sb) * t2)


def d_matrix(m_lep, Y2, tan_beta, v=V_EW):
    """D = -sin(beta) Y1 + cos(beta) Y2 with Y1 fixed by the charged-lepton masses."""
    cb = 1.0 / np.sqrt(1.0 + tan_beta**2)
    return -tan_beta * np.sqrt(2.0) * np.diag(m_lep) / v + np.asarray(Y2, dtype=complex) / cb


def mnu_finite(f, m_lep, Y2, tan_beta, m1, m2, mu, v=V_EW):
    """Exact-in-m_lepton version: M = a + a^T, a_ij = f_ik w_k D_kj."""
    f = np.asarray(f, dtype=complex)
    s2 = s2phi_from(m1, m2, mu, v)
    w = np.array([s2 / (16.0 * np.pi**2) * ml * loop_kernel(m1, m2, ml) for ml in m_lep])
    a = (f * w[np.newaxis, :]) @ d_matrix(m_lep, Y2, tan_beta, v)
    return a + a.T


def takagi(M):
    """Takagi decomposition M = U* diag(m) U^dagger for complex symmetric M.

    Returns (m ascending, U). Assumes non-degenerate masses; a degenerate spectrum
    leaves the columns of U ambiguous within each degenerate subspace.
    """
    M = np.asarray(M, dtype=complex)
    if np.max(np.abs(M - M.T)) > 1e-12 * max(1.0, np.max(np.abs(M))):
        raise ValueError("matrix is not symmetric")
    w, U = np.linalg.eigh(M.conj().T @ M)
    order = np.argsort(w)
    w, U = w[order], U[:, order]
    m = np.sqrt(np.clip(w, 0.0, None))
    for k in range(3):
        c = U[:, k] @ M @ U[:, k]
        if abs(c) > 0:
            U[:, k] *= np.exp(-0.5j * np.angle(c))
    return m, U


def oscillation_observables(M, ordering="NO"):
    """Masses [GeV], sin^2 angles, delta [deg] and splittings [eV^2] from a Majorana M.

    PDG parametrisation, M = U* diag(m1,m2,m3) U^dagger. In normal ordering the mass
    states are m1<m2<m3; in inverted ordering m3<m1<m2.
    """
    m, U = takagi(M)
    if ordering == "NO":
        idx = [0, 1, 2]
    elif ordering == "IO":
        idx = [1, 2, 0]
    else:
        raise ValueError("ordering must be 'NO' or 'IO'")
    m = m[idx]
    U = U[:, idx]
    ue3 = abs(U[0, 2]) ** 2
    s13sq = ue3
    s12sq = abs(U[0, 1]) ** 2 / (1.0 - ue3)
    s23sq = abs(U[1, 2]) ** 2 / (1.0 - ue3)
    c13sq = 1.0 - s13sq
    jcp = np.imag(U[0, 0] * U[1, 1] * np.conj(U[0, 1]) * np.conj(U[1, 0]))
    s12, c12 = np.sqrt(s12sq), np.sqrt(1 - s12sq)
    s23, c23 = np.sqrt(s23sq), np.sqrt(1 - s23sq)
    s13, c13 = np.sqrt(s13sq), np.sqrt(c13sq)
    denom = 2.0 * s12 * c12 * s23 * c23 * s13
    if denom > 0:
        cosd = (abs(U[1, 0]) ** 2 - s12sq * c23**2 - c12**2 * s23sq * s13sq) / denom
        cosd = float(np.clip(cosd, -1.0, 1.0))
        sind = jcp / (c12 * s12 * c23 * s23 * c13sq * s13)
        delta = np.degrees(np.arctan2(sind, cosd)) % 360.0
    else:
        delta = float("nan")
    ev = 1e9
    dm21 = ((m[1] * ev) ** 2 - (m[0] * ev) ** 2)
    dm3l = ((m[2] * ev) ** 2 - (m[0] * ev) ** 2) if ordering == "NO" else ((m[2] * ev) ** 2 - (m[1] * ev) ** 2)
    return dict(masses_eV=m * ev, s12sq=s12sq, s13sq=s13sq, s23sq=s23sq, delta_deg=delta,
                dm21_eV2=dm21, dm3l_eV2=dm3l, jarlskog=jcp, U=U)
