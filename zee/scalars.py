"""Tree-level scalar sector of the general Zee model in the Higgs basis (HG sec. 2.1).

Charged sector: HG eqs. (2.5)-(2.9). Neutral sector: eqs. (2.10)-(2.14).
The relations were taken from a machine summary of the paper, not from the PDF
text, so tests/test_scalars.py checks them for internal consistency only.
"""
import numpy as np

from . import V_EW


def charged_mass_matrix(MHp2, M332, mu, v=V_EW):
    """Charged mass matrix in the (H+, h+) basis, eq. (2.5)."""
    off = -mu * v / np.sqrt(2.0)
    return np.array([[MHp2, off], [off, M332]], dtype=float)


def charged_spectrum(MHp2, M332, mu, v=V_EW):
    """Return (m1^2, m2^2, phi) for h1+ = s h+ + c H+, h2+ = c h+ - s H+.

    m1 <= m2. phi lies in [0, pi/2] for mu >= 0, so h1+ is mostly the singlet
    for phi > pi/4. Closed form of eq. (2.9); s2phi of eq. (2.8) follows from it.
    """
    d = np.sqrt((MHp2 - M332) ** 2 + 2.0 * v**2 * mu**2)
    m1sq = 0.5 * (MHp2 + M332 - d)
    m2sq = 0.5 * (MHp2 + M332 + d)
    phi = np.arctan2(MHp2 - m1sq, mu * v / np.sqrt(2.0))
    return m1sq, m2sq, phi


def sin2phi(m1sq, m2sq, mu, v=V_EW):
    """Eq. (2.8): s_{2 phi} = sqrt(2) v mu / (m_{h2+}^2 - m_{h1+}^2)."""
    return np.sqrt(2.0) * v * mu / (m2sq - m1sq)


def charged_from_physical(m1, m2, mu, v=V_EW, singlet_lighter=False):
    """Invert eq. (2.9): physical masses and mu -> (M_H+^2, M_33^2, phi).

    Two solutions share the same masses; singlet_lighter selects phi > pi/4
    (h1+ mostly the singlet). Raises if |s2phi| > 1, i.e. the requested mu is
    too large for the mass splitting.
    """
    if not (0 < m1 <= m2):
        raise ValueError("need 0 < m1 <= m2")
    if m1 == m2 and mu != 0:
        raise ValueError("mu != 0 requires m1 < m2")
    s2 = 0.0 if mu == 0 else np.sqrt(2.0) * v * mu / (m2**2 - m1**2)
    if abs(s2) > 1.0 + 1e-12:
        raise ValueError("|sin 2phi| > 1: mu too large for this splitting")
    phi = 0.5 * np.arcsin(min(1.0, s2))
    if singlet_lighter:
        phi = 0.5 * np.pi - phi
    s, c = np.sin(phi), np.cos(phi)
    MHp2 = c**2 * m1**2 + s**2 * m2**2
    M332 = s**2 * m1**2 + c**2 * m2**2
    return MHp2, M332, phi


def neutral_from_physical(mh, mH, mA, c_ba, MHp2, v=V_EW):
    """Physical neutral masses and c_{beta-alpha} -> quartics lam1, lam4, lam5, lam6.

    CP-even matrix eq. (2.10) with h = s_ba phi1 + c_ba phi2 (eq. 2.12):
    M11 = lam1 v^2, M12 = lam6 v^2, M22 = mA^2 + lam5 v^2. The pseudoscalar mass
    eq. (2.11) then fixes lam4 once M_H+^2 is known.
    """
    s_ba = np.sqrt(1.0 - c_ba**2)
    m11 = mh**2 * s_ba**2 + mH**2 * c_ba**2
    m12 = (mh**2 - mH**2) * s_ba * c_ba
    m22 = mh**2 * c_ba**2 + mH**2 * s_ba**2
    lam1 = m11 / v**2
    lam6 = m12 / v**2
    lam5 = (m22 - mA**2) / v**2
    lam4 = lam5 - 2.0 * (MHp2 - mA**2) / v**2
    return dict(lam1=lam1, lam4=lam4, lam5=lam5, lam6=lam6)


def neutral_spectrum(lam1, lam5, lam6, mA2, v=V_EW):
    """Eigenvalues (mh^2, mH^2) of eq. (2.10), lightest first, and s_{2(beta-alpha)}."""
    m = np.array([[lam1 * v**2, lam6 * v**2], [lam6 * v**2, mA2 + lam5 * v**2]])
    w = np.linalg.eigvalsh(m)
    return w[0], w[1]


def alignment_s2(lam6, mh2, mH2, v=V_EW):
    """Eq. (2.14): s_{2(beta-alpha)} = -2 lam6 v^2 / (mH^2 - mh^2)."""
    return -2.0 * lam6 * v**2 / (mH2 - mh2)
