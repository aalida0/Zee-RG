"""Barr-Zee two-loop dipoles from a closed fermion loop (t, b, tau) and a neutral scalar.

c_R^{ij} (operator  e_i sigma P_R e_j F, decay j -> i) =
    e alpha N_c Q_f^2 /(16 pi^3 m_f) * [ -S_f f(z) - i P_f g(z) ] * Lambda_ij,   z = m_f^2/m_phi^2
with -L = phi ( ebar_L Lambda e_R + h.c.) for leptons and -L = phi fbar (S_f + i P_f gamma5) f.
Normalisation fixed by the Barr-Zee a_mu formula Delta a = alpha m_mu^2/(4 pi^3 v^2) N Q^2 y_mu y_f (-f or +g),
via c = e a/(4 m); sign convention checked against the one-loop neutral scalar term of HG (3.17).
f(z) = (1-2z) g(z) + z (ln z + 2),  g(z) = z/2 int dx ln(x(1-x)/z)/(x(1-x)-z).
"""
import numpy as np
from scipy.special import spence

E = np.sqrt(4 * np.pi / 137.035999)
ALPHA = 1 / 137.035999


def _li2(x):
    return spence(1.0 - x)


def g_fn(z):
    y = np.sqrt(complex(1.0 - 4.0 * z))
    val = z / y * (_li2(1.0 - (1.0 - y) / (2.0 * z)) - _li2(1.0 - (1.0 + y) / (2.0 * z)))
    return float(np.real(val))


def f_fn(z):
    return (1.0 - 2.0 * z) * g_fn(z) + z * (np.log(z) + 2.0)


def cR_bz(Lam, scal, fermions):
    """Lam: dict phi -> 3x3 Lambda; scal: phi -> mass; fermions: list of (m_f, N_c Q^2, {phi: (S_f, P_f)}).
    Returns the 3x3 matrix c_R^{ij} / e  (so C_R = 2 c_R/(e m_j) = 2 (c_R/e) / m_j)."""
    out = np.zeros((3, 3), complex)
    for phi, L in Lam.items():
        for mf, ncq2, coup in fermions:
            S, P = coup[phi]
            z = mf**2 / scal[phi] ** 2
            out += ALPHA * ncq2 / (16 * np.pi**3 * mf) * (-S * f_fn(z) - 1j * P * g_fn(z)) * L
    return out
