"""Lepton-flavour observables of the general Zee model (first layer).

Implemented here, with the derivation in each docstring:
  * h -> l_i l_j via the off-diagonal Higgs-basis Yukawa D (c_{beta-alpha} suppressed)
  * l_j -> l_i gamma and Delta a_l from the singly-charged scalars h1+, h2+ and the
    antisymmetric f ONLY (massless neutrino in the loop, no chirality enhancement)
  * the matching Delta a_l

Checked against HG: Br = alpha/(48 pi G_F^2) K^2 |f_et* f_mt|^2 follows from HG eqs. (3.12),
(3.19) with g = 2 (sin phi, cos phi), and equals the expression below. Eq. (3.14), the
H+ term with the Y2 coupling (-sqrt2 tb c_phi, sqrt2 tb s_phi), is NOT included.

NOT implemented yet (their absence biases any fit that used this file alone):
  * neutral-scalar (h, H, A) one-loop dipoles with Y2 and the tau-mass chirality
    enhancement, and the two-loop Barr-Zee terms
  * l -> 3l, mu-e conversion, electron EDM
Index convention: 0=e, 1=mu, 2=tau, in the charged-lepton mass basis.
"""
import numpy as np

from . import V_EW
from .neutrino import d_matrix

ALPHA = 1.0 / 137.035999
GF = 1.1663788e-5            # GeV^-2
M_H = 125.25
GAMMA_H_SM = 4.07e-3         # GeV
BR_TAU_LNUNU = {0: 0.1782, 1: 0.1739}  # tau -> e nu nu, mu nu nu
M_LEP = np.array([0.51099895e-3, 0.1056583755, 1.77686])


def higgs_lfv_width(i, j, D, c_ba, m_h=M_H):
    """Gamma(h -> l_i^- l_j^+) + conjugate = m_h/(8 pi) (|y_ij|^2 + |y_ji|^2)  (HG eq. 3.48),
    with y = c_{beta-alpha} D / sqrt(2) the h coupling in -L = y_ij lbar_Li l_Rj h + h.c.
    (massless final leptons). Depends only on |D_ij|^2 + |D_ji|^2, so the
    D vs D^dagger ambiguity of the paper convention drops out.
    """
    y = c_ba * np.asarray(D) / np.sqrt(2.0)
    return m_h / (8.0 * np.pi) * (abs(y[i, j]) ** 2 + abs(y[j, i]) ** 2)


def higgs_lfv_br(i, j, Y2, tan_beta, c_ba, m_lep=M_LEP, v=V_EW):
    D = d_matrix(m_lep, Y2, tan_beta, v)
    g = higgs_lfv_width(i, j, D, c_ba)
    return g / (GAMMA_H_SM + g)


def _scalar_weights(m1, m2, phi):
    """h+ = sin(phi) h1+ + cos(phi) h2+ -> (s^2, c^2)."""
    return np.sin(phi) ** 2, np.cos(phi) ** 2


def charged_coupling(f):
    """Coupling matrix y[a, i] of nubar^c_a P_R l_i h+ : the antisymmetric f gives
    L-tilde-bar f L h+ = 2 f_ab nubar^c_a l_b h+, so y = 2 f."""
    return 2.0 * np.asarray(f, dtype=complex)


def delta_a(i, f, m1, m2, phi, m_lep=M_LEP):
    """Delta a_l = - m_l^2 sum_a |y_ai|^2 K / (96 pi^2), K = s^2/m1^2 + c^2/m2^2.

    Charged scalar, massless neutral fermion, one chirality coupling (heavy scalar).
    """
    y = charged_coupling(f)
    s2, c2 = _scalar_weights(m1, m2, phi)
    K = s2 / m1**2 + c2 / m2**2
    return -m_lep[i] ** 2 * np.sum(abs(y[:, i]) ** 2) * K / (96.0 * np.pi**2)


def dipole_amplitude(i, j, f, m1, m2, phi):
    """S_ij K with S_ij = sum_a conj(y_ai) y_aj. Units GeV^-2 times coupling^2."""
    y = charged_coupling(f)
    s2, c2 = _scalar_weights(m1, m2, phi)
    K = s2 / m1**2 + c2 / m2**2
    return np.sum(np.conj(y[:, i]) * y[:, j]) * K


def width_lgamma(j, i, f, m1, m2, phi, m_lep=M_LEP):
    """Gamma(l_j -> l_i gamma) = alpha m_j^5 |S K|^2 / (16 (96 pi^2)^2) (single dipole chirality).

    From Gamma = m^3 |c_R|^2/(4 pi) with c_R = e a_ji/(4 m_j) and the Delta a above with
    |y|^2 -> S_ij.
    """
    amp = dipole_amplitude(i, j, f, m1, m2, phi)
    return ALPHA * m_lep[j] ** 5 * abs(amp) ** 2 / (16.0 * (96.0 * np.pi**2) ** 2)


def br_mu_egamma(f, m1, m2, phi):
    """Br = alpha/(768 pi G_F^2) |S K|^2 (GF-normalised, Gamma_mu = G_F^2 m^5/192 pi^3)."""
    amp = dipole_amplitude(0, 1, f, m1, m2, phi)
    return ALPHA / (768.0 * np.pi * GF**2) * abs(amp) ** 2


def br_tau_lgamma(i, f, m1, m2, phi):
    """tau -> l_i gamma, i in {0 (e), 1 (mu)}, normalised to Br(tau -> l_i nu nu)."""
    amp = dipole_amplitude(i, 2, f, m1, m2, phi)
    return BR_TAU_LNUNU[i] * ALPHA / (768.0 * np.pi * GF**2) * abs(amp) ** 2
