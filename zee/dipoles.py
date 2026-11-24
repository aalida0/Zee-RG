"""l_i -> l_j gamma Wilson coefficients following HG sec. 3.3.2 (eqs. 3.12-3.19).

Br(l_i -> l_j gamma) / Br(l_i -> l_j nu nu) = 48 pi^3 alpha / G_F^2 (|C_L|^2 + |C_R|^2).

Contributions included (all one loop):
  * neutral h, H, A with a tau in the loop, chirality-enhanced (3.13, 3.16, 3.17), as
    written by HG: only Y2^{tau tau}-type flips, valid for the leading m_tau terms.
  * singly-charged scalars with massless neutrinos: the f term (C_R, = 3.19) and the
    Y2 (H+) term (C_L). The H+ term is written from first principles as
    -(1/192 pi^2) K_H (D D^dag)_{ji}, because the printed eq. (3.14) is garbled in the PDF
    text; it reproduces (3.14)'s structure (Y2^{mu tau} (D_tautau)^*) for tau -> mu.
NOT included: two-loop Barr-Zee, neutral-scalar loops with e or mu inside, W/Z loops.
All lepton indices 0=e, 1=mu, 2=tau, charged-lepton mass basis; f in the HG convention.
"""
import numpy as np

from . import V_EW
from .neutrino import d_matrix
from .observables import ALPHA, GF, M_H, M_LEP, BR_TAU_LNUNU

PREF = 48.0 * np.pi**3 * ALPHA / GF**2


def neutral_couplings(tb, c_ba):
    """HG (2.19)-(2.20) for (h, H, A): returns g1 (x m/v) and g2 (x Y2^dag)."""
    cb = 1.0 / np.sqrt(1.0 + tb**2)
    sb = tb * cb
    s_ba = np.sqrt(1.0 - c_ba**2)
    sa = sb * c_ba - cb * s_ba
    ca = cb * c_ba + sb * s_ba
    g1 = np.array([-sa / cb, ca / cb, -1j * tb])
    g2 = np.array([c_ba, -s_ba, 1j]) / (np.sqrt(2.0) * cb)
    return g1, g2


def _flip_bracket(g1, g2, Ytt, mtau, v):
    """The two brackets of eq. (3.13), per neutral scalar."""
    b1 = g1 * mtau / v + g2 * np.conj(Ytt)
    b2 = np.conj(g1) * mtau / v + np.conj(g2) * Ytt
    return b1, b2


def tau_to_l(j, Y2, tb, c_ba, masses, v=V_EW):
    """(C_L^0, C_R^0) for tau -> l_j gamma (j=0 e, 1 mu), eq. (3.13) and its CR counterpart.

    masses = (mh, mH, mA)."""
    g1, g2 = neutral_couplings(tb, c_ba)
    m2 = np.array(masses, float) ** 2
    mt = M_LEP[2]
    L = np.log(m2 / mt**2) - 1.5
    Ytt = Y2[2, 2]
    b1, b2 = _flip_bracket(g1, g2, Ytt, mt, v)
    cl = np.sum(np.conj(g2) * Y2[j, 2] / m2 * (b1 / 6.0 + b2 * L))
    # C_R: Y^{ab} -> (Y^{ba})^*, so Y_tt -> conj(Y_tt) and Y^{j tau} -> conj(Y^{tau j})
    b1r, b2r = _flip_bracket(g1, g2, np.conj(Ytt), mt, v)
    cr = np.sum(np.conj(g2) * np.conj(Y2[2, j]) / m2 * (b1r / 6.0 + b2r * L))
    return cl / (16.0 * np.pi**2), cr / (16.0 * np.pi**2)


def mu_to_e(Y2, tb, c_ba, masses):
    """(C_L^0, C_R^0) for mu -> e gamma, eqs. (3.16)-(3.17)."""
    _, g2 = neutral_couplings(tb, c_ba)
    m2 = np.array(masses, float) ** 2
    mt, mm = M_LEP[2], M_LEP[1]
    L = np.log(m2 / mt**2) - 1.5
    cl = np.sum(g2**2 / m2 * L) * Y2[2, 1] * Y2[0, 2]
    cr = np.sum(g2**2 / m2 * L) * np.conj(Y2[1, 2]) * np.conj(Y2[2, 0])
    pre = mt / mm / (16.0 * np.pi**2)
    return pre * cl, pre * cr


def charged(i_init, j_fin, f, Y2, tb, m1, m2, phi, v=V_EW):
    """(C_L^+, C_R^+): H+ (D D^dag) term in C_L, f term in C_R."""
    s2, c2 = np.sin(phi) ** 2, np.cos(phi) ** 2
    K_f = s2 / m1**2 + c2 / m2**2          # h+ = s h1 + c h2
    K_H = c2 / m1**2 + s2 / m2**2          # H+ = c h1 - s h2
    D = d_matrix(M_LEP, Y2, tb, v)
    y = 2.0 * np.asarray(f, complex)
    S_f = np.sum(np.conj(y[:, j_fin]) * y[:, i_init])
    S_H = (D @ D.conj().T)[j_fin, i_init]
    k = -1.0 / (192.0 * np.pi**2)
    return k * K_H * S_H, k * K_f * S_f


def br(i_init, j_fin, f, Y2, tb, c_ba, mH, mA, m1, m2, phi, mh=M_H, v=V_EW):
    """Br(l_i -> l_j gamma) for (mu,e), (tau,e), (tau,mu)."""
    masses = (mh, mH, mA)
    cl_c, cr_c = charged(i_init, j_fin, f, Y2, tb, m1, m2, phi, v)
    if (i_init, j_fin) == (1, 0):
        cl_n, cr_n = mu_to_e(Y2, tb, c_ba, masses)
        norm = 1.0
    elif i_init == 2 and j_fin in (0, 1):
        cl_n, cr_n = tau_to_l(j_fin, Y2, tb, c_ba, masses, v)
        norm = BR_TAU_LNUNU[j_fin]
    else:
        raise ValueError("supported: (1,0), (2,0), (2,1)")
    return norm * PREF * (abs(cl_n + cl_c) ** 2 + abs(cr_n + cr_c) ** 2)
