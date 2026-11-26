"""One-loop RGEs with ALL fermion Yukawa couplings of the Higgs-basis Zee model (report v10): top and bottom with both doublets (rho_t, rho_b = r y), the charged-lepton
Yukawas of H1 (diagonal masses) and of H2 (the 3x3 matrix Y2 = Dd, complex) and the singlet coupling f, together with g1, g2, g3 and the eleven quartics.
Machacek-Vaughn for real scalars / Weyl fermions (zee.rge.beta_Y, zee.rgeq.quartic_beta) with three extra Weyl states for b^c; the first-two-generation quarks
are neglected (y < 0.007).  State z = [g1,g2,g3, yt, rt, yb, rb, Y1(9 complex), Y2(9), f(9), c(11)].  r_t = rho_t is NOT assumed to be -tan(beta) y_t: it runs.
Basis: H1 carries the vev (Higgs basis); Y_{ij}: L-index i, e^c index j, as in zee.rge."""
import numpy as np
from scipy.integrate import solve_ivp
from . import rge, rgeq
NU, E, EC, TL, BL, TC, BC = 0, 3, 6, 9, 12, 15, 18
NF = 21
NC = 11
_Ts = None


def _quartic_basis():
    global _Ts
    if _Ts is None:
        Ts = [rgeq.tensor_from(np.eye(NC)[i]) for i in range(NC)]
        M = np.array([t.ravel() for t in Ts]).T
        _Ts = (Ts, M, np.linalg.pinv(M))
    return _Ts


def build_Y(Y1, Y2, f, yt, rt, yb, rb):
    """real-scalar Yukawa tensor (10, NF, NF).  Complex scalars H1p,H10,H2p,H20,eta -> (Re,Im) pairs; T multiplies phi, Tc multiplies phi*."""
    T = np.zeros((5, NF, NF), complex); Tc = np.zeros((5, NF, NF), complex)
    def add(M, k, a, b, c):
        M[k, a, b] += c; M[k, b, a] += c
    for k, Yl in ((0, Y1), (2, Y2)):
        for i in range(3):
            for j in range(3):
                add(Tc, k + 1, EC + j, E + i, Yl[i, j]); add(Tc, k, EC + j, NU + i, Yl[i, j])
    for i in range(3):
        for j in range(3):
            add(T, 4, NU + i, E + j, f[i, j]); add(T, 4, E + i, NU + j, -f[i, j])
    for c in range(3):
        for k, (y_u, y_d) in ((0, (yt, yb)), (2, (rt, rb))):        # doublet 1 (k=0 charged, k+1 neutral) and doublet 2 (k=2)
            for (a, b, kk, s_) in ((TC + c, TL + c, k + 1, 1.0), (TC + c, BL + c, k, -1.0)):
                T[kk, a, b] += s_ * y_u; T[kk, b, a] += s_ * y_u
            add(Tc, k + 1, BC + c, BL + c, y_d); add(Tc, k, BC + c, TL + c, y_d)
    Y = np.zeros((10, NF, NF), complex)
    for k in range(5):
        Y[2*k] = (T[k] + Tc[k])/np.sqrt(2); Y[2*k + 1] = 1j*(T[k] - Tc[k])/np.sqrt(2)
    return Y


def casimirs(g):
    g1, g2, g3 = g
    C = np.zeros(NF)
    for i in range(3):
        for base, y, c2 in ((NU, -.5, .75), (E, -.5, .75), (EC, 1, 0)):
            C[base + i] = g1**2*y**2 + g2**2*c2
    for base, y, c2, c3 in ((TL, 1/6, .75, 4/3), (BL, 1/6, .75, 4/3), (TC, -2/3, 0, 4/3), (BC, 1/3, 0, 4/3)):
        for i in range(3):
            C[base + i] = g1**2*y**2 + g2**2*c2 + g3**2*c3
    return C


BG = rge.BG


def pack(g, yt, rt, yb, rb, Y1, Y2, f, c):
    return np.concatenate([g, [yt, rt, yb, rb], np.concatenate([Y1.ravel(), Y2.ravel(), f.ravel()]).view(float), c])


def unpack(z):
    cpx = z[7:61].view(complex)
    return z[:3], z[3], z[4], z[5], z[6], cpx[:9].reshape(3, 3), cpx[9:18].reshape(3, 3), cpx[18:27].reshape(3, 3), z[61:]


def rhs(t, z, return_resid=False):
    g, yt, rt, yb, rb, Y1, Y2, f, c = unpack(z)
    Y = build_Y(Y1, Y2, f, yt, rt, yb, rb)
    bY = rge.beta_Y(Y, g, casimirs(g))                                   # (10, NF, NF)
    Tk = np.zeros((5, NF, NF), complex); Tck = np.zeros((5, NF, NF), complex)
    for k in range(5):
        re, im = bY[2*k], bY[2*k + 1]; Tk[k] = (re - 1j*im)/np.sqrt(2); Tck[k] = (re + 1j*im)/np.sqrt(2)
    dY1 = np.array([[Tck[1][EC + j, E + i] for j in range(3)] for i in range(3)])
    dY2 = np.array([[Tck[3][EC + j, E + i] for j in range(3)] for i in range(3)])
    G = np.array([[Tk[4][NU + i, E + j] for j in range(3)] for i in range(3)])
    df = (G - G.T)/4           # the tensor entry is y^{nu_i e_j} = f_ij - f_ji = 2 f_ij (antisymmetric f): beta_f = antisymmetric part of G / 2 (report v11 fix; was G)
    dyt = Tk[1][TC, TL].real; drt = Tk[3][TC, TL].real
    dyb = Tck[1][BC, BL].real; drb = Tck[3][BC, BL].real
    Ts, M, pinv = _quartic_basis()
    lam = sum(ci*Ti for ci, Ti in zip(c, Ts))
    dq = rgeq.quartic_beta(lam, Y, g)
    dc = pinv @ dq.ravel()
    if return_resid: return np.abs(M @ dc - dq.ravel()).max(), np.abs(dq).max()         # (discarded part of the quartic beta tensor, its overall size)
    dg = g**3*BG/(16*np.pi**2)
    return pack(dg, dyt, drt, dyb, drb, dY1, dY2, df, dc)


def yukawa_max(z):
    g, yt, rt, yb, rb, Y1, Y2, f, c = unpack(z)
    return max(abs(yt), abs(rt), abs(yb), abs(rb), np.abs(Y1).max(), np.abs(Y2).max(), 2*np.abs(f).max())
