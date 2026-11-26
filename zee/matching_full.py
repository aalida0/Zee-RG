"""Full one-loop matching of the Higgs-basis boundary quartics with momentum-dependent self-energies (report v11).

Same unknowns, tadpole equations and physical inputs as zee/matching.py, but the zero-momentum Hessian of V_1 is replaced by the on-shell-momentum self-energy Pi(s) of zee/selfenergy.py:
the pole-mass conditions read  [d2 V_tree(p) + Pi(s_ij)/(16 pi^2)]_ij = (M^2_input)_ij  in the mass basis of the input point, i.e. the pole positions are the input masses and the null vectors of the
inverse propagator at the poles are the input mixing vectors.  The self-energies are evaluated at the input (tree-level) point, whose spectrum is the PHYSICAL input, so the Goldstone modes are exactly
massless and no Goldstone-log regulator is needed (the difference to evaluating at the matched parameters is two-loop).  Off-diagonal elements need a prescription for s_ij: s_ij = (m_i^2 + m_j^2)/2
('sym'), min or max of the two; the spread is the systematic of the mixing treatment; 'hpole-X' evaluates every pair containing the 125 GeV state at s = m_h^2 (what on-shell h observables probe) and the other pairs with X.  Bottom, tau, lighter fermions and the photon-loop IR structure beyond Landau gauge are as in selfenergy.py; v = 246.22 GeV is
the minimum of the one-loop effective potential and is held fixed.  Real parts only (no widths).
"""
import numpy as np
from scipy.optimize import root
from . import matching as M, selfenergy as SE

V = M.V; K16 = 1/(16*np.pi**2)


def pi_eff(c0, sc, mu0, g, yt, rt, prescription='sym', parts='SFV'):
    """effective (symmetric) self-energy matrix in the field basis, built from Pi(s_ij) with the chosen prescription; also returns the target matrix and diagnostics"""
    phi0 = np.zeros(10); phi0[2] = V
    P0 = M.tree_params(c0, sc); T0 = M.mass_tensors(c0, P0); Mt = M.mass2_at(phi0, T0)
    eng = SE.Pi(phi0, T0, (g[0], g[1]), yt, rt, mu0)
    m2 = eng.m2; O = eng.O; n = 10
    phys = np.array([m > 1.0 for m in m2]); cache = {}; Pm = np.zeros((n, n)); ih = int(np.argmin(np.abs(m2 - M.MH**2)))
    for i in range(n):
        for j in range(i, n):
            if not (phys[i] and phys[j]): continue
            pr = prescription
            if prescription.startswith('hpole-'):        # pairs with the 125 GeV state are evaluated at its pole (the composition of h seen by on-shell h observables), the others as given
                pr = prescription.split('-')[1]
                if ih in (i, j): s = m2[ih]; key = round(s, 3)
                else: s = {'sym': 0.5*(m2[i] + m2[j]), 'min': min(m2[i], m2[j]), 'max': max(m2[i], m2[j])}[pr]
            else: s = {'sym': 0.5*(m2[i] + m2[j]), 'min': min(m2[i], m2[j]), 'max': max(m2[i], m2[j])}[pr]
            key = round(s, 3)
            if key not in cache: cache[key] = O.T @ eng.at(s, parts) @ O
            Pm[i, j] = Pm[j, i] = cache[key][i, j]
    return O @ Pm @ O.T*K16, Mt, dict(m2=m2, nS=len(cache))


def match_full(c0, sc, mu0, g, yt, r, prescription='sym', kappa=1.0, tol=1e-9, parts='SFV'):
    phi0 = np.zeros(10); phi0[2] = V; X0 = kappa*M.MH**2; rt = r*yt
    PiE, Mt, info = pi_eff(c0, sc, mu0, g, yt, rt, prescription, parts)
    P0 = M.tree_params(c0, sc)
    u0 = np.array([P0['m11'], P0['m22'], P0['m12'], P0['mh2'], P0['mu'], c0[0], c0[3], c0[4], c0[5]])
    scale = np.array([V**2, V**2, V**2, V**2, V, 1, 1, 1, 1])
    E = np.eye(10); h = 1.5

    def resid(w):
        u = u0 + w*scale; P, c = M._unpack(u, c0); T = M.mass_tensors(c, P)
        fun = lambda ph: M.V1(ph, T, mu0, g[0], g[1], yt, rt, X0, 'SVT')
        g1l = np.array([(-fun(phi0 + 2*h*E[i]) + 8*fun(phi0 + h*E[i]) - 8*fun(phi0 - h*E[i]) + fun(phi0 - 2*h*E[i]))/(12*h) for i in M.IDX])
        gt = M._vac().vgrad(phi0, c, P['m11'], P['m22'], P['m12'], P['mh2'], P['mu'])[1]
        Ht = M.mass2_at(phi0, T)
        R = [(gt[i] + g1l[k])/V**3 for k, i in enumerate(M.IDX)] + [(Ht + PiE - Mt)[i, j]/V**2 for i, j in M.H_IDX]
        return np.array(R)
    sol = root(resid, np.zeros(9), method='hybr', tol=tol)
    u = u0 + sol.x*scale; P, c = M._unpack(u, c0)
    return dict(c=c, delta=c - np.asarray(c0, float), P=P, ok=bool(sol.success), resid=float(np.abs(resid(sol.x)).max()), nfev=int(sol.nfev), nS=info['nS'])
