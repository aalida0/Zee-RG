"""One-loop threshold matching of the Higgs-basis boundary quartics in the effective-potential approximation (report v10).

Given the tree-level parameters p0 of a low-energy point (the spectrum and mixings of the fit are read as the PHYSICAL inputs), find the MSbar parameters
p = (m11^2, m22^2, m12^2, mh^2, mu; lambda_1, lambda_4, lambda_5, lambda_6) at the matching scale mu0 such that the one-loop effective potential
V_eff = V_tree(p) + V_1(p; mu0) (Landau gauge, MSbar; scalar, W/Z and top loops) has its minimum at (v, 0) with the SAME scalar mass matrix as the tree potential of p0.
lambda_2, lambda_3, lambda_7, lambda_8, lambda_9, lambda_10, lambda_h are not fixed by the spectrum and are kept at their input values (read as MSbar inputs at mu0).

Approximations (the calculation is an ESTIMATE of the size of the one-loop threshold corrections, not a complete matching): (i) zero-momentum self-energies, i.e. the second
derivatives of V_1 instead of Pi(p^2 = m^2): the momentum dependence of the loop functions is neglected; (ii) the Goldstone logarithm is regulated, ln|x| -> ln sqrt(x^2 + X0^2) with
X0 = kappa m_h^2 (kappa = 1; 1/4 and 4 as a systematic); (iii) bottom, tau and lighter fermions and the photon are neglected; (iv) v = 246.22 GeV is held fixed.
"""
import numpy as np
from scipy.optimize import root
import itertools, os, sys
from . import rgeq

V = 246.22; MH = 125.5      # MH = the light-state mass of the model input (hgmodel.MH, HG convention); it also sets the Goldstone-log regulator X0 = kappa MH^2 of the EP matching (a regulator scale, varied through kappa)
K16 = 1/(64*np.pi**2)


def _vac():
    from . import vac as _v            # tensor of fourth derivatives of the quartic form
    return _v


def quartic_tensor(c):
    vac = _vac()
    F = (vac.v4(c, vac._ARG)*vac._SG).sum(-1)/8
    T = np.zeros(10**4); T[vac._dst] = F[vac._src]
    return T.reshape((10,)*4)


def tree_params(c, sc):
    l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, lh = c
    return dict(m11=-l1*V**2/2, m22=sc['MHp']**2 - 0.5*V**2*l3, m12=l6*V**2/2, mh2=sc['M33']**2 - 0.5*V**2*l8, mu=sc['mu'])


def vfull_(p, c, P):
    return _vac().vfull(p, c, P['m11'], P['m22'], P['m12'], P['mh2'], P['mu'])


def mass_tensors(c, P):
    """M^2(phi) = mu2 + kappa.phi + 1/2 lam.phi.phi (exact for the polynomial potential); mu2 and kappa from exact five-point derivatives of V_tree"""
    vac = _vac(); E = np.eye(10)
    def hess(phi, h=1.0):
        # exact for polynomials up to degree 5 in each direction: f'' = (-f(2h)+16f(h)-30f+16f(-h)-f(-2h))/(12 h^2); mixed via polarisation
        def d2(d):
            f = lambda s: vfull_(phi + s*h*d, c, P)
            return (-f(2) + 16*f(1) - 30*f(0) + 16*f(-1) - f(-2))/(12*h*h)
        H = np.zeros((10, 10)); dd = np.array([d2(E[i]) for i in range(10)])
        for i in range(10):
            H[i, i] = dd[i]
            for j in range(i + 1, 10): H[i, j] = H[j, i] = (d2(E[i] + E[j]) - dd[i] - dd[j])/2
        return H
    mu2 = hess(np.zeros(10)); kap = np.zeros((10, 10, 10)); h = 10.0
    for k in range(10):
        kap[:, :, k] = (hess(h*E[k]) - hess(-h*E[k]))/(2*h)
    return mu2, kap, quartic_tensor(c)


def mass2_at(phi, T):
    mu2, kap, lam = T
    return mu2 + kap @ phi + 0.5*np.einsum('abcd,c,d->ab', lam, phi, phi)


_GENS = {}


def vector_masses2(phi, g1, g2):
    key = (g1, g2)
    if key not in _GENS: _GENS[key] = rgeq._gens(g1, g2)
    th = _GENS[key]; w = np.array([t @ phi for t in th])
    return np.linalg.eigvalsh(w @ w.T)


def top_mass2(phi, yt, rt):
    H1p, H10 = (phi[0] + 1j*phi[1])/np.sqrt(2), (phi[2] + 1j*phi[3])/np.sqrt(2)
    H2p, H20 = (phi[4] + 1j*phi[5])/np.sqrt(2), (phi[6] + 1j*phi[7])/np.sqrt(2)
    return abs(yt*H10 + rt*H20)**2 + abs(yt*H1p + rt*H2p)**2


def f_(x, lnmu2, X0, c):
    r = np.sqrt(x*x + X0*X0)
    return x*x*(np.log(r) - lnmu2 - c)


def V1(phi, T, mu0, g1, g2, yt, rt, X0, parts='SVT'):
    """parts: S scalars (10 real eigenvalues), V gauge bosons (3 polarisations), T top (12 dof); used for diagnostics"""
    lnmu2 = 2*np.log(mu0); out = 0.0
    if 'S' in parts: out += f_(np.linalg.eigvalsh(mass2_at(phi, T)), lnmu2, X0, 1.5).sum()
    if 'V' in parts: out += 3*f_(vector_masses2(phi, g1, g2), lnmu2, 1e-6, 5/6).sum()
    if 'T' in parts: out -= 12*f_(top_mass2(phi, yt, rt), lnmu2, 1e-6, 1.5)
    return K16*out


def derivs(fun, phi0, h=1.5):
    """gradient and Hessian of fun at phi0 by exact-for-degree-4 five-point stencils (the loop function is smooth on the scale h)"""
    E = np.eye(10); f0 = fun(phi0)
    def d1(d):
        return (-fun(phi0 + 2*h*d) + 8*fun(phi0 + h*d) - 8*fun(phi0 - h*d) + fun(phi0 - 2*h*d))/(12*h)
    def d2(d):
        return (-fun(phi0 + 2*h*d) + 16*fun(phi0 + h*d) - 30*f0 + 16*fun(phi0 - h*d) - fun(phi0 - 2*h*d))/(12*h*h)
    g = np.array([d1(E[i]) for i in range(10)]); dd = np.array([d2(E[i]) for i in range(10)]); H = np.diag(dd)
    for i in range(10):
        for j in range(i + 1, 10): H[i, j] = H[j, i] = (d2(E[i] + E[j]) - dd[i] - dd[j])/2
    return g, H


# unknown vector layout: m11, m22, m12, mh2, mu, l1, l4, l5, l6
IDX = [2, 6]                              # tadpole components: Re H1^0, Re H2^0
H_IDX = [(2, 2), (2, 6), (6, 6), (7, 7), (4, 4), (4, 8), (8, 8)]


def _unpack(u, c0):
    P = dict(m11=u[0], m22=u[1], m12=u[2], mh2=u[3], mu=u[4]); c = np.array(c0, float).copy(); c[0], c[3], c[4], c[5] = u[5], u[6], u[7], u[8]
    return P, c


def match(c0, sc, mu0, g, yt, r, kappa=1.0, tol=1e-9, verbose=False, no_loop=False, parts='SVT'):
    """return dict(c=matched quartics, delta=c-c0, P=matched mass parameters, resid). g = (g', g, g3), yt = y_t(mu0), rho_t = r yt."""
    phi0 = np.zeros(10); phi0[2] = V; X0 = kappa*MH**2; rt = r*yt
    P0 = tree_params(c0, sc); T0 = mass_tensors(c0, P0)
    Mt = mass2_at(phi0, T0)                                              # target: tree mass matrix of the input point
    u0 = np.array([P0['m11'], P0['m22'], P0['m12'], P0['mh2'], P0['mu'], c0[0], c0[3], c0[4], c0[5]])
    scale = np.array([V**2, V**2, V**2, V**2, V, 1, 1, 1, 1])

    def resid(w):
        u = u0 + w*scale; P, c = _unpack(u, c0); T = mass_tensors(c, P)
        gt = _vac().vgrad(phi0, c, P['m11'], P['m22'], P['m12'], P['mh2'], P['mu'])[1]
        Ht = mass2_at(phi0, T)
        if no_loop: g1l, H1l = np.zeros(10), np.zeros((10, 10))
        else: g1l, H1l = derivs(lambda ph: V1(ph, T, mu0, g[0], g[1], yt, rt, X0, parts), phi0)
        R = [(gt + g1l)[i]/V**3 for i in IDX] + [(Ht + H1l - Mt)[i, j]/V**2 for i, j in H_IDX]
        return np.array(R)
    sol = root(resid, np.zeros(9), method='hybr', tol=tol)
    u = u0 + sol.x*scale; P, c = _unpack(u, c0)
    return dict(c=c, delta=c - np.asarray(c0, float), P=P, ok=bool(sol.success), resid=float(np.abs(resid(sol.x)).max()), nfev=int(sol.nfev))
