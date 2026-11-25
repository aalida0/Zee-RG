"""Tree-level perturbative unitarity and global-minimum (charge-breaking / deeper vacuum) test for the Zee potential.
Fields: H1=(p0+ip1, p2+ip3)/sqrt2, H2=(p4+ip5, p6+ip7)/sqrt2, h+=(p8+ip9)/sqrt2 (canonical real fields).
Quartics c=(l1..l10, lh) as in bfb2: V4 = l1/2 x^2 + l2/2 y^2 + l3 xy + l4|z|^2 + Re(l5 z^2) + 2Re((l6 x + l7 y) z)
 + s (l8 x + l9 y + 2Re(l10 z)) + lh s^2,  x=|H1|^2, y=|H2|^2, z=H1^dag H2, s=|h|^2."""
import itertools, numpy as np
from scipy.optimize import minimize
def _cplx(p):
    H1 = np.stack([p[..., 0] + 1j*p[..., 1], p[..., 2] + 1j*p[..., 3]], -1)/np.sqrt(2)
    H2 = np.stack([p[..., 4] + 1j*p[..., 5], p[..., 6] + 1j*p[..., 7]], -1)/np.sqrt(2)
    h = (p[..., 8] + 1j*p[..., 9])/np.sqrt(2); return H1, H2, h
def v4(c, p):
    l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, lh = c; H1, H2, h = _cplx(p)
    x = (abs(H1)**2).sum(-1); y = (abs(H2)**2).sum(-1); z = (H1.conj()*H2).sum(-1); s = abs(h)**2
    return (0.5*l1*x**2 + 0.5*l2*y**2 + l3*x*y + l4*abs(z)**2 + (l5*z**2).real + 2*((l6*x + l7*y)*z).real
            + s*(l8*x + l9*y + 2*(l10*z).real) + lh*s**2)
_Q = list(itertools.combinations_with_replacement(range(10), 4)); _E = np.eye(10)
_ARG = np.array([[_E[i] + a*_E[j] + b*_E[k] + d*_E[l] for a in (1, -1) for b in (1, -1) for d in (1, -1)] for i, j, k, l in _Q])
_SG = np.array([a*b*d for a in (1, -1) for b in (1, -1) for d in (1, -1)])
_PAIRS = list(itertools.combinations_with_replacement(range(10), 2)); _PI = {p: n for n, p in enumerate(_PAIRS)}
_src, _dst = [], []
for _n, _q in enumerate(_Q):
    for _p in set(itertools.permutations(_q)): _src.append(_n); _dst.append(np.ravel_multi_index(_p, (10,)*4))
_src = np.array(_src); _dst = np.array(_dst)
_PA = np.array(_PAIRS); _N = np.where(_PA[:, 0] == _PA[:, 1], 1/np.sqrt(2), 1.0); _NN = np.outer(_N, _N)
_IDX = np.ravel_multi_index((_PA[:, 0][:, None], _PA[:, 1][:, None], _PA[:, 0][None, :], _PA[:, 1][None, :]), (10,)*4)
def unit_max(c):
    """largest |eigenvalue| of the s-wave 2->2 matrix of the quartics (unitarity: <= 8 pi)."""
    F = (v4(c, _ARG)*_SG).sum(-1)/8          # d^4 V / dp_i dp_j dp_k dp_l  (= 24 * polarised form / 192 * 24 ... = sum/8)
    T = np.zeros(10**4); T[_dst] = F[_src]
    return float(abs(np.linalg.eigvalsh(-T[_IDX]*_NN)).max())
def vfull(p, c, m11, m22, m12, mh2, mu):
    H1, H2, h = _cplx(p)
    x = (abs(H1)**2).sum(-1); y = (abs(H2)**2).sum(-1); z = (H1.conj()*H2).sum(-1); s = abs(h)**2
    v3 = 2*(mu*(H1[..., 0]*H2[..., 1] - H1[..., 1]*H2[..., 0])*np.conj(h)).real
    return m11*x + m22*y - 2*(m12*z).real + mh2*s + v3 + v4(c, p)
def vgrad(p, c, m11, m22, m12, mh2, mu):
    """V and dV/dp for one real 10-vector p (analytic; Wirtinger derivatives, dV/dp_re = sqrt2 Re dV/dphibar,
    dV/dp_im = sqrt2 Im dV/dphibar)."""
    l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, lh = c
    H1, H2, h = _cplx(p)
    x = (abs(H1)**2).sum(); y = (abs(H2)**2).sum(); z = (H1.conj()*H2).sum(); s = abs(h)**2
    E = H1[0]*H2[1] - H1[1]*H2[0]
    Vx = m11 + l1*x + l3*y + 2*(l6*z).real + l8*s
    Vy = m22 + l2*y + l3*x + 2*(l7*z).real + l9*s
    Vz = -m12 + l4*np.conj(z) + l5*z + l6*x + l7*y + l10*s
    Vs = mh2 + l8*x + l9*y + 2*(l10*z).real + 2*lh*s
    g1 = Vx*H1 + Vz*H2 + mu*h*np.array([np.conj(H2[1]), -np.conj(H2[0])])
    g2 = Vy*H2 + np.conj(Vz)*H1 + mu*h*np.array([-np.conj(H1[1]), np.conj(H1[0])])
    gh = Vs*h + mu*E
    w = np.array([g1[0], g1[1], g2[0], g2[1], gh])
    grad = np.sqrt(2)*np.stack([w.real, w.imag], -1).ravel()
    val = (m11*x + m22*y - 2*(m12*z).real + mh2*s + 2*(mu*E*np.conj(h)).real
           + 0.5*l1*x**2 + 0.5*l2*y**2 + l3*x*y + l4*abs(z)**2 + (l5*z**2).real + 2*((l6*x + l7*y)*z).real
           + s*(l8*x + l9*y + 2*(l10*z).real) + lh*s**2)
    return float(val), grad


def _starts(rng, nstart):
    """deterministic directions (EW vacuum, H2-only, singlet-only, charge-breaking H1+/H2+/h+ mixtures) and random ones"""
    E = np.eye(10); P = []
    radii = (0.5, 1.0, 3.0) if nstart >= 20 else (1.0,)
    for d in (E[2], E[6], E[8], E[2] + E[4], E[2] + E[8], E[6] + E[8], E[0] + E[6], E[2] + E[6] + E[8], E[2] - E[6] + E[8]):
        for r in radii: P.append(d*r/np.linalg.norm(d))
    for k in range(nstart):
        p0 = rng.normal(size=10); P.append(p0*np.exp(rng.uniform(np.log(.1), np.log(30)))/np.linalg.norm(p0))
    return P


def global_search(c, sc, V=246.22, nstart=40, rng=None, gtol=1e-7):
    """Search for a minimum deeper than the EW vacuum (fields in units of v). Fails closed:
    status 'deeper' (any start ends below V_EW - tol, stationary or not, disproves globality),
    'unresolved' (fewer than half of the starts converge to a finite stationary point),
    'ok' (no deeper point found by the stated search). Returns a dict with counts and the deepest point."""
    rng = rng or np.random.default_rng(0)
    l1, l3, l6, l8 = c[0], c[2], c[5], c[7]
    args = (c, -l1/2, (sc['MHp']**2 - 0.5*V**2*l3)/V**2, l6/2, (sc['M33']**2 - 0.5*V**2*l8)/V**2, sc['mu']/V)
    vew = -l1/8; tol = 1e-6*max(abs(vew), 1e-3)
    pew = np.zeros(10); pew[2] = 1.0
    _, gew = vgrad(pew, *args)
    out = dict(V_EW=float(vew), n=0, n_conv=0, n_fail=0, deepest=np.inf, p_deepest=None, ew_grad=float(np.abs(gew).max()))
    for p0 in _starts(rng, nstart):
        out['n'] += 1
        with np.errstate(all='ignore'):
            r = minimize(vgrad, p0, args=args, jac=True, method='L-BFGS-B', options={'maxiter': 3000, 'gtol': gtol, 'ftol': 1e-15})
        if not np.isfinite(r.fun) or not np.all(np.isfinite(r.x)):
            out['n_fail'] += 1; continue
        if r.fun < out['deepest']: out['deepest'] = float(r.fun); out['p_deepest'] = r.x.tolist()
        if r.fun < vew - tol:
            out['status'] = 'deeper'; return out
        if np.abs(r.jac).max() < 1e-5: out['n_conv'] += 1
        else: out['n_fail'] += 1
    out['status'] = 'ok' if (out['n_conv'] >= out['n']/2 and out['ew_grad'] < 1e-8) else 'unresolved'
    return out


def global_ok(c, sc, V=246.22, nstart=40, rng=None):
    """(ok, deepest, V_EW): ok only when global_search returns 'ok' (fail closed)."""
    r = global_search(c, sc, V=V, nstart=nstart, rng=rng)
    return r['status'] == 'ok', r['deepest'], r['V_EW']
