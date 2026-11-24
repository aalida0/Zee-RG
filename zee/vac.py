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
def global_ok(c, sc, V=246.22, nstart=40, rng=None):
    """True if no minimum deeper than the EW vacuum (H1=(0,v/sqrt2)) is found. Fields in units of v."""
    rng = rng or np.random.default_rng(0)
    l1, l3, l6, l8 = c[0], c[2], c[5], c[7]
    m11 = -l1/2; m22 = (sc['MHp']**2 - 0.5*V**2*l3)/V**2; m12 = l6/2; mh2 = (sc['M33']**2 - 0.5*V**2*l8)/V**2; mu = sc['mu']/V
    f = lambda p: float(vfull(p, c, m11, m22, m12, mh2, mu))
    vew = -l1/8; tol = 1e-6*max(abs(vew), 1e-3)
    for k in range(nstart):
        p0 = rng.normal(size=10); p0 *= np.exp(rng.uniform(np.log(.1), np.log(30)))/np.linalg.norm(p0)
        with np.errstate(all="ignore"): r = minimize(f, p0, method="L-BFGS-B", options={"maxiter": 500})
        if not np.isfinite(r.fun): continue
        if r.fun < vew - tol: return False, float(r.fun), float(vew)
    return True, None, float(vew)
