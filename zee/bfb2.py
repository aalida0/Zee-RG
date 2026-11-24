"""Boundedness from below of the Zee quartic potential by exact elimination of the singlet direction.
Gauge-invariant variables x=H1^dag H1, y=H2^dag H2, z=H1^dag H2 (|z|^2<=xy), s=|h+|^2>=0:
V4 = V2(x,y,z) + B s + lh s^2,  B = l8 x + l9 y + 2 Re(l10 z).
For lh>0, min over s>=0 gives  V_red = V2 - min(0,B)^2/(4 lh); the potential is bounded from below iff V_red>=0 on the compact
set {x=a, y=1-a, z=sqrt(a(1-a)) rho e^{i phi}} (a, rho in [0,1], phi in [0,2pi)), found by a dense grid plus local refinement.
For lh=0 we require B>=0 on that set and V2>=0 (sufficient; the boundary case is not treated)."""
import numpy as np
from scipy.optimize import minimize
_A = np.linspace(0, 1, 81); _R = np.linspace(0, 1, 41); _P = np.linspace(0, 2*np.pi, 73)[:-1]
_G = np.meshgrid(_A, _R, _P, indexing='ij'); _G = np.stack([g.ravel() for g in _G], -1)
def vred(c, q):
    l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, lh = c
    a, r, ph = q[..., 0], q[..., 1], q[..., 2]; x = a; y = 1 - a
    z = np.sqrt(np.clip(a*(1-a), 0, None)) * r * np.exp(1j*ph)
    v2 = 0.5*l1*x**2 + 0.5*l2*y**2 + l3*x*y + l4*abs(z)**2 + (l5*z**2).real + 2*((l6*x + l7*y)*z).real
    B = l8*x + l9*y + 2*(l10*z).real
    if lh > 0: return v2 - np.minimum(B, 0)**2/(4*lh)
    return np.where(B < -1e-12, -1e9, v2)
def min_vred(c, refine=4):
    v = vred(c, _G); idx = np.argsort(v)[:refine]; best = v.min()
    for i in idx:
        q0 = _G[i]
        f = lambda q: vred(c, np.array([np.clip(q[0], 0, 1), np.clip(q[1], 0, 1), q[2]]))
        r = minimize(f, q0, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-12, 'maxiter': 300})
        best = min(best, r.fun)
    return float(best)
