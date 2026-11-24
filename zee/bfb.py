import numpy as np
from scipy.optimize import minimize
def v4(c, p):
    """quartic form of the HG potential on gauge-orbit representatives, normalised by (sum|z|^2)^2.
    p: (...,6) = a, b1, b2, phi, e, chi (H1=(0,a), H2=(b1,b2 e^{i phi}), h=e e^{i chi})."""
    l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, lh = c
    a, b1, b2, ph, e, ch = (p[..., i] for i in range(6))
    a11 = a**2; a22 = b1**2 + b2**2; hh = e**2
    a12 = a * b2 * np.exp(1j * ph)            # H1^dag H2
    v = (0.5 * l1 * a11**2 + 0.5 * l2 * a22**2 + l3 * a11 * a22 + l4 * abs(a12)**2 + (l5 * a12**2).real
         + 2 * ((l6 * a11 + l7 * a22) * a12).real + lh * hh**2 + l8 * hh * a11 + l9 * hh * a22 + 2 * (l10 * hh * a12).real)
    return v / (a11 + a22 + hh)**2
def min_v4(c, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    P = np.column_stack([rng.random((n, 3)), rng.uniform(0, 2 * np.pi, n), rng.random(n), rng.uniform(0, 2 * np.pi, n)])
    P[:, :3] = P[:, :3] ** 1.0
    P = np.vstack([P, [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]]])
    v = v4(c, P); idx = np.argsort(v)[:3]; best = v.min()
    for i in idx:
        r = minimize(lambda q: v4(c, q), P[i], method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-10, 'maxiter': 400})
        best = min(best, r.fun)
    return best
