"""Independent boundedness-from-below check for REAL quartics by a one-dimensional reduction (cross-check of zee/bfb2.py).
With H1^dag H1 = a, H2^dag H2 = 1-a, z = H1^dag H2 = u + i w, u^2 + w^2 <= R^2 = a(1-a) and real couplings,
  V2 = c0(a) + (l4+l5) u^2 + (l4-l5) w^2 + 2 g(a) u,   c0 = l1 a^2/2 + l2 (1-a)^2/2 + l3 a (1-a),  g = l6 a + l7 (1-a),
  B  = l8 a + l9 (1-a) + 2 l10 u.
For lh > 0 the singlet is eliminated, V_red = V2 - min(0,B)^2/(4 lh).  At fixed a the w-dependence is (l4-l5) w^2: w = 0 if l4 >= l5, else w^2 = R^2 - u^2
(the boundary of the disk).  What remains is piecewise quadratic in u with the breakpoint B = 0, so its minimum is found EXACTLY from the endpoints,
the breakpoint and the vertices of both branches; only the one-dimensional minimisation over a is numerical (dense grid).  For lh = 0 the potential is bounded
iff V2 >= 0 on the set and B >= 0 on it (min_u B = l8 a + l9 (1-a) - 2 |l10| R).  lh < 0: unbounded.  min_vred_1d returns the minimum of V_red (>= 0 required)."""
import numpy as np


def _branch_min(c0, k2, g, R, u_lo, u_hi, extra):
    """minimum over u in [u_lo,u_hi] of c0 + q(u) where q is quadratic: k2 u^2 + 2 g u + extra(u) given as (a2, a1, a0) coefficients added"""
    a2, a1, a0 = extra
    A = k2 + a2; Bc = 2*g + a1; C = c0 + a0
    f = lambda u: C + A*u*u + Bc*u
    cand = [f(u_lo), f(u_hi)]
    with np.errstate(divide='ignore', invalid='ignore'):
        v = np.where(A > 1e-14, -Bc/(2*A), np.nan)
    inside = (v > u_lo) & (v < u_hi)
    cand.append(np.where(inside, f(np.where(inside, v, 0.0)), np.inf))
    m = np.minimum.reduce(cand)
    return np.where(u_hi >= u_lo, m, np.inf)


def min_vred_1d(c, na=4001):
    l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, lh = [float(v) for v in c]
    if lh < 0: return -np.inf
    a = np.linspace(0.0, 1.0, na); R2 = a*(1 - a); R = np.sqrt(R2)
    c0 = 0.5*l1*a**2 + 0.5*l2*(1 - a)**2 + l3*a*(1 - a); g = l6*a + l7*(1 - a); b0 = l8*a + l9*(1 - a)
    lo = -R; hi = R
    # w-branch: l4 >= l5 -> w = 0 (k2 = l4+l5, constant c0); else w^2 = R^2 - u^2 (k2 = 2 l5, constant c0 + (l4-l5) R^2)
    w0 = l4 >= l5
    k2 = (l4 + l5) if w0 else 2*l5
    cc = c0 if w0 else c0 + (l4 - l5)*R2
    best = np.full_like(a, np.inf)
    if lh == 0:
        Bmin = b0 - 2*abs(l10)*R
        v2 = _branch_min(cc, k2, g, R, lo, hi, (0.0, 0.0, 0.0))
        return float(min(v2.min(), np.where(Bmin < -1e-12, -np.inf, np.inf).min(), np.inf)) if (Bmin >= -1e-12).all() else -np.inf
    # B(u) = b0 + 2 l10 u;  B<0 region contributes -B^2/(4 lh)
    if abs(l10) < 1e-14:
        pen = np.where(b0 < 0, -b0**2/(4*lh), 0.0)
        return float(np.min(_branch_min(cc + pen, k2, g, R, lo, hi, (0.0, 0.0, 0.0))))
    u0 = -b0/(2*l10)
    # B < 0  <=>  u < u0 (l10 > 0) or u > u0 (l10 < 0)
    neg_lo, neg_hi, pos_lo, pos_hi = (lo, np.minimum(u0, hi), np.maximum(u0, lo), hi) if l10 > 0 else (np.maximum(u0, lo), hi, lo, np.minimum(u0, hi))   # empty intervals (hi < lo) give +inf
    pen = (-4*l10**2/(4*lh), -2*2*l10*b0/(4*lh), -b0**2/(4*lh))      # -(b0 + 2 l10 u)^2/(4 lh) as (a2, a1, a0)
    vneg = _branch_min(cc, k2, g, R, neg_lo, neg_hi, pen)
    vpos = _branch_min(cc, k2, g, R, pos_lo, pos_hi, (0.0, 0.0, 0.0))
    return float(np.minimum(vneg, vpos).min())
