"""Two-loop Barr-Zee loop functions, tabulated once in log z (Omura-Senaha-Tobe 1511.08880 Eqs. 22-26;
Ilisie 1502.04199 Eq. 27). Integrands are written in forms that are finite where x(1-x) = z.
    F_H(z) = z/2 int (1-2x(1-x)) ln(x(1-x)/z)/(x(1-x)-z)
    F_A(z) = z/2 int ln(x(1-x)/z)/(x(1-x)-z)
    G(z)   = -z/2 int 1/(x(1-x)-z) [1 - z/(x(1-x)-z) ln(x(1-x)/z)]
    F2(w)  = 1/2 int x(x-1)/(w-x(1-x)) ln(w/(x(1-x)))      (charged-scalar loop)
"""
import os
import numpy as np
from scipy.integrate import quad

_LZ = np.linspace(-8.0, 6.0, 1401)        # log10 z grid, 0.01 dex
_CACHE = os.path.join(os.path.dirname(__file__), "_loops.npz")


def _lr(a, b):
    """ln(a/b)/(a-b), stable for a ~ b."""
    d = a - b
    if abs(d) < 1e-7 * b:
        u = d / b
        return (1 - u / 2 + u * u / 3) / b
    return np.log(a / b) / d


def _h(u):
    """(1 - ln(1+u)/u)/u, stable for small u."""
    if abs(u) < 1e-4:
        return 0.5 - u / 3 + u * u / 4
    return (1 - np.log1p(u) / u) / u


def _quad(f, z):
    pts = []
    if z < 0.25:
        r = np.sqrt(0.25 - z); pts = [0.5 - r, 0.5 + r]
    return quad(f, 0, 1, points=pts or None, limit=400, epsabs=0, epsrel=1e-10)[0]


def FH_exact(z):
    return z / 2 * _quad(lambda x: (1 - 2 * x * (1 - x)) * _lr(x * (1 - x), z), z)


def FA_exact(z):
    return z / 2 * _quad(lambda x: _lr(x * (1 - x), z), z)


def G_exact(z):
    return -0.5 * _quad(lambda x: _h((x * (1 - x) - z) / z), z)


def F2_exact(w):
    return 0.5 * _quad(lambda x: x * (x - 1) * _lr(w, x * (1 - x)), w)


def _build():
    import warnings
    z = 10.0 ** _LZ
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        T = np.array([[fn(v) for v in z] for fn in (FH_exact, FA_exact, G_exact, F2_exact)])
    try: np.savez(_CACHE, lz=_LZ, T=T, src=_SRC)
    except OSError: pass                              # read-only checkout: keep the in-memory table
    return T


import hashlib, inspect
_SRC = hashlib.sha256("".join(inspect.getsource(f) for f in (_lr, _h, _quad, FH_exact, FA_exact, G_exact, F2_exact)).encode()).hexdigest()[:16]


try:
    _d = np.load(_CACHE)
    _T = _d["T"] if (np.array_equal(_d["lz"], _LZ) and str(_d["src"]) == _SRC) else _build()   # table tied to the integrands, not only the grid
except (OSError, KeyError):
    _T = _build()


def _interp(k, z):
    lz = np.log10(z)
    if np.any(lz < _LZ[0]) or np.any(lz > _LZ[-1]):
        raise ValueError("loop-function argument outside table: %r" % (z,))
    return np.interp(lz, _LZ, _T[k])


FH = lambda z: _interp(0, z)
FA = lambda z: _interp(1, z)
G = lambda z: _interp(2, z)
F2 = lambda w: _interp(3, w)
def _dd(k, x, y):
    """(x f(y) - y f(x))/(x - y) for f = table k.  The direct form amplifies the table interpolation error as 1/(x-y); for
    |x - y| < 2e-2 x we use the Taylor form f - x f' - x f'' (y - x)/2 (f', f'' by central differences in log10 z)."""
    x = np.asarray(x, float); y = np.asarray(y, float); f = lambda z: _interp(k, z)
    close = np.abs(x - y) < 2e-2 * np.maximum(np.abs(x), np.abs(y))
    z = np.where(close, x, 1.0); h = 0.05; a, b = 10**h, 10**-h
    f0, fp, fm = f(z), f(z*a), f(z*b)
    # derivatives w.r.t. z from the three-point stencil in u = log10 z:  f_u = (fp-fm)/(2h), f_uu = (fp-2f0+fm)/h^2
    fu = (fp - fm)/(2*h); fuu = (fp - 2*f0 + fm)/h**2; L10 = np.log(10)
    d1 = fu/(L10*z); d2 = (fuu - fu)/(L10**2*z**2) + 0.0*z
    d2 = (fuu/L10**2 - fu/L10)/z**2
    taylor = f0 - z*d1 - 0.5*z*d2*(y - x)
    return np.where(close, taylor, (x * f(y) - y * f(x)) / np.where(close, 1.0, x - y))


FHt = lambda x, y: _dd(0, x, y)
FAt = lambda x, y: _dd(1, x, y)
