"""Minimum chi2 for NO with theta23 forced into the upper octant (penalty for s23^2 < 0.5). Usage: ZEE_INPUTS=modern python octant_no.py NSEED"""
import sys, pickle, numpy as np
_R = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
sys.path.insert(0, _R)
from scipy.optimize import least_squares
from zee import hgmodel as H, neutrino as N
S0 = 1e-6; ZM = np.arcsinh(0.1/S0); LO, HI = H.LO.copy(), H.HI.copy(); LO[:11] = -ZM; HI[:11] = ZM
xs = np.maximum(HI - LO, 1e-3)*0.1; xs[:11] = 1.0
def tx(z): x = z.copy(); x[:11] = S0*np.sinh(z[:11]); return x
def s23(x):
    f, Y = H.unpack(x); sc = H.scalar_sector(x); M = N.mnu_hg_leading(f, H.M_LEP, Y, sc['tb'], sc['m1'], sc['m2'], sc['mu'])
    return N.oscillation_observables(M, 'NO')['s23sq']
best = 1e9
for c, x in pickle.load(open("/tmp/mbest_NO.pkl", "rb"))[:int(sys.argv[1])]:
    z = np.array(x[:18], float); z[:11] = np.arcsinh(np.clip(z[:11], -.1, .1)/S0)
    try:
        r = least_squares(lambda z: np.append(H.evaluate(tx(z), 'NO')['res'], max(0, .5 - s23(tx(z)))/0.002), np.clip(z, LO, HI), bounds=(LO, HI), max_nfev=200, x_scale=xs)
        x1 = tx(r.x); c1 = H.evaluate(x1, 'NO')['chi2']; s = s23(x1)
        if s >= 0.5 and c1 < best: best = c1; print('upper octant chi2', round(c1, 2), 's23sq', round(s, 3), flush=True)
    except Exception: pass
print('best upper-octant chi2', best)
