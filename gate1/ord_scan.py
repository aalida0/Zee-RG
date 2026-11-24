"""Ordering preference with different external inputs. Usage: ZEE_INPUTS=modern python ord_scan.py ORD NSEED
Variants: lab+DESI2024 (0.072), lab+DESIDR2 (0.0642), lab only (no Sum m bound), oscillation chi2 only."""
import sys, pickle, time, numpy as np
sys.path.insert(0, '/home/claude/zee_fit')
from scipy.optimize import least_squares
from zee import hgmodel as H
o = sys.argv[1]; ns = int(sys.argv[2]); S0 = 1e-6; ZM = np.arcsinh(0.1 / S0)
LO, HI = H.LO.copy(), H.HI.copy(); LO[:11] = -ZM; HI[:11] = ZM
xs = np.maximum(HI - LO, 1e-3) * 0.1; xs[:11] = 1.0
def tx(z): x = z.copy(); x[:11] = S0 * np.sinh(z[:11]); return x
seeds = [x for _, x in pickle.load(open(f'/tmp/mbest_{o}.pkl', 'rb'))[:ns]]
VAR = {'DESI2024_0.072': dict(sum_m=(0.072, 1.949)), 'DESIDR2_0.0642': dict(sum_m=(0.0642, 1.949)), 'no_cosmo': dict(sum_m=(1e3, 1.949)), 'osc_only': None}
base = dict(H.BOUNDS)
res = {}
for name, upd in VAR.items():
    H.BOUNDS.clear(); H.BOUNDS.update(base)
    if upd is None:
        for k in base: H.BOUNDS[k] = (1e9, base[k][1])
        fun = lambda z: np.array([np.sqrt(max(H.evaluate(tx(z), o)['chi_osc'], 0.0))]); getchi = lambda x: H.evaluate(x, o)['chi_osc']
    else:
        H.BOUNDS.update(upd); fun = lambda z: H.evaluate(tx(z), o)['res']; getchi = lambda x: H.evaluate(x, o)['chi2']
    best = 1e9; t0 = time.time()
    for x in seeds:
        z = np.array(x, float); z[:11] = np.arcsinh(np.clip(z[:11], -.1, .1) / S0)
        try:
            r = least_squares(fun, np.clip(z, LO, HI), bounds=(LO, HI), max_nfev=200, x_scale=xs); best = min(best, float(getchi(tx(r.x))))
        except Exception as e: pass
    res[name] = best; print(o, name, round(best, 2), round(time.time() - t0), 's', flush=True)
pickle.dump(res, open(f'/tmp/ordscan_{o}.pkl', 'wb'))
