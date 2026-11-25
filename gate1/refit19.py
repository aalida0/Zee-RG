"""Minimum chi2 with lambda4 free (19 parameters) for NO and IO with the current likelihood. Seeds: best sample points and
stored best fits. Usage: ZEE_INPUTS=modern python gate1/refit19.py NSEED OUT.pkl"""
import sys, pickle, numpy as np
sys.path.insert(0, '/home/claude/zee_fit')
from scipy.optimize import least_squares
from zee import hgmodel as H
ns = int(sys.argv[1]); S0 = 1e-6; ZM = np.arcsinh(0.1/S0); SQ = np.sqrt(4*np.pi)
LO = np.append(H.LO, -SQ); HI = np.append(H.HI, SQ); LO[:11] = -ZM; HI[:11] = ZM
xs = np.maximum(HI - LO, 1e-3)*0.1; xs[:11] = 1.0
tx = lambda z: np.concatenate([S0*np.sinh(z[:11]), z[11:]])
tz = lambda x: np.concatenate([np.arcsinh(np.clip(np.asarray(x[:11], float), -.1, .1)/S0), np.asarray(list(x[11:19]) + [0.0]*(19 - len(x)), float)])
out = {}
for o in ('NO', 'IO'):
    seeds = []
    for f in (f'/tmp/mbest_{o}.pkl', f'/tmp/mbest2_{o}.pkl', f'/tmp/best_{o}.pkl'):
        try: seeds += [x for _, x in pickle.load(open(f, 'rb'))[:ns]]
        except Exception: pass
    if o == 'NO':
        S = pickle.load(open('/tmp/mfilt4w_NO.pkl', 'rb'))['S']
        ch = sorted([(H.evaluate(list(x), 'NO')['chi2'], x) for _, x in S], key=lambda t: t[0])
        seeds = [x for _, x in ch[:ns]] + seeds
    best = (1e9, None)
    for x in seeds:
        try:
            r = least_squares(lambda z: H.evaluate(tx(z), o)['res'], np.clip(tz(x), LO, HI), bounds=(LO, HI), max_nfev=200, x_scale=xs)
            c = H.evaluate(tx(r.x), o)['chi2']
            if c < best[0]: best = (float(c), tx(r.x))
        except Exception: pass
    out[o] = best; print(o, 'chi2min', round(best[0], 2), flush=True)
pickle.dump(out, open(sys.argv[2], 'wb'))
