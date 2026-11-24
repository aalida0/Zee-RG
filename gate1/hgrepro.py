"""Controlled reproduction of HG: refit the minimum chi2 for NO and IO with one more modern ingredient switched on per step.
Usage: ZEE_ING=osc,lfv,... python hgrepro.py NSEED   (ingredients: osc, lfv, cr, cosmo, amu, univ)"""
import sys, os, pickle, time, numpy as np
sys.path.insert(0, '/home/claude/zee_fit')
from scipy.optimize import least_squares
from zee import hgmodel as H
ns = int(sys.argv[1]); S0 = 1e-6; ZM = np.arcsinh(0.1/S0)
LO, HI = H.LO.copy(), H.HI.copy(); LO[:11] = -ZM; HI[:11] = ZM
xs = np.maximum(HI - LO, 1e-3)*0.1; xs[:11] = 1.0
def tx(z): x = z.copy(); x[:11] = S0*np.sinh(z[:11]); return x
out = {}
for o in ('NO', 'IO'):
    seeds = [x for _, x in pickle.load(open(f'/tmp/best_{o}.pkl', 'rb'))[:ns]] + [x for _, x in pickle.load(open(f'/tmp/mbest_{o}.pkl', 'rb'))[:ns]]
    best = (1e9, None); t0 = time.time()
    for x in seeds:
        z = np.array(x[:18], float); z[:11] = np.arcsinh(np.clip(z[:11], -.1, .1)/S0)
        try:
            r = least_squares(lambda z: H.evaluate(tx(z), o)['res'], np.clip(z, LO, HI), bounds=(LO, HI), max_nfev=150, x_scale=xs)
            c = H.evaluate(tx(r.x), o)['chi2']
            if c < best[0]: best = (float(c), tx(r.x))
        except Exception: pass
    e = H.evaluate(best[1], o); sc = e['sc']
    out[o] = dict(chi2=best[0], tb=sc['tb'], mA=sc['mA'], m1=sc['m1'], tmg=e['obs']['tau_mu_g'], meg=e['obs']['mu_e_g'], chi_osc=e['chi_osc'])
    print(os.environ.get('ZEE_ING', ''), o, {k: float('%.3g' % v) for k, v in out[o].items()}, round(time.time() - t0), 's', flush=True)
pickle.dump(out, open('/tmp/hgrepro_%s.pkl' % (os.environ.get('ZEE_ING', 'none').replace(',', '+') or 'none'), 'wb'))
