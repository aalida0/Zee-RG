import sys, pickle, time, os, numpy as np
sys.path.insert(0, '/home/claude/zee_fit')
from scipy.optimize import least_squares
from zee import hgmodel as H
ordering, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rng = np.random.default_rng(seed); LO, HI = H.LO.copy(), H.HI.copy(); S0 = 1e-6; ZM = np.arcsinh(0.1 / S0)
def to_x(z):
    x = z.copy(); x[:11] = S0 * np.sinh(z[:11]); return x
lo = LO.copy(); hi = HI.copy(); lo[:11] = -ZM; hi[:11] = ZM
xs = np.maximum(hi - lo, 1e-3) * 0.1; xs[:11] = 1.0
def fun(z): return H.evaluate(to_x(z), ordering)['res']
res = []; t0 = time.time(); n = 0
while time.time() - t0 < float(os.environ.get('TLIM', '540')):
    z = np.zeros(18)
    z[:11] = rng.choice([-1, 1], 11) * np.arcsinh(10 ** rng.uniform(-6, -1, 11) / S0)
    z[11] = rng.uniform(np.log(.3), np.log(3.0)); z[12:14] = rng.uniform(2.0, 3.5, 2); z[14] = rng.uniform(0.01, 0.9)
    z[15] = rng.uniform(-1, 3); z[16] = rng.uniform(-2, 2); z[17] = rng.uniform(-.3, .3)
    try:
        r = least_squares(fun, np.clip(z, lo, hi), bounds=(lo, hi), max_nfev=250, x_scale=xs)
        res.append((float(H.evaluate(to_x(r.x), ordering)['chi2']), to_x(r.x)))
    except Exception: pass
    n += 1
    if n % 10 == 0:
        res.sort(key=lambda t: t[0]); pickle.dump(res[:100], open(out, 'wb'))
        print(n, round(time.time() - t0), [round(c, 1) for c, _ in res[:5]], flush=True)
res.sort(key=lambda t: t[0]); pickle.dump(res[:100], open(out, 'wb')); print('done', flush=True)
