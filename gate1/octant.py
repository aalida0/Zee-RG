import sys, pickle, time, os, numpy as np
sys.path.insert(0, '/home/claude/zee_fit')
from scipy.optimize import least_squares
from zee import hgmodel as H
seed, out = int(sys.argv[1]), sys.argv[2]; files = sys.argv[3:]
base = []
for f in files: base += pickle.load(open(f, 'rb'))
rng = np.random.default_rng(seed); LO, HI = H.LO, H.HI; xs = np.maximum(HI - LO, 1e-3) * 0.1
def fun(x, w):
    e = H.evaluate(x, 'IO'); s23 = 0.587 + e['osc'][2] * 0.024
    return np.concatenate([e['res'], [w * max(0.0, 0.5 - s23) / 0.024]])
res = []; t0 = time.time(); n = 0
while time.time() - t0 < float(os.environ.get('TLIM', '480')):
    c, x = base[rng.integers(len(base))]; x0 = x.copy()
    if rng.random() < 0.7:
        x0[:11] *= np.exp(rng.normal(0, 0.5, 11)); x0[11] += rng.normal(0, 0.4)
        x0[12:15] += rng.normal(0, 0.2, 3)
    try:
        r = least_squares(fun, np.clip(x0, LO, HI), args=(3.0,), bounds=(LO, HI), max_nfev=150, x_scale=xs)
        r = least_squares(fun, r.x, args=(1.0,), bounds=(LO, HI), max_nfev=150, x_scale=xs)
        e = H.evaluate(r.x, 'IO'); s23 = 0.587 + e['osc'][2] * 0.024
        res.append((float(e['chi2']), r.x, float(s23)))
    except Exception: pass
    n += 1
    if n % 10 == 0:
        res.sort(key=lambda t: t[0]); pickle.dump(res[:60], open(out, 'wb'))
        print(n, round(time.time() - t0), [(round(a, 1), round(s, 2)) for a, _, s in res[:4]], flush=True)
res.sort(key=lambda t: t[0]); pickle.dump(res[:60], open(out, 'wb')); print('done', flush=True)
