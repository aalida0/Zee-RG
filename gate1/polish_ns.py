import sys, pickle, numpy as np
sys.path.insert(0, '/home/claude/zee_fit')
from scipy.optimize import least_squares
from zee import hgmodel as H
part = int(sys.argv[1]); S0 = 1e-6; ZM = np.arcsinh(0.1 / S0)
LO, HI = H.LO.copy(), H.HI.copy(); LO[:11] = -ZM; HI[:11] = ZM
xs = np.maximum(HI - LO, 1e-3) * 0.1; xs[:11] = 1.0
def tx(z): x = z.copy(); x[:11] = S0 * np.sinh(z[:11]); return x
fun = lambda z: H.evaluate(tx(z), 'NO')['res']
seeds = []
for f in ('/tmp/nsres300_NO.pkl', '/tmp/nsres_NO.pkl'):
    r = pickle.load(open(f, 'rb')); w = np.exp(r['logwt'] - r['logwt'].max()); w /= w.sum()
    seeds += list(r['samples'][np.random.default_rng(1).choice(len(w), 200, p=w)])
out = []
for x in seeds[part::2]:
    z = np.array(x, float); z[:11] = np.arcsinh(np.clip(z[:11], -.1, .1) / S0)
    try:
        r = least_squares(fun, np.clip(z, LO, HI), bounds=(LO, HI), max_nfev=300, x_scale=xs)
        out.append((float(H.evaluate(tx(r.x), 'NO')['chi2']), tx(r.x)))
    except Exception: pass
out.sort(key=lambda t: t[0]); pickle.dump(out, open(f'/tmp/mnsbest_{part}.pkl', 'wb')); print(part, [round(c, 2) for c, _ in out[:8]], len(out), flush=True)
