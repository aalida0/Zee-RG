import sys, pickle, time, numpy as np
sys.path.insert(0, '/home/claude/zee_fit')
from scipy.optimize import least_squares, brentq
from zee import hgmodel as H, scalars as S
ordering, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
files = sys.argv[4:]
V = 246.21965
base = []
for f in files:
    base += [(c, y) for c, y in pickle.load(open(f, 'rb')) if c < 0.5]
rng = np.random.default_rng(seed)
LO, HI = H.LO, H.HI
def fun(x):
    return H.evaluate(x, ordering)['res']
def make_start(y):
    tb = np.exp(y[11]); kappa = np.exp(y[12])
    for _ in range(30):
        lMH = rng.uniform(2.0, 3.6); lM33 = rng.uniform(2.0, 3.6)
        MH, M33 = 10**lMH, 10**lM33
        need = kappa * 8 * np.sqrt(2) * np.pi**2 * V / tb      # s2phi * ln(m2^2/m1^2)
        def g(u):
            mu = u * np.sqrt(2) * MH * M33 / V
            m1s, m2s, phi = S.charged_spectrum(MH**2, M33**2, mu)
            return np.sin(2 * phi) * np.log(m2s / m1s) - need
        if g(0.999) < 0 or g(1e-9) > 0:
            # kappa too large for these masses: rescale f down instead (M ~ f * A)
            continue
        u = brentq(g, 1e-9, 0.999)
        x = np.zeros(18); x[:11] = y[:11]; x[11] = y[11]; x[12] = lMH; x[13] = lM33; x[14] = u
        x[15] = rng.uniform(-1, 3); x[16] = rng.uniform(-2, 2); x[17] = rng.uniform(-0.3, 0.3)
        return x
    # fall back: shrink f by factor and raise A target
    return None
res = []; t0 = time.time(); n = 0
while time.time() - t0 < float(__import__('os').environ.get('TLIM', '1200')):
    c, y = base[rng.integers(len(base))]
    y = y.copy()
    # trade f against A: f -> f/lam, kappa -> kappa*lam, with lam in [1, 100]
    lam = 10 ** rng.uniform(0, 2)
    y[:2] = y[:2] / lam; y[12] += np.log(lam)
    x0 = make_start(y)
    if x0 is None: continue
    try:
        r = least_squares(fun, np.clip(x0, LO, HI), bounds=(LO, HI), max_nfev=200, x_scale=np.maximum(HI - LO, 1e-3) * 0.1)
        e = H.evaluate(r.x, ordering); res.append((float(e['chi2']), r.x))
    except Exception: pass
    n += 1
    if n % 20 == 0:
        res.sort(key=lambda t: t[0]); pickle.dump(res[:100], open(out, 'wb'))
        print(n, round(time.time() - t0), [round(c, 1) for c, _ in res[:5]], flush=True)
res.sort(key=lambda t: t[0]); pickle.dump(res[:100], open(out, 'wb')); print('done', flush=True)
