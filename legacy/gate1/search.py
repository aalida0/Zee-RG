import sys, pickle, time, numpy as np
_R = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
sys.path.insert(0, _R)
from scipy.optimize import least_squares
from zee import hgmodel as H, neutrino as N

ordering, seed, nstart, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
rng = np.random.default_rng(seed)
LO, HI = H.LO, H.HI


def draw():
    x = LO + (HI - LO) * rng.random(18)
    x[0:2] = rng.choice([-1, 1], 2) * 10 ** rng.uniform(-4, -1, 2)
    x[2] = rng.choice([-1, 1]) * 10 ** rng.uniform(-4, -1)
    for k in range(4):
        r = 10 ** rng.uniform(-4, -1); th = rng.uniform(0, 2 * np.pi)
        x[3 + 2 * k], x[4 + 2 * k] = r * np.cos(th), r * np.sin(th)
    x[11] = rng.uniform(np.log(.3), np.log(50.))
    x[12], x[13] = rng.uniform(2.0, 3.5, 2)
    x[14] = rng.uniform(0.02, 0.9)
    x[15] = rng.uniform(-1, 3); x[16] = rng.uniform(-2, 2); x[17] = rng.uniform(-0.3, 0.3)
    sc = H.scalar_sector(x); f, Y = H.unpack(x)
    M = N.mnu_hg_leading(f, H.M_LEP, Y, sc['tb'], sc['m1'], sc['m2'], sc['mu'])
    sv = np.linalg.svd(M, compute_uv=False)[0]
    if sv > 0:
        x[:2] = np.clip(x[:2] * 0.05e-9 / sv, -.1, .1)
    return x


def fun(x):
    return H.evaluate(x, ordering)['res']


res = []
t0 = time.time()
for n in range(nstart):
    x0 = np.clip(draw(), LO, HI)
    try:
        r = least_squares(fun, x0, bounds=(LO, HI), max_nfev=200, x_scale=np.maximum(HI - LO, 1e-3) * 0.1)
        e = H.evaluate(r.x, ordering)
        res.append((float(e['chi2']), r.x))
    except Exception as ex:
        pass
    if (n + 1) % 10 == 0:
        res.sort(key=lambda t: t[0])
        pickle.dump(res[:100], open(out, 'wb'))
        print(n + 1, round(time.time() - t0), [round(c, 1) for c, _ in res[:5]], flush=True)
res.sort(key=lambda t: t[0])
pickle.dump(res[:100], open(out, 'wb')); print("done", flush=True)
