"""Relaxed texture: f_emu and Y2^{ee,emu,mue,mumu} (complex) free on top of the 19-parameter fit (lambda4 free).
Stage 'fit': multistart least squares for NO and IO from the slice best points with random small extra entries.
Stage 'walk': random walk (Delta chi2 <= 3.84) from the best relaxed NO points, as walk2.
Usage: ZEE_INPUTS=modern python texture.py fit NSEED | walk NCHAIN NIT"""
import sys, pickle, time, numpy as np
_R = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
sys.path.insert(0, _R)
from multiprocessing import Pool
from scipy.optimize import least_squares
from zee import hgmodel as H
S0 = 1e-6; ZM = np.arcsinh(0.1/S0); SQ = np.sqrt(4*np.pi)
YI = list(range(11)) + list(range(19, 28))            # Yukawa-like entries (signed-log)
LO = np.concatenate([H.LO, [-SQ], -0.1*np.ones(9)]); HI = np.concatenate([H.HI, [SQ], 0.1*np.ones(9)])
LO[YI] = -ZM; HI[YI] = ZM
xs = np.maximum(HI - LO, 1e-3)*0.1; xs[YI] = 1.0
def tx(z): x = z.copy(); x[YI] = S0*np.sinh(z[YI]); return x
def tz(x): z = np.array(x, float); z[YI] = np.arcsinh(np.clip(z[YI], -.1, .1)/S0); return z
def extend(x, rng):
    x = list(x) + [0.0]*(19 - len(x))
    return np.array(x + list(rng.choice([-1, 1], 9)*10**rng.uniform(-9, -4, 9)))
def fit1(a):
    o, k, x = a; rng = np.random.default_rng(k); z = np.clip(tz(extend(x, rng)), LO, HI)
    try:
        r = least_squares(lambda z: H.evaluate(tx(z), o)['res'], z, bounds=(LO, HI), max_nfev=300, x_scale=xs)
        return o, float(H.evaluate(tx(r.x), o)['chi2']), tx(r.x)
    except Exception: return o, 1e9, None
def chain(a):
    k, c0, x0, lim = a; rng = np.random.default_rng(300 + k); z = tz(x0); span = HI - LO; span[YI] = 3.0; step = 0.02; out = []; acc = n = 0
    for it in range(NIT):
        y = np.clip(z + rng.normal(size=28)*span*step*(rng.random(28) < 0.3), LO, HI)
        try: e = H.evaluate(tx(y), 'NO')
        except Exception: continue
        n += 1
        if e['pen'] == 0 and e['chi2'] <= lim: z = y; acc += 1
        if it % 100 == 99: step = float(np.clip(step*(1.3 if acc/max(n, 1) > .3 else .7), 1e-4, .3)); acc = n = 0
        if it >= NIT//6 and it % 15 == 0: x = tx(z); out.append((float(H.evaluate(x, 'NO')['chi2']), x))
    return out
NIT = 0
if __name__ == '__main__':
    if sys.argv[1] == 'fit':
        ns = int(sys.argv[2]); jobs = []
        for o in ('NO', 'IO'):
            for k, (_, x) in enumerate(pickle.load(open(f'/tmp/mbest_{o}.pkl', 'rb'))[:ns]):
                for r in range(2): jobs.append((o, 1000*k + r + (0 if o == 'NO' else 500000), x))
        with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 1) as P: res = P.map(fit1, jobs, chunksize=1)
        res = [r for r in res if r[2] is not None]; pickle.dump(res, open('/tmp/texture_fit.pkl', 'wb'))
        for o in ('NO', 'IO'): print(o, 'relaxed chi2min', min(c for oo, c, _ in res if oo == o))
    else:
        nc, NIT = int(sys.argv[2]), int(sys.argv[3])
        res = sorted([r for r in pickle.load(open('/tmp/texture_fit.pkl', 'rb')) if r[0] == 'NO'], key=lambda t: t[1]); lim = res[0][1] + 3.84
        seeds = [r for r in res if r[1] <= lim][:nc]
        d = pickle.load(open('/tmp/mfilt4w_NO.pkl', 'rb'))['S']; rng = np.random.default_rng(9)
        for k in rng.permutation(len(d)):
            if len(seeds) >= nc: break
            x = extend(d[k][1], rng); x[19:] *= 1e-3; c = H.evaluate(x, 'NO')['chi2']
            if c <= lim: seeds.append(('NO', c, x))
        with Pool(int(sys.argv[4]) if len(sys.argv) > 4 else 1) as P: out = P.map(chain, [(k, c, x, lim) for k, (_, c, x) in enumerate(seeds)])
        out = sum(out, []); pickle.dump(out, open('/tmp/texture_walk.pkl', 'wb')); print('walk', len(out), len(seeds))
