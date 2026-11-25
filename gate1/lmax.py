"""Targeted estimate of Lambda_max(theta): for each allowed point, best cutoff over the free quartics eta=(l2,l7,l8,l9,l10,lh)
subject to the h->gamma gamma constraint. Random stage (n0 draws, as in the filter) then adaptive local search.
Usage: ZEE_INPUTS=modern python lmax.py STRIDE NPROC [N0 NIT]"""
import sys, pickle, numpy as np
_R = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, _R + '/gate1')
from multiprocessing import Pool
from zee import rgfast2, bfb2
import m_filter_l4w as M
from m_filter_l4w import H
SC = np.log(10.0**np.arange(3, 16.01, 1.0)); CM = 4*np.pi
def lam_valid(c, t0, r):
    """log10 of the largest scale (<=1e16) up to which couplings stay < 4 pi and the quartic form stays bounded from below.
    Failed candidates return log10(last good scale) + 5*min(0, min V_red) so the local search sees a gradient."""
    v0 = bfb2.min_vred(c, refine=2)
    if v0 < -1e-4: return np.log10(np.exp(t0)) + 5*v0 - 1.0
    ts = SC[SC > t0]
    s = rgfast2.run(c, t0, t1=np.log(1e16), cmax=CM, scales=ts if len(ts) else None, r=r)
    lim = s.t_events[0][0] if len(s.t_events[0]) else np.log(1e16)
    best = t0
    Y = np.asarray(s.y)
    if Y.size == 0: return np.log10(np.exp(lim))
    for t, zz in zip(s.t, Y.T):
        if t > lim: break
        v = bfb2.min_vred(zz[4:], refine=0)
        if v < -1e-4: return np.log10(np.exp(best)) + 5*min(v, 0) - 0.5
        best = t
    return np.log10(np.exp(lim))
def job(a):
    i, x = a; n0, nit = NN
    rng = np.random.default_rng(5000 + i)
    e = H.evaluate(x, 'NO'); sc = e['sc']; t0 = np.log(min(sc['m1'], sc['mA'], sc['mH'])); r = -sc['tb']
    mu_gg = e['mus']['gg']; g0 = ((mu_gg - 1.14)/0.185)**2; rgg0 = e['rgg']
    def build(eta):
        c = np.zeros(11); c[0], c[2], c[3], c[4], c[5] = sc['lam1'], sc['lam3'], sc['lam4'], sc['lam5'], sc['lam6']
        c[1], c[6], c[7], c[8], c[9], c[10] = eta; return c
    def ok(c):
        d = ((mu_gg*M.rgg_gen(sc, c[6], c[7], c[9])/rgg0 - 1.14)/0.185)**2 - g0
        return e['chi2'] + d <= M.LIM
    lo = np.array([0.02, -1.5, -1.5, -1.5, -1.5, 0.0]); hi = np.array([3.5449, 1.5, 1.5, 1.5, 1.5, 3.0])
    draw = lambda: np.array([np.exp(rng.uniform(np.log(.02), np.log(3.5449))), rng.uniform(-1.5, 1.5), *(rng.uniform(-1.5, 1.5, 3) * (rng.random(3) < .6)), rng.uniform(0, 3)])
    pool = []
    for k in range(n0):
        eta = draw(); c = build(eta)
        if ok(c): pool.append((lam_valid(c, t0, r), eta))
    if not pool: return i, None, None
    pool.sort(key=lambda t: -t[0]); L_rand = pool[0][0]
    cur, eta = pool[0]; step = 0.3*(hi - lo)
    for it in range(nit):
        cand = np.clip(eta + step*rng.normal(size=6)*(rng.random(6) < .5), lo, hi); c = build(cand)
        if not ok(c): continue
        L = lam_valid(c, t0, r)
        if L >= cur: cur, eta = L, cand
        else: step *= 0.97
    return i, L_rand, cur
NN = (100, 150)
if __name__ == '__main__':
    stride, npr = int(sys.argv[1]), int(sys.argv[2])
    if len(sys.argv) > 4: NN = (int(sys.argv[3]), int(sys.argv[4]))
    S = pickle.load(open('/tmp/lam4wide_NO.pkl', 'rb')); best = pickle.load(open('/tmp/mbest_NO.pkl', 'rb'))[0][0]; M.LIM = best + 3.84
    pts = [(i, S[i][1]) for i in range(0, len(S), stride) if S[i][0] < best + 3.84]
    off = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    pts = pts[off:]
    with Pool(npr) as P: out = P.map(job, pts, chunksize=1)
    pickle.dump(out, open('/tmp/lmax.pkl', 'wb'))
    o = [x for x in out if x[1] is not None]
    for L in (4, 8, 16):
        print('Lambda>=1e%d' % L, 'random', np.mean([x[1] >= L - 1e-9 for x in o]).round(3), 'optimised', np.mean([x[2] >= L - 1e-9 for x in o]).round(3), len(o))
