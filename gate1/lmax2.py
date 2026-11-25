"""Targeted estimate of Lambda_max(theta): for each allowed point, best cutoff over the free quartics eta=(l2,l7,l8,l9,l10,lh)
subject to the h->gamma gamma constraint. Random stage (n0 draws, as in the filter) then adaptive local search.
Usage: ZEE_INPUTS=modern python lmax.py STRIDE NPROC [N0 NIT]"""
import sys, pickle, numpy as np
sys.path.insert(0, '/home/claude/zee_fit'); sys.path.insert(0, '/home/claude/zee_fit/gate1')
from multiprocessing import Pool
from zee import rgfast2, bfb2, vac
import m_filter_l4w as M
from m_filter_l4w import H
SC = np.log(10.0**np.arange(3, 16.01, 1.0)); CM = 4*np.pi
def _bfb(c):
    """scan-level BFB margin: coarse grid, refined on the fine grid only when the coarse margin is small"""
    v = bfb2.min_vred(c, refine=0, coarse=True)
    return v if v > 0.02 else bfb2.min_vred(c, refine=2)
def lam_valid(c, t0, r, status=False):
    """Search score: log10 of the largest scale (<=1e16) up to which couplings stay < 4 pi, unitarity holds and the quartic
    form stays bounded from below, on the grid SC. Failed candidates return log10(last good scale) minus a penalty so the
    local search sees a gradient. Fails closed: an integration that stops without a perturbativity event, returns nothing
    or produces non-finite couplings scores the last good scale (never the endpoint); status=True also returns the reason.
    This is a discovery heuristic; benchmark claims use gate2/bench.trajectory (fine grid, bracketing, endpoint check)."""
    def ret(v, why): return (v, why) if status else v
    if not np.all(np.isfinite(c)): return ret(-99.0, 'nonfinite')
    um = vac.unit_max(c)
    if um > 8*np.pi: return ret(np.log10(np.exp(t0)) - 1.0 - (um - 8*np.pi)/8, 'unitarity@start')
    v0 = bfb2.min_vred(c, refine=2)
    if v0 < 0: return ret(np.log10(np.exp(t0)) + max(5*v0, -3.0) - 1.0, 'BFB@start')
    t1 = np.log(1e16); ts = SC[SC > t0]
    s = rgfast2.run(c, t0, t1=t1, cmax=CM, scales=ts if len(ts) else None, r=r)
    ev = s.t_events[0] if s.t_events is not None else []
    if s.status == -1: lim, why = None, 'integration'
    elif len(ev): lim, why = ev[0], 'perturbativity'
    elif s.status == 0: lim, why = t1, 'none'
    else: lim, why = None, 'integration'
    best = t0; Y = np.asarray(s.y)
    for t, zz in zip(s.t, Y.T if Y.size else []):
        if lim is not None and t > lim: break
        if not np.all(np.isfinite(zz)): return ret(np.log10(np.exp(best)) - 0.5, 'nonfinite')
        if vac.unit_max(zz[4:]) > 8*np.pi: return ret(np.log10(np.exp(best)) - 0.25, 'unitarity')
        v = _bfb(zz[4:])
        if v < 0: return ret(np.log10(np.exp(best)) + max(5*min(v, 0), -3.0) - 0.5, 'BFB')
        best = t
    if lim is None: return ret(np.log10(np.exp(best)) - 0.5, why)      # solver failure: last checked scale, penalised
    return ret(np.log10(np.exp(lim)), why)
def job(a):
    i, x = a; n0, nit = NN
    rng = np.random.default_rng(5000 + i)
    e = H.evaluate(x, 'NO'); sc = e['sc']; t0 = np.log(min(sc['m1'], sc['mA'], sc['mH'])); r = -sc['tb']
    mu_gg = e['mus']['gg']; g0 = ((mu_gg - 1.14)/0.185)**2; rgg0 = e['rgg']
    def build(eta):
        c = np.zeros(11); c[0], c[2], c[3], c[4], c[5] = sc['lam1'], sc['lam3'], sc['lam4'], sc['lam5'], sc['lam6']
        c[1], c[6], c[7], c[8], c[9], c[10] = eta; return c
    def ok(c):   # the full statistic with this choice of free quartics (same as gate2/accept.py)
        return H.evaluate(x, 'NO', eta=[c[1], c[6], c[7], c[8], c[9], c[10]])['chi2'] <= M.LIM
    lo = np.array([0.02, -1.5, -1.5, -1.5, -1.5, 0.0]); hi = np.array([3.5449, 1.5, 1.5, 1.5, 1.5, 3.0])
    draw = lambda: np.array([np.exp(rng.uniform(np.log(.02), np.log(3.5449))), rng.uniform(-1.5, 1.5), *(rng.uniform(-1.5, 1.5, 3) * (rng.random(3) < .6)), rng.uniform(0, 3)])
    pool = []
    for k in range(n0):
        eta = draw(); c = build(eta)
        if ok(c): pool.append((lam_valid(c, t0, r), eta))
    if not pool: return i, None, None
    pool.sort(key=lambda t: -t[0]); L_rand = pool[0][0]
    gok = lambda eta, n=8: vac.global_ok(build(eta), sc, nstart=n, rng=rng)[0]
    good = [p for p in pool[:8] if gok(p[1])]
    if not good: return i, L_rand, None
    cur, eta = good[0]; L_rand_ccb = cur; step = 0.3*(hi - lo)
    for it in range(nit):
        cand = np.clip(eta + step*rng.normal(size=6)*(rng.random(6) < .5), lo, hi); c = build(cand)
        if not ok(c): continue
        L = lam_valid(c, t0, r)
        if L >= cur and gok(cand): cur, eta = L, cand
        else: step *= 0.97
    return i, L_rand, cur, L_rand_ccb, list(map(float, eta)), bool(gok(eta, 80))
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
