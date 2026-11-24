import sys, pickle, numpy as np
sys.path.insert(0, '/home/claude/zee_fit')
from multiprocessing import Pool
from zee import hgmodel as H, rgfast, bfb
LAMS = [1e4, 1e8, 1e16]; CM = {'hg': np.sqrt(4*np.pi), '4pi': 4*np.pi}
SCAL = np.log(np.array([1e3, 1e4, 1e5, 1e6, 1e8, 1e10, 1e12, 1e14, 1e16]))
def stable_to(c, t0, tmax, cmax):
    ts = [t for t in SCAL if t0 < t <= tmax]
    s = rgfast.run(c, t0, t1=tmax, cmax=cmax, scales=ts if ts else None)
    if len(s.t_events[0]): return False
    if bfb.min_v4(c, n=4000) < -1e-4: return False
    for zz in s.y.T:
        if bfb.min_v4(zz[4:], n=4000) < -1e-4: return False
    return True
def one(args):
    i, x = args
    rng = np.random.default_rng(i)
    sc = H.scalar_sector(x); t0 = np.log(min(sc['m1'], sc['mA'], sc['mH']))
    base = np.array([sc['lam1'], 0, sc['lam3'], 0, sc['lam5'], sc['lam6'], 0, 0, 0, 0, 0])
    out = {}
    cands = []
    for _ in range(120):
        c = base.copy(); c[1] = np.exp(rng.uniform(np.log(0.02), np.log(3.5449))); c[6] = rng.uniform(-1.5, 1.5); c[10] = rng.uniform(0, 3)
        cands.append(c)
    c = base.copy(); c[1] = 3.5449; cands.insert(0, c)            # HG literal
    for key, cm in CM.items():
        runs = [(c, rgfast.run(c, t0, cmax=cm)) for c in cands]
        poles = [np.exp(s.t_events[0][0]) if len(s.t_events[0]) else 1e19 for _, s in runs]
        out[key + '_hgliteral'] = {L: (poles[0] > L) and stable_to(cands[0], t0, np.log(L), cm) for L in LAMS}
        for L in LAMS:
            ok = False
            for (c, _), p in zip(runs, poles):
                if p > L and stable_to(c, t0, np.log(L), cm): ok = True; break
            out[(key, L)] = ok
    out['ewBFB_any'] = any(bfb.min_v4(c, n=4000) >= -1e-4 for c in cands[:40])
    return i, out
if __name__ == '__main__':
    S = pickle.load(open('/tmp/final_samples.pkl', 'rb'))[::int(sys.argv[1])]
    with Pool(2) as p: res = p.map(one, list(enumerate([x for _, x in S])), chunksize=8)
    pickle.dump(dict(res=res, S=S), open('/tmp/final_filter.pkl', 'wb')); print('done', len(res), flush=True)
