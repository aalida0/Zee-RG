import sys, pickle, numpy as np
_R = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, _R + '/gate1')
from multiprocessing import Pool
from zee import hgmodel as H, rgfast, bfb
import final_filter as F
V = H.V; MH = H.MH
LIM = 3.2785303291396612 + 3.84
def rgg_gen(sc, l7, l8, l10, mt=172.5, mw=80.379):
    mh2 = MH**2
    fx = lambda x: np.arcsin(np.sqrt(x))**2 if x <= 1 else -0.25 * (np.log((1 + np.sqrt(1 - 1/x + 0j)) / (1 - np.sqrt(1 - 1/x + 0j))) - 1j*np.pi)**2
    A12 = lambda t: 2*t + 2*t*(1-t)*fx(1/t); A1 = lambda t: -2 - 3*t - 3*t*(2-t)*fx(1/t); A00 = lambda t: -t + t**2*fx(1/t)
    sba, cba, tb, ph = sc['sba'], sc['cba'], sc['tb'], sc['phi']
    kt = sba - tb*cba; c, s = np.cos(ph), np.sin(ph)
    dHH = sba*sc['lam3'] + cba*l7; dhh = sba*l8 + cba*l10; dHh = -sba*sc['mu']/(np.sqrt(2)*V)
    lam = (c*c*dHH + s*s*dhh + 2*s*c*dHh, s*s*dHH + c*c*dhh - 2*s*c*dHh)
    tW, tt = 4*mw**2/mh2, 4*mt**2/mh2
    num = sba*A1(tW) + (4/3)*kt*A12(tt)
    for l, m in zip(lam, (sc['m1'], sc['m2'])): num += l*V**2/(2*m**2)*A00(4*m**2/mh2)
    return float(abs(num/(A1(tW) + (4/3)*A12(tt)))**2)
def one(args):
    i, x = args
    rng = np.random.default_rng(1000 + i)
    e = H.evaluate(x, 'NO'); sc = e['sc']; t0 = np.log(min(sc['m1'], sc['mA'], sc['mH']))
    mu_gg = e['mus']['gg']; g0 = ((mu_gg - 1.14)/0.185)**2; rgg0 = e['rgg']
    out = {}
    cands = []
    for k in range(400):
        c = np.zeros(11); c[0], c[2], c[4], c[5] = sc['lam1'], sc['lam3'], sc['lam5'], sc['lam6']
        c[1] = np.exp(rng.uniform(np.log(0.02), np.log(3.5449))); c[6] = rng.uniform(-1.5, 1.5); c[10] = rng.uniform(0, 3)
        if k >= 120:
            for j in (7, 8, 9):
                if rng.random() < 0.6: c[j] = rng.uniform(-1.5, 1.5)
        d = ((mu_gg*rgg_gen(sc, c[6], c[7], c[9])/rgg0 - 1.14)/0.185)**2 - g0
        if e['chi2'] + d <= LIM: cands.append(c)
    out['n_rgg_ok'] = len(cands)
    cm = 4*np.pi
    runs = [(c, rgfast.run(c, t0, cmax=cm)) for c in cands]
    poles = [np.exp(s.t_events[0][0]) if len(s.t_events[0]) else 1e19 for _, s in runs]
    out['ew'] = any(bfb.min_v4(c, n=4000) >= -1e-4 for c in cands[:60])
    for L in (1e4, 1e8, 1e16):
        ok = False
        for (c, _), p in zip(runs, poles):
            if p > L and F.stable_to(c, t0, np.log(L), cm): ok = True; break
        out[L] = ok
    return i, out
if __name__ == '__main__':
    S = pickle.load(open('/tmp/final_samples.pkl', 'rb'))[::8]
    with Pool(2) as p: res = p.map(one, list(enumerate([x for _, x in S])), chunksize=6)
    pickle.dump(dict(res=res), open('/tmp/final_filter3.pkl', 'wb')); print('done', len(res), flush=True)
