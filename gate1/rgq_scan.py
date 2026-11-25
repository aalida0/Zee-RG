import sys, pickle, numpy as np, itertools
_R = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, _R + '/gate1')
from multiprocessing import Pool
from scipy.integrate import solve_ivp
from zee import rge, rgeq
import rgtest as R, rgq
L2S = [0.1, 0.3, 0.6, 1.0, 1.6, 2.5, 3.5449]; LHS = [0.0, 0.3, 1.0, 2.0]
def job(args):
    i, x, l2, lh = args
    sc, f, Y, Y2 = R.HB(x); t0 = np.log(min(sc['m1'], sc['mA'], sc['mH']))
    Y1 = np.diag(np.sqrt(2)*R.ML/R.V).astype(complex)
    c = (sc['lam1'], l2, sc['lam3'], 0, sc['lam5'], sc['lam6'], 0, 0, 0, 0, lh)
    lam0 = rgeq.tensor_from(c)
    z0 = np.concatenate([rge.pack(Y1, Y2, f, R.YT0, R.G0), lam0.ravel()])
    ev1 = lambda t, z: rgq.T1 - np.abs(z[58:]).max(); ev1.terminal = False
    ev2 = lambda t, z: rgq.T2 - np.abs(z[58:]).max(); ev2.terminal = True
    tg = [np.log(s) for s in rgq.SCALES if np.log(s) > t0]
    s = solve_ivp(rgeq.rhs, (t0, np.log(1e19)), z0, events=[ev1, ev2], t_eval=tg, rtol=1e-6, atol=1e-9)
    ph = float(np.exp(s.t_events[0][0])) if len(s.t_events[0]) else 1e19
    p4 = float(np.exp(s.t_events[1][0])) if len(s.t_events[1]) else 1e19
    un = 1e19
    if rgq.minq(lam0, n=15) < -1e-4: un = float(np.exp(t0))
    else:
        for tt, zz in zip(s.t, s.y.T):
            if rgq.minq(zz[58:].reshape((10,)*4), n=15) < -1e-4: un = float(np.exp(tt)); break
    return (i, l2, lh, ph, p4, un)
if __name__ == '__main__':
    pts = pickle.load(open(sys.argv[1], 'rb')); cmin = pts[0][0]
    pts = [(c, x) for c, x in pts if c <= cmin + 3.84]; uniq = []
    for c, x in pts:
        if all(np.abs(x-u).max() > 1e-4 for _, u in uniq): uniq.append((c, x))
    jobs = [(i, x, a, b) for i, (c, x) in enumerate(uniq) for a, b in itertools.product(L2S, LHS)]
    print(len(uniq), 'points', len(jobs), 'jobs', flush=True)
    with Pool(2) as p: res = p.map(job, jobs, chunksize=4)
    pickle.dump(dict(res=res, pts=uniq), open(sys.argv[2], 'wb')); print('done', flush=True)
