import sys, pickle, time, numpy as np
_R = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, _R + '/gate1')
from scipy.integrate import solve_ivp
from zee import rge, rgeq
import rgtest as R
T1, T2 = 6*np.sqrt(4*np.pi), 6*4*np.pi
SCALES = [1e3, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14, 1e16, 1e19]
def minq(lam, n=30, seed=0):
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed); best = 1e9
    def fg(p):
        n2 = p @ p; q = np.einsum('abcd,a,b,c,d', lam, p, p, p, p); g = 4*np.einsum('abcd,b,c,d->a', lam, p, p, p)
        f = q/24/n2**2
        return f, g/24/n2**2 - 4*q/24*p/n2**3
    for _ in range(n):
        r = minimize(fg, rng.normal(size=10), jac=True, method='BFGS'); best = min(best, r.fun)
    return best
def run_point(x, lam2, ordering='NO', scales=SCALES, check_stab=True):
    sc, f, Y, Y2 = R.HB(x); t0 = np.log(min(sc['m1'], sc['mA'], sc['mH']))
    Y1 = np.diag(np.sqrt(2)*R.ML/R.V).astype(complex)
    c = (sc['lam1'], lam2, sc['lam3'], 0, sc['lam5'], sc['lam6'], 0, 0, 0, 0, 0)
    lam0 = rgeq.tensor_from(c)
    z0 = np.concatenate([rge.pack(Y1, Y2, f, R.YT0, R.G0), lam0.ravel()])
    ev1 = lambda t, z: T1 - np.abs(z[58:]).max(); ev1.terminal = False
    ev2 = lambda t, z: T2 - np.abs(z[58:]).max(); ev2.terminal = True
    tgrid = [np.log(s) for s in scales if np.log(s) > t0]
    s = solve_ivp(rgeq.rhs, (t0, np.log(1e19)), z0, events=[ev1, ev2], t_eval=tgrid, rtol=1e-6, atol=1e-9, method='RK45')
    out = dict(mu0=np.exp(t0), pole_hg=float(np.exp(s.t_events[0][0])) if len(s.t_events[0]) else 1e19,
               pole_4pi=float(np.exp(s.t_events[1][0])) if len(s.t_events[1]) else 1e19, unstable=1e19)
    if check_stab:
        # also check initial point
        for tt, zz in zip(s.t, s.y.T):
            if minq(zz[58:].reshape((10,)*4)) < -1e-6: out['unstable'] = float(np.exp(tt)); break
        if minq(lam0) < -1e-6: out['unstable'] = out['mu0']
    return out
if __name__ == '__main__':
    pts = pickle.load(open(sys.argv[1], 'rb')); c0, x = pts[0]
    for l2 in (float(a) for a in sys.argv[2].split(',')):
        t = time.time(); print(l2, run_point(x, l2), round(time.time()-t, 1), flush=True)
