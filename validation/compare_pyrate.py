"""Executable gate: one-loop beta functions of zee.rgfast2 vs PyR@TE 3 (validation/Zee.model -> pyrate_RGEs.py).
Checks, at 200 random points and in the rho -> 0 limit: all 11 quartics, y_t, the three gauge couplings, and
beta_rho = r beta_yt (so rho/y_t = r is RG invariant, which rgfast2 assumes). Exits nonzero on any disagreement.
Zee.model contains only the third-generation Q and t_R; for the gauge couplings the analytic one-loop contribution of the
remaining SM fermions, (2/3) sum_Weyl (Y^2, T2, T3) = (17/3, 3, 3), is added to the PyR@TE coefficients.
Usage: python validation/compare_pyrate.py  (from the repo root)"""
import os, sys, inspect, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import pyrate_RGEs as P
from zee import rgfast2 as R

NAMES = ['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9', 'l10', 'lh']; K = 16*np.pi**2; RTOL = 1e-9


def call(fn, env):
    f = getattr(P, fn); return f(1, *[env[a] for a in list(inspect.signature(f).parameters)[1:]])


def compare(n=200, seed=3):
    rng = np.random.default_rng(seed); worst = {}
    def upd(key, a, b):
        d = abs(a - b)/max(abs(b), 1e-3); worst[key] = max(worst.get(key, 0.0), d)
    for trial in range(n + 20):
        g1, g2, g3, yt = rng.uniform(0.3, 1.1, 4); r = 0.0 if trial >= n else rng.uniform(-3, 0); c = rng.uniform(-2, 2, 11)
        env = dict(g1=g1, g2=g2, g3=g3, yt=yt, rho=r*yt, **dict(zip(NAMES, c)))
        dz = R.rhs_r(r)(0, np.concatenate([[g1, g2, g3, yt], c]))*K
        for i, nm in enumerate(NAMES): upd(nm, dz[4 + i], call('beta_' + nm, env))
        upd('yt', dz[3], call('beta_yt', env))
        upd('rho', r*dz[3], call('beta_rho', env))
        for i, (nm, db) in enumerate((('g1', 17/3), ('g2', 3.0), ('g3', 3.0))):
            gi = (g1, g2, g3)[i]; upd(nm, dz[i], call('beta_' + nm, env) + db*gi**3)
    return worst


if __name__ == '__main__':
    w = compare()
    for k, v in w.items(): print('%-4s max rel. diff %.2e' % (k, v))
    bad = [k for k, v in w.items() if not v < RTOL]
    if bad: sys.exit('FAIL: ' + ', '.join(bad))
    print('PASS (all components within %.0e)' % RTOL)
