"""Compare the one-loop quartic, y_t and rho_t beta functions of zee.rgfast2 with PyR@TE 3 (validation/Zee.model, one loop).
Usage: python validation/compare_pyrate.py  (run from the repo root)"""
import sys, inspect, numpy as np
sys.path.insert(0, 'validation'); sys.path.insert(0, '.')
import pyrate_RGEs as P
from zee import rgfast2 as R
rng = np.random.default_rng(3); k = 16*np.pi**2; names = ['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9', 'l10', 'lh']; worst = 0
for trial in range(200):
    g1, g2, g3, yt = rng.uniform(0.3, 1.1, 4); r = rng.uniform(-2, 0); rho = r*yt; c = rng.uniform(-2, 2, 11)
    dz = R.rhs_r(r)(0, np.concatenate([[g1, g2, g3, yt], c])); env = dict(g1=g1, g2=g2, g3=g3, yt=yt, rho=rho, **dict(zip(names, c)))
    th = np.array([getattr(P, 'beta_'+n)(1, *[env[a] for a in list(inspect.signature(getattr(P, 'beta_'+n)).parameters)[1:]]) for n in names])
    worst = max(worst, np.max(np.abs(dz[4:]*k - th)/(np.abs(th) + 1e-6)), abs(dz[3]*k - P.beta_yt(1, g1, g2, g3, yt, rho))/abs(P.beta_yt(1, g1, g2, g3, yt, rho)))
print('max relative difference over 200 random points:', worst)
