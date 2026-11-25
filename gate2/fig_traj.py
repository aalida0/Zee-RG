"""Fig. traj: one-loop flow for BM1, BM2 and an illustrative non-benchmark (BM1 with the matching-scale quartics x2.5)."""
import sys, json, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as B
from bench import R, np as _np
from scipy.integrate import solve_ivp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
inp = json.load(open('results/benchmarks_input.json'))
def flow(x, eta, scale=1.0):
    sc, c = B.coup(x, eta); c = c*scale; r = -sc['tb']
    t0 = np.log(min(sc['m1'], sc['mA'], sc['mH'])); g = np.array([0.36, 0.64, 1.05, 0.88])
    s = solve_ivp(R.rhs_r(r), (t0, B.LOG16), np.concatenate([g, c]), dense_output=True, rtol=1e-8, atol=1e-10)
    ts = np.linspace(t0, min(s.t[-1], B.LOG16), 400); out = []
    for t in ts:
        cc = s.sol(t)[4:]; vb, u, mx = B.margins(cc); out.append((t/np.log(10), mx, u/(8*np.pi), vb))
        if not np.isfinite(mx) or u > 8*np.pi*1.5 or mx > 4*np.pi*1.5: break
    return np.array(out)
cases = [('BM1', inp['BM1'], 1.0), ('BM2', inp['BM2'], 1.0), (r'BM1, quartics $\times2.5$ (illustrative)', inp['BM1'], 2.5)]
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
for lab, p, k in cases:
    d = flow(p['x'], p['eta'], k)
    for a, j in zip(ax, (1, 2, 3)): a.plot(d[:, 0], d[:, j], label=lab)
ax[0].axhline(4*np.pi, c='k', ls=':'); ax[0].set_ylabel(r'$\max|\lambda_i|$'); ax[1].axhline(1, c='k', ls=':'); ax[1].set_ylabel(r'$\max|\Lambda_{2\to2}|/8\pi$')
ax[2].axhline(0, c='k', ls=':'); ax[2].set_ylabel(r'$\min V_{\rm red}$'); ax[0].legend(fontsize=7)
for a in ax: a.set_xlabel(r'$\log_{10}(\mu/\mathrm{GeV})$')
fig.tight_layout(); fig.savefig('../zee-rg-draft/figures/fig_trajectories.pdf'); fig.savefig('results/fig_trajectories.pdf')
