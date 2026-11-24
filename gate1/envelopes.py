"""Cutoff-dependent envelopes BR_max(X; Lambda_min): maximum over the final sample and over the verified points of the
dedicated Condor searches (gate1/envopt.py, checked with gate1/env_verify.py). Usage: python gate1/envelopes.py DRAFTDIR ENV_DF.pkl"""
import sys, pickle, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
d = sys.argv[1]; c = pd.read_csv(d + '/figures/final_sample.csv')
G = pickle.load(open(sys.argv[2], 'rb')); G = G[G.chiok & G.hbok & (G.dhs <= 3.84) & G.gok]
obs = [('tau_mu_g', 'Br_tau_mu_gamma', r'${\rm Br}(\tau\to\mu\gamma)$', [(4.2e-8, '-'), (1e-9, '--')]),
       ('mu_e_g', 'Br_mu_e_gamma', r'${\rm Br}(\mu\to e\gamma)$', [(1.5e-13, '-'), (6e-14, '--')]),
       ('Cr_Al', 'CR_Al', r'${\rm CR}(\mu\,{\rm Al}\to e\,{\rm Al})$', [(6.1e-17, '--'), (6e-18, ':')]),
       ('h_tau_mu', 'Br_h_tau_mu', r'${\rm Br}(h\to\tau\mu)$', [(1.8e-3, '-'), (1e-4, '--')])]
L = np.arange(3, 16.01, 1)
def env(o, col, l):
    a = G[(G.obs == o) & (G.Lv >= l - 1e-6)].val; b = c[c.logLmax >= l][col]
    v = pd.concat([a, b]); return v.max() if len(v) else np.nan
fig, axs = plt.subplots(2, 2, figsize=(6.4, 4.8), sharex=True)
for ax, (o, col, lab, ls) in zip(axs.flat, obs):
    y = [env(o, col, l) for l in L]; ys = [c[c.logLmax >= l][col].max() if (c.logLmax >= l).any() else np.nan for l in L]
    ax.step(L, y, where='post', marker='o', ms=3, label='sample + dedicated search'); ax.step(L, ys, where='post', ls='--', c='0.5', lw=0.8, label='sample only')
    ax.set_yscale('log'); ax.set_ylabel(lab + r'$_{\max}$')
    for v, s in ls: ax.axhline(v, c='k', ls=s, lw=0.8)
axs[0, 0].legend(fontsize=6)
for ax in axs[1]: ax.set_xlabel(r'$\log_{10}(\Lambda_{\min}/{\rm GeV})$')
fig.tight_layout(); fig.savefig(d + '/figures/fig_envelope.pdf', dpi=300)
for l in [3, 4, 6, 8, 9, 10, 12, 14, 16]: print(l, *['%.1e' % env(o, col, l) for o, col, _, _ in obs])
