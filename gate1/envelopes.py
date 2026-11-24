"""Cutoff-dependent envelopes BR_max(X; Lambda_min) from final_sample.csv. Usage: python gate1/envelopes.py DRAFTDIR"""
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
d = sys.argv[1]; c = pd.read_csv(d + '/figures/final_sample.csv')
L = np.arange(3, 12.01, 1)
obs = [('Br_tau_mu_gamma', r'${\rm Br}(\tau\to\mu\gamma)$', [(4.2e-8, '-'), (1e-9, '--')]), ('Br_mu_e_gamma', r'${\rm Br}(\mu\to e\gamma)$', [(1.5e-13, '-'), (6e-14, '--')]),
       ('CR_Al', r'${\rm CR}(\mu\,{\rm Al}\to e\,{\rm Al})$', [(6.1e-17, '--')]), ('Br_h_tau_mu', r'${\rm Br}(h\to\tau\mu)$', [(1.8e-3, '-'), (1e-4, '--')])]
fig, axs = plt.subplots(2, 2, figsize=(6.4, 4.8), sharex=True)
for ax, (k, lab, ls) in zip(axs.flat, obs):
    y = [c[c.logLmax >= l][k].max() if (c.logLmax >= l).any() else np.nan for l in L]
    ax.step(L, y, where='post', marker='o', ms=3); ax.set_yscale('log'); ax.set_ylabel(lab + r'$_{\max}$')
    for v, s in ls: ax.axhline(v, c='k', ls=s, lw=0.8)
for ax in axs[1]: ax.set_xlabel(r'$\log_{10}(\Lambda_{\min}/{\rm GeV})$')
fig.tight_layout(); fig.savefig(d + '/figures/fig_envelope.pdf', dpi=300)
print(' & '.join(['logL', 'N'] + [k for k, _, _ in obs]))
for l in [3, 4, 6, 8, 10, 12]:
    s = c[c.logLmax >= l]; print(l, len(s), *[f"{s[k].max():.2e}" for k, _, _ in obs])
