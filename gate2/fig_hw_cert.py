"""Figures from results/benchmarks.json: (a) BR(l_i->l_j gamma) versus the multiplier s of the H+-W-+ gamma amplitude, (b) first-failure scale of the
one-loop flow for every boundary variant.  Writes fig_hw.pdf and fig_variants.pdf into the draft directory (argv[1])."""
import json, sys, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); out = sys.argv[1] if len(sys.argv) > 1 else '.'
B = json.load(open(os.path.join(ROOT, 'results', 'benchmarks.json')))
def amp(dc, s):
    tot = 0.0
    for ch in ('A_L', 'A_R'):
        a = sum(complex(*dc[k][ch]) for k in dc if k != 'bzHW'); h = complex(*dc['bzHW'][ch]); tot += abs(a + s*h)**2
    return tot
S = np.linspace(-1.5, 1.5, 301)
lim = {'tau_mu_g': (4.2e-8, 1e-9, r'$\tau\to\mu\gamma$'), 'mu_e_g': (1.5e-13, 6e-14, r'$\mu\to e\gamma$')}
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
for a_, ch in zip(ax, ('tau_mu_g', 'mu_e_g')):
    for nm, c in (('BM1', 'C0'), ('BM2', 'C3')):
        r = B[nm]; dc = r['dipole_components'][ch]; nom = r['observables'][ch]
        br = nom*np.array([amp(dc, s) for s in S])/amp(dc, 1.0); a_.semilogy(S, br, c, label=nm)
        smin = S[np.argmin(br)]; a_.plot([smin], [br.min()], 'o', c=c, ms=3)
    a_.axhline(lim[ch][0], c='k', ls='-', lw=0.8); a_.axhline(lim[ch][1], c='gray', ls='--', lw=0.8)
    a_.axvspan(-1, 1, color='0.92', zorder=0); a_.set_xlabel(r'multiplier $s$ of the $H^\pm W^\mp\gamma$ amplitude'); a_.set_ylabel(r'$\mathrm{Br}$ ' + lim[ch][2]); a_.legend(frameon=False, fontsize=8)
ax[0].text(-1.45, 1.2e-9, 'Belle II reach', fontsize=7, color='gray'); ax[0].text(-1.45, 5.0e-8, 'limit', fontsize=7); ax[1].text(-1.45, 7.5e-14, 'MEG II reach', fontsize=7, color='gray'); ax[1].text(-1.45, 1.8e-13, 'limit', fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(out, 'fig_hw.pdf'))
for nm in B:
    r = B[nm]; dc = r['dipole_components']['tau_mu_g']; nom = r['observables']['tau_mu_g']; br = nom*np.array([amp(dc, s) for s in np.linspace(-1, 1, 401)])/amp(dc, 1.0)
    print(nm, 'tau->mu gamma: s=+1 %.2e, s=0 %.2e, s=-1 %.2e, min over [-1,1] %.2e' % (nom, nom*amp(dc, 0)/amp(dc, 1), nom*amp(dc, -1)/amp(dc, 1), br.min()))
names = ['default', 'match_x0.5', 'match_x1', 'match_x2', 'matchH_x0.5', 'matchH_x1', 'matchH_x2', 'qscale_1.1', 'qscale_0.9', 'sm_yt_1.02', 'sm_yt_0.98', 'sm_gs_1.03', 'sm_gs_0.97']
fig, ax = plt.subplots(figsize=(7.2, 2.9)); w = 0.38
for i, (nm, c) in enumerate((('BM1', 'C0'), ('BM2', 'C3'))):
    ax.bar(np.arange(len(names)) + (i - 0.5)*w, [B[nm]['robustness'][n]['log10Lambda'] for n in names], w, color=c, label=nm)
ax.set_xticks(range(len(names))); ax.set_xticklabels([n.replace('_', ' ') for n in names], rotation=40, ha='right', fontsize=7)
ax.set_ylabel(r'first failure, $\log_{10}(\Lambda/{\rm GeV})$'); ax.set_ylim(10, 16); ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(out, 'fig_variants.pdf'))
