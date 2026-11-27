"""Figure of the prod1 campaign: LFV rates versus scale for the re-verified points (grey), the 37 finalists flagged robust by the matched-margin proxy, their matched
stress minimum under the effective-potential matching (results/prod1/screen_ep) and the campaign benchmarks.
Usage: python3 gate2/make_fig_prod1.py OUT.pdf"""
import json, os, sys, glob, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P = os.path.join(ROOT, 'results', 'prod1')
S = json.load(open(os.path.join(P, 'campaign_summary.json'))); F = json.load(open(os.path.join(P, 'campaign_finalists.json')))
key = lambda x: tuple(np.round(x, 12)); idx = {key(v['x']): v for v in S['verified']}
V = [v for v in S['verified'] if v['status'] == 'PASS' and v['screen13'] is not None]
OBS = (('mu_e_g', r'$\mathrm{Br}(\mu\to e\gamma)$'), ('tau_mu_g', r'$\mathrm{Br}(\tau\to\mu\gamma)$'), ('h_tau_mu', r'$\mathrm{Br}(h\to\tau\mu)$'))
fin = {}
for f in glob.glob(os.path.join(P, 'screen_ep', 'validate_C*.json')):
    n = os.path.basename(f)[9:-5]; d = json.load(open(f)); m = d['matching']['ep']; v = idx[key(F[n]['x'])]
    clean = m['complete'] and m['n_start_fail'] == 0 and m['min_stress'] is not None and m['min_stress'] >= 10
    fin[n] = (d['tree']['min'], m['min_stress'], clean, v['obs'])
BP = {'BP1': 'C42', 'BP2': 'C57', 'BP3': 'C35'}
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
for a, (k, lab) in zip(ax, OBS):
    a.scatter([v['screen13'] for v in V], [max(v['obs'][k], 1e-19) for v in V], s=5, c='0.75', lw=0, label='verified points')
    for n, (t, ms, clean, o) in fin.items():
        y = max(o[k], 1e-19)
        if clean and ms is not None: a.plot([t, ms], [y, y], c='#009E73', lw=0.5, alpha=0.6); a.scatter([ms], [y], s=14, c='#009E73', marker='o', lw=0, zorder=3)
        elif ms is not None: a.plot([t, ms], [y, y], c='#D55E00', lw=0.5, alpha=0.6); a.scatter([ms], [y], s=14, c='#D55E00', marker='o', lw=0, zorder=3)
        a.scatter([t], [y], s=14, c='#D55E00' if not clean else '#009E73', marker='x', lw=0.8, zorder=3)
    for b, c in BP.items(): a.annotate(b, (fin[c][1], fin[c][3][k]), fontsize=7, xytext=(3, 3), textcoords='offset points')
    a.set_yscale('log'); a.set_xlabel(r'$\log_{10}(\Lambda/\mathrm{GeV})$'); a.set_ylabel(lab); a.set_xlim(2, 16.5)
ax[0].set_ylim(1e-19, 3e-13); ax[1].set_ylim(1e-10, 5e-8); ax[2].set_ylim(1e-9, 3e-3)
h = [plt.Line2D([], [], c='0.75', marker='o', ls='', ms=3, label='re-verified points (screening scale)'),
     plt.Line2D([], [], c='#009E73', marker='x', ls='', ms=4, label='finalist, tree scale (clean under matching)'),
     plt.Line2D([], [], c='#009E73', marker='o', ls='', ms=3, label='same, matched stress minimum'),
     plt.Line2D([], [], c='#D55E00', marker='x', ls='', ms=4, label='finalist, tree scale (not clean)'),
     plt.Line2D([], [], c='#D55E00', marker='o', ls='', ms=3, label='same, matched stress minimum (if defined)')]
fig.legend(handles=h, loc='lower center', ncol=3, fontsize=6.5, frameon=False); fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(sys.argv[1] if len(sys.argv) > 1 else 'fig_prod1_rates.pdf')
