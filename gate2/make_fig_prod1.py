"""Figure of the prod1 campaign: LFV rates versus scale for the re-verified points (grey), the 37 finalists flagged robust by the matched-margin proxy, their matched
minimum under the effective-potential matching (results/prod1/screen_ep) and the validated witnesses BM1, BM2, BP1, BP3 (stress-inclusive minimum over the defined variants of the
validation records, hollow marker if a prescription has undefined variants).
Conventions (report v12): points are identified by the full vector x+eta; the tree scale is the fine-grid one of the screen records; a finalist whose matched boundary is unbounded at the
matching scale in every variant (no scale) is drawn at the left margin in a separate 'unbounded' category, never at an invented scale; a finalist whose tree scale collapses under the 0.9
quartic stress (C40) is joined to its stress minimum by an arrow.  Horizontal lines: present 90% CL limits and projected sensitivities as in Table 1 of the draft (MEG II 1.5e-13, MEG II design
6e-14, Belle 4.2e-8, Belle II reach 1e-9, CMS h->tau mu 1.5e-3).
Usage: python3 gate2/make_fig_prod1.py OUT.pdf"""
import json, os, sys, glob, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P = os.path.join(ROOT, 'results', 'prod1')
S = json.load(open(os.path.join(P, 'campaign_summary.json'))); F = json.load(open(os.path.join(P, 'campaign_finalists.json')))
key = lambda x, eta: tuple(np.round(np.concatenate([np.asarray(x, float), np.asarray(eta, float)]), 12))      # full vector x+eta, not x alone
idx = {key(v['x'], v['eta']): v for v in S['verified']}
V = [v for v in S['verified'] if v['status'] == 'PASS' and v['screen13'] is not None]
OBS = (('mu_e_g', r'$\mathrm{Br}(\mu\to e\gamma)$', ((1.5e-13, 'MEG II limit'), (6e-14, 'MEG II design'))),
       ('tau_mu_g', r'$\mathrm{Br}(\tau\to\mu\gamma)$', ((4.2e-8, 'Belle limit'), (1e-9, 'Belle II reach'))),
       ('h_tau_mu', r'$\mathrm{Br}(h\to\tau\mu)$', ((1.5e-3, 'CMS limit'),)))
XU = 1.0                                                                                                        # x position of the 'unbounded at matching scale' category
fin = {}
for f in glob.glob(os.path.join(P, 'screen_ep', 'validate_C*.json')):
    n = os.path.basename(f)[9:-5]; d = json.load(open(f)); m = d['matching']['ep']; v = idx[key(F[n]['x'], F[n]['eta'])]
    clean = m['complete'] and m['n_start_fail'] == 0 and m['min_stress'] is not None and m['min_stress'] >= 10
    # no scale at all (every variant unbounded at the matching scale): separate category; a partial minimum over the defined variants is drawn as such
    fin[n] = dict(tree=d['tree']['min'], ms=m['min_stress'], n_def=m['n_defined'], clean=clean, obs=v['obs'])
BP = {'BP1': 'C42', 'BP3': 'C35'}
REF = {}                                                       # validated witnesses: stress-inclusive minimum over the defined variants of all prescriptions in the validation record
B0 = json.load(open(os.path.join(ROOT, 'results', 'benchmarks.json')))
def _ref(vr, obs):
    mn = [m['min_stress'] for m in vr['matching'].values() if m['min_stress'] is not None]
    return dict(ms=min(mn), partial=not (len(vr['matching']) == 7 and all(m['complete'] for m in vr['matching'].values())), obs=obs)
for n in ('BM1', 'BM2'):
    REF[n] = _ref(json.load(open(os.path.join(ROOT, 'results', 'validation', 'validate_%s.json' % n))), dict(B0[n]['observables'], h_tau_mu=B0[n]['collider']['BR_h_tau_mu']))
for b_, c_ in BP.items(): REF[b_] = _ref(json.load(open(os.path.join(P, 'validate_%s.json' % c_))), fin[c_]['obs'])
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.4))
for a, (k, lab, lines) in zip(ax, OBS):
    a.scatter([v['screen13'] for v in V], [max(v['obs'][k], 1e-19) for v in V], s=5, c='0.75', lw=0, label='verified points')
    for yl, tl in lines: a.axhline(yl, c='0.4', lw=0.6, ls=':'); a.text(16.4, yl, tl, fontsize=5.5, ha='right', va='bottom', color='0.3')
    for n, r in fin.items():
        y = max(r['obs'][k], 1e-19); col = '#009E73' if r['clean'] else '#D55E00'
        if r['ms'] is None:                                                                         # no scale: category, not a number
            a.scatter([XU], [y], s=14, c=col, marker='v', lw=0, zorder=3); continue
        a.plot([r['tree'], r['ms']], [y, y], c=col, lw=0.5, alpha=0.6)
        a.scatter([r['ms']], [y], s=14, c=col, marker='o' if r['n_def'] == 13 else 'D', lw=0, zorder=3)    # diamond: minimum over a subset of the variants (some are undefined)
        a.scatter([r['tree']], [y], s=14, c=col, marker='x', lw=0.8, zorder=3)
        if r['tree'] - r['ms'] > 5: a.annotate('', (r['ms'], y), (r['tree'], y), arrowprops=dict(arrowstyle='->', color=col, lw=0.8))         # collapse under the matched stress (C39, C40)
    for b, r in REF.items():
        y = max(r['obs'][k], 1e-19)
        a.scatter([r['ms']], [y], s=30, facecolors='none' if r['partial'] else '#0072B2', edgecolors='#0072B2', marker='*', lw=0.8, zorder=4); a.annotate(b, (r['ms'], y), fontsize=7, xytext=(3, -8), textcoords='offset points', color='#0072B2')
    a.set_yscale('log'); a.set_xlabel(r'$\log_{10}(\Lambda/\mathrm{GeV})$'); a.set_ylabel(lab); a.set_xlim(0.2, 16.5); a.set_xticks([XU, 4, 8, 12, 16]); a.set_xticklabels(['unb.', '4', '8', '12', '16'])
ax[0].set_ylim(1e-19, 3e-13); ax[1].set_ylim(1e-10, 5e-8); ax[2].set_ylim(1e-9, 3e-3)
h = [plt.Line2D([], [], c='0.75', marker='o', ls='', ms=3, label='re-verified points (screening scale)'),
     plt.Line2D([], [], c='#009E73', marker='x', ls='', ms=4, label='finalist, tree scale (clean under EP matching)'),
     plt.Line2D([], [], c='#009E73', marker='o', ls='', ms=3, label='same, matched minimum incl. stress variants'),
     plt.Line2D([], [], c='#D55E00', marker='x', ls='', ms=4, label='finalist, tree scale (not clean)'),
     plt.Line2D([], [], c='#D55E00', marker='D', ls='', ms=3, label='same, minimum over the defined variants only'),
     plt.Line2D([], [], c='#D55E00', marker='v', ls='', ms=3, label='unbounded at the matching scale in all variants (no scale)'),
     plt.Line2D([], [], c='#0072B2', marker='*', ls='', ms=5, label='BM1, BM2, BP1, BP3: stress-inclusive minimum, 7 prescriptions (hollow: undefined variants)')]
fig.legend(handles=h, loc='lower center', ncol=3, fontsize=6, frameon=False); fig.tight_layout(rect=(0, 0.15, 1, 1)); fig.savefig(sys.argv[1] if len(sys.argv) > 1 else 'fig_prod1_rates.pdf')
