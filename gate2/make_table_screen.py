"""LaTeX rows of the matched screen of the 37 finalists flagged robust by the matched-margin proxy (results/prod1/screen_ep, written by gate2/validate_finalist.py with
--skip yukawa,vacuum --modes ep). Columns: id, tree-level minimum over the 13 variants, minimum over the re-matched variants (effective-potential matching; defined variants only),
minimum including the stress variants, BFB margin V_red of the tree-level boundary, the three rates, number of variants whose matched boundary is unbounded at the start, verdict.
Clean = complete (13 defined variants), no start failure, stress minimum >= 10.   Usage: python3 gate2/make_table_screen.py > /tmp/screen_rows.tex"""
import json, os, glob, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P = os.path.join(ROOT, 'results', 'prod1')
S = json.load(open(os.path.join(P, 'campaign_summary.json'))); F = json.load(open(os.path.join(P, 'campaign_finalists.json')))
key = lambda x: tuple(np.round(x, 12)); idx = {key(v['x']): v for v in S['verified']}
def sci(v):
    if v <= 0: return '0'
    e = int(np.floor(np.log10(v))); m = v/10**e
    if round(m, 1) >= 10: m, e = m/10, e + 1
    return '%.1f\\!\\times\\!10^{%d}' % (m, e)
ids = sorted((os.path.basename(f)[9:-5] for f in glob.glob(os.path.join(P, 'screen_ep', 'validate_C*.json'))), key=lambda s: int(s[1:]))
nclean = 0
for n in ids:
    d = json.load(open(os.path.join(P, 'screen_ep', 'validate_%s.json' % n))); m = d['matching']['ep']; v = idx[key(F[n]['x'])]; o = v['obs']
    clean = m['complete'] and m['n_start_fail'] == 0 and m['min_stress'] is not None and m['min_stress'] >= 10; nclean += clean
    f1 = lambda x: '--' if x is None else '%.1f' % x
    print('%s & %.1f & %s & %s & %.3f & $%s$ & $%s$ & $%s$ & %d & %s\\\\' % (n, d['tree']['min'], f1(m['min_rematched'] if m['complete'] or m['n_start_fail'] == 0 else None), f1(m['min_stress'] if m['complete'] or m['n_start_fail'] == 0 else None), d['margins']['vred'], sci(o['mu_e_g']), sci(o['tau_mu_g']), sci(o['h_tau_mu']), m['n_start_fail'], 'clean' if clean else 'no'))
print('%% %d of %d clean' % (nclean, len(ids)))
