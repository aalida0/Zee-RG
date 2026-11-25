"""Collect campaign outputs (condor/campaign.sub) and re-verify the interesting points locally with the full gate.
Usage: ZEE_INPUTS=modern python3 -W ignore gate2/collect_campaign.py OUTDIR [TOPK=20] [NSAMPLE=300]
Writes results/campaign_summary.json: per (mode, logL) job counts, proposals, accepted fraction, stored points; the best point per
(OBS, logL) and a random sample of region points, each re-run through accept(collider=True, n_vac=200) with the cutoff logL requested
(status must be PASS).  Counts of points are chain-sample diagnostics (correlated random walks), not probabilities."""
import sys, os, json, glob, collections
import numpy as np
os.environ.setdefault('ZEE_INPUTS', 'modern')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
import accept as AC
out = sys.argv[1]; TOPK = int(sys.argv[2]) if len(sys.argv) > 2 else 20; NS = int(sys.argv[3]) if len(sys.argv) > 3 else 300
cmin = float(json.load(open(os.path.join(ROOT, 'results', 'refit_hw.json')))['best']['chi2'])
J = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out, 'camp_*.json')))]
codes = collections.Counter(j['code'] for j in J)
summ = collections.defaultdict(lambda: dict(jobs=0, prop=0, acc=0, stored=0, wall=0.0)); cands = collections.defaultdict(list); samp = []
for j in J:
    k = '%s|%g' % (j['mode'], j['logL']); s = summ[k]; s['jobs'] += 1; s['prop'] += j['n_prop']; s['acc'] += j['n_acc']; s['stored'] += len(j['points']); s['wall'] += j['wall']
    if j['best']: cands[k].append(j['best'])
    for p in j['points']: samp.append((j['logL'], p))
rng = np.random.default_rng(0); res = dict(code_hashes=dict(codes), summary=summ, verified=[])
def verify(logL, p):
    r = AC.accept(p['x'], p['eta'], cmin, logL=logL, collider=True, n_vac=200)
    return dict(logL=logL, status=r['status'], failed=r.get('failed_cuts'), chi2=r['chi2'], Lf=r.get('Lf'), obs={k: r['obs'][k] for k in ('mu_e_g', 'tau_mu_g', 'tau_e_g', 'h_tau_mu')}, x=list(p['x']), eta=list(p['eta']))
for k, L_ in cands.items():
    key = k.split('|')[0].split('-')[1]; top = sorted(L_, key=lambda b: -b['obs'][key])[:TOPK]
    res['verified'] += [dict(group=k, **verify(float(k.split('|')[1]), b)) for b in top]
for i in rng.choice(len(samp), size=min(NS, len(samp)), replace=False) if samp else []:
    L_, p = samp[i]; res['verified'].append(dict(group='sample', **verify(L_, p)))
res['n_verified'] = len(res['verified']); res['n_pass'] = sum(v['status'] == 'PASS' for v in res['verified'])
json.dump(res, open(os.path.join(ROOT, 'results', 'campaign_summary.json'), 'w'), indent=1, default=float)
print({k: dict(v) for k, v in summ.items()}); print('verified', res['n_verified'], 'PASS', res['n_pass'], 'codes', dict(codes))
