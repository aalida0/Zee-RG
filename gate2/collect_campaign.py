"""Collect campaign outputs (condor/campaign.sub), report sampler diagnostics and re-verify points locally with the full gate.
Usage: ZEE_INPUTS=modern python3 -W ignore gate2/collect_campaign.py OUTDIR [TOPK=10] [NSAMPLE=200] [NFINAL=6]
Sample chains: for every (logL, start) basin the chains are compared with the split-R-hat and an autocorrelation ESS (post burn-in stored points);
fractions (e.g. certified scale >= X, tau->mu gamma > Y) are estimated per basin under the stated measure (uniform in the sampling coordinates on
the region R_L) with a chain-bootstrap 68% interval, and flagged `converged` only if R-hat < 1.1 and ESS >= 100.  No fraction is pooled across
basins (their relative weights are not defined by the chains) and none is a probability of nature.
Verification: the best point of every (OBS, logL) group (TOPK) and NSAMPLE random sample points are re-run through accept(collider=True, n_vac=200)
and the 13-variant certification; the NFINAL highest certified points are written for benchmarks.py (100 scales per decade, 17 pieces).
Writes results/campaign_summary.json and results/campaign_finalists.json."""
import sys, os, json, glob, collections
import numpy as np
os.environ.setdefault('ZEE_INPUTS', 'modern')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]


def act(x):
    x = np.asarray(x, float) - np.mean(x); n = len(x)
    if n < 8 or np.allclose(x, 0): return 1.0
    f = np.fft.rfft(x, 2*n); ac = np.fft.irfft(f*np.conj(f))[:n]; ac /= ac[0]; t = 1.0
    for k in range(1, n):
        if ac[k] < 0.05: break
        t += 2*ac[k]
    return max(t, 1.0)


def rhat(chains):
    """split R-hat for a list of 1-d arrays"""
    ch = [np.asarray(c, float) for c in chains if len(c) >= 8]
    if len(ch) < 2: return float('nan')
    ch = [h for c in ch for h in (c[:len(c)//2], c[len(c)//2:2*(len(c)//2)])]; n = min(len(c) for c in ch); ch = np.array([c[:n] for c in ch])
    W = ch.var(axis=1, ddof=1).mean(); B = n*ch.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n - 1)/n*W + B/n)/W)) if W > 0 else float('nan')


def main():
    out = sys.argv[1]; TOPK = int(sys.argv[2]) if len(sys.argv) > 2 else 10; NS = int(sys.argv[3]) if len(sys.argv) > 3 else 200; NF = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    cmin = float(json.load(open(os.path.join(ROOT, 'results', 'refit_hw.json')))['best']['chi2'])
    J = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out, 'camp_*.json')))]
    res = dict(code_hashes=dict(collections.Counter(j['code'] for j in J)), n_files=len(J), groups={}, basins={}, verified=[])
    for j in J:
        g = res['groups'].setdefault('%s|%g' % (j['mode'], j['logL']), dict(jobs=0, prop=0, acc=0, stored=0, wall=0.0, certs=0))
        g['jobs'] += 1; g['prop'] += j['n_prop']; g['acc'] += j['n_acc']; g['stored'] += len(j['points']); g['wall'] += j['wall']; g['certs'] += j.get('n_cert', 0)
    # sampler diagnostics per (logL, start)
    bas = collections.defaultdict(list)
    for j in J:
        if j['mode'] == 'sample' and j['points']: bas[(j['logL'], j['start'])].append(j)
    rng = np.random.default_rng(0)
    for (L_, st), chs in sorted(bas.items()):
        q = {'log_tmg': lambda p: np.log10(p['obs']['tau_mu_g']), 'log_m1': lambda p: np.log10(p['masses']['m1']), 'cert': lambda p: p['cert']['min']}
        d = dict(n_chains=len(chs), n_points=sum(len(c['points']) for c in chs))
        for nm, f in q.items():
            ser = [[f(p) for p in c['points']] for c in chs]
            ess = sum(len(s)/act(s) for s in ser); d['rhat_' + nm] = rhat(ser); d['ess_' + nm] = float(ess)
        d['converged'] = bool(all(d['rhat_' + n_] < 1.1 for n_ in q) and min(d['ess_' + n_] for n_ in q) >= 100)
        for X in (10.0, 11.0, 12.0):                      # fraction of the basin with certified scale >= X (5-variant minimum)
            per = [np.mean([p['cert']['min'] >= X for p in c['points']]) for c in chs]
            bs = [np.mean(rng.choice(per, len(per))) for _ in range(300)]
            d['frac_cert_ge_%g' % X] = dict(mean=float(np.mean(per)), lo=float(np.percentile(bs, 16)), hi=float(np.percentile(bs, 84)))
        res['basins']['logL=%g|start=%d' % (L_, st)] = d
    # verification
    import accept as AC, cert as CE
    def verify(group, p, logL):
        x, eta = np.array(p['x']), np.array(p['eta'])
        r = AC.accept(x, eta, cmin, logL=None, collider=True, n_vac=200); cs, cd = CE.cert(x, eta, CE.FULL)
        ok = (r['status'] == 'PASS') and cs >= logL
        return dict(group=group, logL=logL, status=r['status'], failed=r.get('failed_cuts'), certified13=cs, meets_cutoff=bool(ok), chi2=r['chi2'],
                    obs={k: r['obs'][k] for k in ('mu_e_g', 'tau_mu_g', 'tau_e_g', 'h_tau_mu')}, x=list(map(float, x)), eta=list(map(float, eta)))
    cands = collections.defaultdict(list)
    for j in J:
        if j['best']: cands['%s|%g' % (j['mode'], j['logL'])].append((j['best'], j['logL']))
    for k, L_ in cands.items():
        key = k.split('|')[0].split('-')[1]
        for b, lg in sorted(L_, key=lambda t: -t[0]['obs'][key])[:TOPK]: res['verified'].append(verify(k, b, lg))
    pool = [(p, j['logL']) for j in J if j['mode'] == 'sample' for p in j['points']]
    for i in (rng.choice(len(pool), size=min(NS, len(pool)), replace=False) if pool else []): res['verified'].append(verify('sample', *pool[i]))
    res['n_verified'] = len(res['verified']); res['n_pass'] = sum(v['meets_cutoff'] for v in res['verified'])
    fin = sorted([v for v in res['verified'] if v['meets_cutoff']], key=lambda v: -v['certified13'])[:NF]
    json.dump(res, open(os.path.join(ROOT, 'results', 'campaign_summary.json'), 'w'), indent=1, default=float)
    json.dump({'C%d' % i: dict(x=v['x'], eta=v['eta'], origin='campaign ' + v['group']) for i, v in enumerate(fin)}, open(os.path.join(ROOT, 'results', 'campaign_finalists.json'), 'w'), indent=1)
    print(json.dumps(res['groups'], indent=0)[:1500]); print('verified', res['n_verified'], 'meet cutoff', res['n_pass'], 'finalists', len(fin))


if __name__ == '__main__': main()
