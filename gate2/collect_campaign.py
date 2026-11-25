"""Collect campaign outputs (condor/campaign.sub), report sampler diagnostics and re-verify points locally with the full gate.
Usage: ZEE_INPUTS=modern python3 -W ignore gate2/collect_campaign.py OUTDIR [TOPK=10] [NSAMPLE=200] [NFINAL=6]
Sample chains: for every (logL, start) basin the chains are compared with the split-R-hat and an autocorrelation ESS (post burn-in stored points);
fractions (e.g. certified scale >= X, tau->mu gamma > Y) are estimated per basin under the stated measure (uniform in the sampling coordinates on
the region R_L) with a chain-bootstrap 68% interval, and flagged `converged` only if R-hat < 1.1 and ESS >= 100.  No fraction is pooled across
basins (their relative weights are not defined by the chains) and none is a probability of nature.
Verification: the best point of every (OBS, logL) group (TOPK) and NSAMPLE random sample points are re-run through accept(collider=True, n_vac=200)
and the 13-variant certification; the NFINAL highest certified points are written for benchmarks.py (100 scales per decade, 17 pieces).
Writes results/campaign_summary.json and results/campaign_finalists.json; with PLOTDIR=<dir> also scan_lfv_cutoff.pdf, scan_baseline_vs_filtered.pdf and
scan_failure_mechanisms.pdf (maps of what the chains visited, not probabilities)."""
import sys, os, json, glob, collections
import numpy as np
os.environ.setdefault('ZEE_INPUTS', 'modern')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]


def act(x):
    x = np.asarray(x, float) - np.mean(x); n = len(x)
    if np.allclose(x, 0): return float(max(n, 1))           # a constant (stuck) chain carries one effective sample, not n
    if n < 8: return 1.0
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


def make_plots(J, outdir):
    """Figures from the stored states of sample chains (all points, repeats included, i.e. per store event) and the verified best points of max chains.
    The baseline (logL = 0, no high-scale requirement) is shown against the cutoff-filtered chains; every panel is a map of what the chains visited from their
    declared starts, not a probability."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    pts = [(j['logL'], p) for j in J if j['mode'] == 'sample' for p in j['series'] if p.get('cert')]
    if not pts: return
    L = np.array([a for a, _ in pts]); cert = np.array([p['cert']['min'] for _, p in pts]); tmg = np.log10([p['obs']['tau_mu_g'] for _, p in pts])
    meg = np.log10([max(p['obs']['mu_e_g'], 1e-30) for _, p in pts]); m1 = np.array([p['masses']['m1'] for _, p in pts]); cba = np.array([abs(p['masses']['cba']) for _, p in pts])
    cos2 = np.cos([p['masses']['phi'] for _, p in pts])**2
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
    ax[0].scatter(cert, tmg, s=5, c=L, cmap='viridis'); ax[0].axhline(-9, c='gray', ls='--', lw=0.8); ax[0].set_xlabel(r'minimum demonstrated $\log_{10}\Lambda$'); ax[0].set_ylabel(r'$\log_{10}\mathrm{Br}(\tau\to\mu\gamma)$')
    ax[1].scatter(cert, meg, s=5, c=L, cmap='viridis'); ax[1].axhline(np.log10(6e-14), c='gray', ls='--', lw=0.8); ax[1].set_xlabel(r'minimum demonstrated $\log_{10}\Lambda$'); ax[1].set_ylabel(r'$\log_{10}\mathrm{Br}(\mu\to e\gamma)$')
    sc = ax[2].scatter(cba, cert, s=5, c=L, cmap='viridis'); ax[2].set_xlabel(r'$|c_{\beta-\alpha}|$'); ax[2].set_ylabel(r'minimum demonstrated $\log_{10}\Lambda$'); fig.colorbar(sc, ax=ax[2], label='requested cutoff of the chain')
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'scan_lfv_cutoff.pdf')); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for lo, hi, c, lab in ((0, 0.5, 'C7', 'baseline (no cutoff requirement)'), (5.5, 99, 'C3', 'chains requiring a cutoff of $10^6$ GeV or more')):
        m = (L >= lo) & (L <= hi) if lo == 0 else (L >= lo)
        ax[0].scatter(m1[m], cos2[m], s=4, c=c, label=lab, alpha=0.6); ax[1].hist(cert[m], bins=30, color=c, alpha=0.5)
    ax[0].set_xlabel(r'$m_{h_1^+}$ [GeV]'); ax[0].set_ylabel(r'doublet fraction $\cos^2\varphi$ of $h_1^+$'); ax[0].legend(fontsize=6, frameon=False); ax[1].set_xlabel(r'minimum demonstrated $\log_{10}\Lambda$')
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'scan_baseline_vs_filtered.pdf')); plt.close(fig)
    fails = collections.Counter(v['fail'] or 'none' for _, p in pts for v in p['cert']['variants'].values())
    fig, ax = plt.subplots(figsize=(4.2, 2.8)); ax.bar(list(fails), list(fails.values())); ax.set_ylabel('variant evaluations'); ax.set_title('first-failure mechanism', fontsize=9); plt.xticks(rotation=30, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'scan_failure_mechanisms.pdf')); plt.close(fig)


def main():
    out = sys.argv[1]; TOPK = int(sys.argv[2]) if len(sys.argv) > 2 else 10; NS = int(sys.argv[3]) if len(sys.argv) > 3 else 200; NF = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    cmin = float(json.load(open(os.path.join(ROOT, 'results', 'refit_hw.json')))['best']['chi2'])
    J = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out, 'camp_*.json')))]
    man = collections.Counter((j['manifest']['code'], j['manifest']['data'], j['manifest']['run_id'], j['manifest']['commit']) for j in J)
    res = dict(manifests={str(k): v for k, v in man.items()}, n_files=len(J), n_complete=sum(bool(j.get('complete')) for j in J),
               n_dirty=sum(bool(j['manifest'].get('dirty')) for j in J), groups={}, basins={}, verified=[])
    if len(man) > 1 and not os.environ.get('ALLOW_MIXED'): sys.exit('mixed code/data/run-id/commit in %s: %s (set ALLOW_MIXED=1 to override)' % (out, dict(man)))
    for j in J:                                           # expand the proposal-indexed trace: repeated states appear with their dwell counts
        rec = {p['id']: p for p in j['points']}
        j['series'] = [rec[i] for i in j.get('trace', [])] if 'trace' in j else j['points']
    for j in J:
        g = res['groups'].setdefault('%s|%g' % (j['mode'], j['logL']), dict(jobs=0, complete=0, prop=0, acc=0, stored=0, unique=0, wall=0.0, certs=0))
        g['jobs'] += 1; g['complete'] += bool(j.get('complete')); g['prop'] += j['n_prop']; g['acc'] += j['n_acc']; g['stored'] += len(j['series']); g['unique'] += len(j['points'])
        g['wall'] += j['wall']; g['certs'] += j.get('n_cert', 0)
    if os.environ.get('PLOTDIR'): make_plots(J, os.environ['PLOTDIR'])
    # sampler diagnostics per (logL, start)
    bas = collections.defaultdict(list)
    for j in J:
        if j['mode'] == 'sample' and j['series']: bas[(j['logL'], j['start'])].append(j)
    rng = np.random.default_rng(0)
    for (L_, st), chs in sorted(bas.items()):
        q = {'log_tmg': lambda p: np.log10(p['obs']['tau_mu_g']), 'log_m1': lambda p: np.log10(p['masses']['m1']), 'cert': lambda p: p['cert']['min']}
        d = dict(n_chains=len(chs), n_points=sum(len(c['series']) for c in chs), n_stuck=sum(len({p['id'] for p in c['series']}) < 2 for c in chs), n_incomplete=sum(not c.get('complete') for c in chs))
        for nm, f in q.items():
            ser = [[f(p) for p in c['series']] for c in chs]
            ess = sum(len(s)/act(s) for s in ser); d['rhat_' + nm] = rhat(ser); d['ess_' + nm] = float(ess)
        d['converged'] = bool(all(d['rhat_' + n_] < 1.1 for n_ in q) and min(d['ess_' + n_] for n_ in q) >= 100)
        for X in (10.0, 11.0, 12.0):                      # fraction of the basin with certified scale >= X (5-variant minimum)
            per = [np.mean([p['cert']['min'] >= X for p in c['series']]) for c in chs]
            bs = [np.mean(rng.choice(per, len(per))) for _ in range(300)]
            d['frac_cert_ge_%g' % X] = dict(mean=float(np.mean(per)), lo=float(np.percentile(bs, 16)), hi=float(np.percentile(bs, 84)))
        res['basins']['logL=%g|start=%d' % (L_, st)] = d
    # verification
    import accept as AC, cert as CE
    def verify(group, p, logL):
        x, eta = np.array(p['x']), np.array(p['eta'])
        r = AC.accept(x, eta, cmin, logL=None, collider=True, n_vac=200); cs, cd = CE.cert(x, eta, CE.FULL)
        ok = (r['status'] == 'PASS') and bool(np.isfinite(cs)) and cs >= logL
        return dict(group=group, logL=logL, status=r['status'], failed=r.get('failed_cuts'), certified13=cs, meets_cutoff=bool(ok), chi2=r['chi2'],
                    obs={**{k: r['obs'][k] for k in ('mu_e_g', 'tau_mu_g', 'tau_e_g')}, 'h_tau_mu': r.get('collider', {}).get('BR_h_tau_mu', r['obs']['h_tau_mu'])}, x=list(map(float, x)), eta=list(map(float, eta)))
    cands = collections.defaultdict(list)
    for j in J:                                           # only fully passing bests are ranked; unresolved ones are listed separately
        if j['best']: cands['%s|%g' % (j['mode'], j['logL'])].append((j['best'], j['logL']))
        if j.get('best_unresolved'): res.setdefault('unresolved_bests', []).append(dict(group='%s|%g' % (j['mode'], j['logL']), obs=j['best_unresolved']['obs']))
    for k, L_ in cands.items():
        key = k.split('|')[0].split(':')[1]
        for b, lg in sorted(L_, key=lambda t: -t[0]['obs'][key])[:TOPK]: res['verified'].append(verify(k, b, lg))
    pool = [(p, j['logL']) for j in J if j['mode'] == 'sample' for p in j['series']]
    for i in (rng.choice(len(pool), size=min(NS, len(pool)), replace=False) if pool else []): res['verified'].append(verify('sample', *pool[i]))
    res['n_verified'] = len(res['verified']); res['n_pass'] = sum(v['meets_cutoff'] for v in res['verified'])
    good = [v for v in res['verified'] if v['meets_cutoff']]
    fin = sorted(good, key=lambda v: -v['certified13'])[:max(NF//2, 1)]           # highest demonstrated scales ...
    for obs_ in ('mu_e_g', 'tau_mu_g', 'tau_e_g', 'h_tau_mu'):                      # ... plus the best verified witness of each observable
        w = sorted([v for v in good if v['obs'][obs_] is not None], key=lambda v: -v['obs'][obs_])[:1]
        fin += [v for v in w if v not in fin]
    from campaign import clean
    json.dump(clean(res), open(os.path.join(ROOT, 'results', 'campaign_summary.json'), 'w'), indent=1, allow_nan=False)
    json.dump({'C%d' % i: dict(x=v['x'], eta=v['eta'], origin='campaign ' + v['group']) for i, v in enumerate(fin)}, open(os.path.join(ROOT, 'results', 'campaign_finalists.json'), 'w'), indent=1)
    print(json.dumps(res['groups'], indent=0)[:1500]); print('verified', res['n_verified'], 'meet cutoff', res['n_pass'], 'finalists', len(fin))


if __name__ == '__main__': main()
