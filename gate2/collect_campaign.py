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


CATS = ((-1.5, -0.5, 'C7', 'logL=-1: nominal-boundary baseline (low-energy fit + nominal BFB/unitarity at the matching scale)'),
        (-0.5, 0.5, 'C0', 'logL=0: rescaled-boundary reference (x1.1, x0.9 also valid at the start)'),
        (5.5, 99, 'C3', 'chains requiring a cutoff of 1e6 GeV or more'))


def rank_norm(chains):
    """rank-normalise (pooled ranks -> normal scores), Vehtari et al. 2021, so that R-hat and ESS are robust to heavy tails"""
    from scipy.stats import rankdata, norm
    allv = np.concatenate([np.asarray(c, float) for c in chains]); r = rankdata(allv); z = norm.ppf((r - 0.375)/(len(allv) + 0.25)); out = []; k = 0
    for c in chains: out.append(z[k:k + len(c)]); k += len(c)
    return out


MIN_AFTER_ENTRY = int(os.environ.get('MIN_AFTER_ENTRY', 2000))     # proposals required after the chain entered the target and finished adaptation


def classify(j):
    """COMPLETE_TARGET_SAMPLED: all proposals done, in the target long enough after burn-in; COMPLETE_NO_TARGET: all proposals done but the target was
    never reached or reached too late; TIMEOUT: wall limit; NUMERICAL_FAILURE: most proposals rejected for numerical/collider errors.  A chain that was
    never in the target contributes no sampling claim, but it stays in the diagnostics (nothing is discarded silently)."""
    rj = j.get('rejects', {}); nerr = sum(v for k, v in rj.items() if k.startswith('hcoll_error') or k in ('nonfinite',))
    if j['n_prop'] and nerr > 0.5*j['n_prop']: return 'NUMERICAL_FAILURE'
    if not j.get('complete'): return 'TIMEOUT'
    ne = j.get('n_enter'); burn_n = j.get('burn', 0.25)*j.get('nprop_target', j['n_prop'])
    if ne is None or j.get('climb_at_end'): return 'COMPLETE_NO_TARGET'
    return 'COMPLETE_TARGET_SAMPLED' if j['n_prop'] - max(ne, burn_n) >= MIN_AFTER_ENTRY else 'COMPLETE_NO_TARGET'


def make_plots(J, outdir, only_pass=False):
    """Maps of what the chains visited (not probabilities).  only_pass=False: every stored state of the sample chains, x axis = SCREENING reach (the
    coarse 5-variant minimum of the stored record, not the final fine-grid result); only_pass=True: only states whose full stage record passed
    (final_status PASS: vacuum + collider + requested cutoff).  The three chain categories (-1 nominal baseline, 0 rescaled reference, high cutoff)
    are kept distinct.  Re-verified points (fine grid, 13 variants) are plotted separately by make_verified_plot."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True); pre = 'pass_' if only_pass else 'raw_'
    pts = [(j['logL'], p) for j in J if j['mode'] == 'sample' for p in j['series'] if p.get('cert') and (not only_pass or p.get('final_status') == 'PASS')]
    if not pts: return
    L = np.array([a for a, _ in pts]); cert = np.array([p['cert']['min'] for _, p in pts]); tmg = np.log10([p['obs']['tau_mu_g'] for _, p in pts])
    meg = np.log10([max(p['obs']['mu_e_g'], 1e-30) for _, p in pts]); m1 = np.array([p['masses']['m1'] for _, p in pts]); cba = np.array([abs(p['masses']['cba']) for _, p in pts])
    cos2 = np.cos([p['masses']['phi'] for _, p in pts])**2; what = 'gate-passing stored states' if only_pass else 'all stored states (diagnostic)'
    xl = r'screening reach $\log_{10}\Lambda$ (coarse 5-variant minimum)'
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
    ax[0].scatter(cert, tmg, s=5, c=L, cmap='viridis'); ax[0].axhline(-9, c='gray', ls='--', lw=0.8); ax[0].set_xlabel(xl); ax[0].set_ylabel(r'$\log_{10}\mathrm{Br}(\tau\to\mu\gamma)$')
    ax[1].scatter(cert, meg, s=5, c=L, cmap='viridis'); ax[1].axhline(np.log10(6e-14), c='gray', ls='--', lw=0.8); ax[1].set_xlabel(xl); ax[1].set_ylabel(r'$\log_{10}\mathrm{Br}(\mu\to e\gamma)$')
    sc = ax[2].scatter(cba, cert, s=5, c=L, cmap='viridis'); ax[2].set_xlabel(r'$|c_{\beta-\alpha}|$'); ax[2].set_ylabel(xl); fig.colorbar(sc, ax=ax[2], label='logL of the chain (-1, 0: baselines)')
    fig.suptitle(what, fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(outdir, pre + 'scan_lfv_cutoff.pdf')); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for lo, hi, c, lab in CATS:
        m = (L >= lo) & (L <= hi)
        if m.any(): ax[0].scatter(m1[m], cos2[m], s=4, c=c, label=lab.split(':')[0] if lab.startswith('logL') else 'logL>=6', alpha=0.6); ax[1].hist(cert[m], bins=30, color=c, alpha=0.5)
    ax[0].set_xlabel(r'$m_{h_1^+}$ [GeV]'); ax[0].set_ylabel(r'doublet fraction $\cos^2\varphi$ of $h_1^+$'); ax[0].legend(fontsize=6, frameon=False); ax[1].set_xlabel(xl)
    fig.suptitle(what + ': local exploration from the declared starts', fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(outdir, pre + 'scan_baseline_vs_filtered.pdf')); plt.close(fig)
    if not only_pass:
        fails = collections.Counter(v['fail'] or 'none' for _, p in pts for v in p['cert']['variants'].values())
        fig, ax = plt.subplots(figsize=(4.2, 2.8)); ax.bar(list(fails), list(fails.values())); ax.set_ylabel('variant evaluations'); ax.set_title('first-failure mechanism (screening variants)', fontsize=9); plt.xticks(rotation=30, fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, 'scan_failure_mechanisms.pdf')); plt.close(fig)


def make_verified_plot(verified, outdir):
    """Re-verified points only (accept(collider=True, n_vac=200) and the 13-variant certification on the fine grid): the quantity quoted in results"""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    good = [v for v in verified if v['meets_cutoff'] and v['certified13'] is not None and v['certified13'] > 0]
    if not good: return
    os.makedirs(outdir, exist_ok=True); fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for a, (k, lim) in zip(ax, (('mu_e_g', 1.5e-13), ('tau_mu_g', 4.2e-8), ('h_tau_mu', 1.8e-3))):
        a.scatter([v['certified13'] for v in good], np.log10([max(v['obs'][k], 1e-30) for v in good]), s=8, c=[v['logL'] for v in good], cmap='viridis'); a.axhline(np.log10(lim), c='gray', ls='--', lw=0.8)
        a.set_xlabel(r'certified $\log_{10}\Lambda$ (fine grid, 13 variants)'); a.set_ylabel(r'$\log_{10}$ ' + k)
    fig.suptitle('re-verified points (full gate)', fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(outdir, 'verified_lfv_cutoff.pdf')); plt.close(fig)


def main():
    out = sys.argv[1]; TOPK = int(sys.argv[2]) if len(sys.argv) > 2 else 10; NS = int(sys.argv[3]) if len(sys.argv) > 3 else 200; NF = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    cmin = float(json.load(open(os.path.join(ROOT, 'results', 'refit_hw.json')))['best']['chi2'])
    J = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out, 'camp_*.json')))]
    man = collections.Counter((j['manifest']['code'], j['manifest']['data'], j['manifest']['run_id'], j['manifest']['commit'],
                               json.dumps(j['manifest'].get('env'), sort_keys=True), json.dumps(j['manifest'].get('collider_data'), sort_keys=True),
                               json.dumps(j['manifest'].get('packages'), sort_keys=True)) for j in J)
    envs = {json.dumps(j['manifest'].get('env'), sort_keys=True) for j in J}
    for e_ in envs:                                       # re-verification must use the same input set as the chains
        e_d = json.loads(e_) or {}
        if e_d.get('ZEE_INPUTS') and e_d['ZEE_INPUTS'] != os.environ.get('ZEE_INPUTS'): sys.exit('collector ZEE_INPUTS=%s differs from the chains (%s)' % (os.environ.get('ZEE_INPUTS'), e_d['ZEE_INPUTS']))
        if (e_d.get('ZEE_SUMM_LIMIT') or None) != (os.environ.get('ZEE_SUMM_LIMIT') or None): sys.exit('collector ZEE_SUMM_LIMIT differs from the chains (%s)' % e_d.get('ZEE_SUMM_LIMIT'))
    import scipy as _sp
    mm = [j['manifest'] for j in J]
    res_env = dict(collector=dict(python=sys.version.split()[0], numpy=np.__version__, scipy=_sp.__version__),
                   chains=sorted({(m_.get('python'), m_.get('numpy'), m_.get('scipy')) for m_ in mm}, key=str), no_collider=sum(not j.get('higgs') for j in J))
    res = dict(manifests={str(k): v for k, v in man.items()}, n_files=len(J), n_complete=sum(bool(j.get('complete')) for j in J),
               n_dirty=sum(bool(j['manifest'].get('dirty')) for j in J), groups={}, basins={}, verified=[], environment=res_env)
    import campaign as C
    mine = C.manifest(); res['collector_manifest'] = {k: mine.get(k) for k in ('code', 'data', 'commit', 'collider_data', 'packages', 'hw_scale', 'dirty')}
    mism = {k: [j['manifest'].get(k) for j in J][0] for k in ('code', 'data', 'commit', 'collider_data', 'packages', 'hw_scale') if any(j['manifest'].get(k) != mine.get(k) for j in J)}
    if mism and not os.environ.get('ALLOW_MIXED'): sys.exit('collector checkout/data/package manifest differs from the chains in %s (re-verify from the frozen run checkout with the same ZEE_DATA; ALLOW_MIXED=1 overrides)' % sorted(mism))
    if any(j['manifest'].get('dirty') for j in J) and not os.environ.get('ALLOW_DIRTY'): sys.exit('chains were run from a dirty checkout (ALLOW_DIRTY=1 overrides)')
    if any(not j.get('higgs') for j in J) and not os.environ.get('ALLOW_NO_COLLIDER'): sys.exit('chains without an initialised HiggsTools collider stage (ALLOW_NO_COLLIDER=1 overrides)')
    ids = collections.Counter((j['mode'], j['logL'], j['seed']) for j in J); dup = [k for k, v in ids.items() if v > 1]; res['duplicate_jobs'] = [list(k) for k in dup]
    if dup: sys.exit('duplicate job outputs: %s' % dup[:5])
    if os.environ.get('EXPECT') and int(os.environ['EXPECT']) != len(J): sys.exit('expected %s job outputs, found %d (missing or extra jobs)' % (os.environ['EXPECT'], len(J)))
    if len(man) > 1 and not os.environ.get('ALLOW_MIXED'): sys.exit('mixed code/data/run-id/commit in %s: %s (set ALLOW_MIXED=1 to override)' % (out, dict(man)))
    for j in J:                                           # expand the proposal-indexed trace: repeated states appear with their dwell counts
        rec = {p['id']: p for p in j['points']}
        j['series'] = [rec[i] for i in j.get('trace', [])] if 'trace' in j else j['points']
    for j in J:
        g = res['groups'].setdefault('%s|%g' % (j['mode'], j['logL']), dict(jobs=0, complete=0, prop=0, acc=0, stored=0, unique=0, wall=0.0, certs=0))
        g['jobs'] += 1; g['complete'] += bool(j.get('complete')); g['prop'] += j['n_prop']; g['acc'] += j['n_acc']; g['stored'] += len(j['series']); g['unique'] += len(j['points'])
        g['wall'] += j['wall']; g['certs'] += j.get('n_cert', 0); cl = classify(j); g.setdefault('classes', {}); g['classes'][cl] = g['classes'].get(cl, 0) + 1
        g.setdefault('job_wall', []).append(j['wall']); g.setdefault('s_per_prop', []).append(j['wall']/max(j['n_prop'], 1))
        g['never_in_target'] = g.get('never_in_target', 0) + (j['mode'] == 'sample' and j.get('n_post', 0) == 0)   # never finished the climb/burn-in inside the target
        g['no_records'] = g.get('no_records', 0) + (len(j['points']) == 0 and j['mode'] == 'sample'); g['no_best'] = g.get('no_best', 0) + (j['mode'] != 'sample' and not j.get('best'))
        g['hcoll_errors'] = g.get('hcoll_errors', 0) + sum(v for k_, v in j.get('rejects', {}).items() if k_.startswith('hcoll_error'))
    for g in res['groups'].values():                      # time accounting: median and 95th percentile (slow nodes decide the wall budget)
        w_ = np.array(g.pop('job_wall')); sp = np.array(g.pop('s_per_prop'))
        g['job_wall_median'] = float(np.median(w_)); g['job_wall_p95'] = float(np.percentile(w_, 95)); g['s_per_proposal_median'] = float(np.median(sp)); g['s_per_proposal_p95'] = float(np.percentile(sp, 95))
    if os.environ.get('PLOTDIR'): make_plots(J, os.environ['PLOTDIR']); make_plots(J, os.environ['PLOTDIR'], only_pass=True)
    # sampler diagnostics per (logL, start)
    bas = collections.defaultdict(list)
    for j in J:
        if j['mode'] == 'sample' and j['series']: bas[(j['logL'], j['start'])].append(j)
    rng = np.random.default_rng(0)
    for (L_, st), chs in sorted(bas.items()):
        q = {'log_tmg': lambda p: np.log10(p['obs']['tau_mu_g']), 'log_m1': lambda p: np.log10(p['masses']['m1']), 'cert': lambda p: p['cert']['min'],
             'log_maxY2': lambda p: np.log10(max(np.abs(p['x'][2:11]).max(), 1e-12)), 'tanb': lambda p: float(np.exp(p['x'][11])), 'cba': lambda p: p['masses']['cba'],
             'lam4': lambda p: p['x'][18], 'lam6': lambda p: p['x'][17], 'log_meg': lambda p: np.log10(max(p['obs']['mu_e_g'], 1e-30))}
        cls = collections.Counter(classify(c) for c in chs)
        d = dict(n_chains=len(chs), n_points=sum(len(c['series']) for c in chs), n_stuck=sum(len({p['id'] for p in c['series']}) < 2 for c in chs), n_incomplete=sum(not c.get('complete') for c in chs), classes=dict(cls),
                 n_enter=[c.get('n_enter') for c in chs])
        for nm, f in q.items():
            ser = [[f(p) for p in c['series']] for c in chs]; zs = rank_norm(ser)         # all chains enter (none is discarded); the rank-normalised values give bulk R-hat/ESS
            d['rhat_' + nm] = rhat(zs); d['ess_' + nm] = float(sum(len(s_)/act(s_) for s_ in zs))
        d['converged'] = bool(len(chs) >= 4 and cls.get('COMPLETE_TARGET_SAMPLED', 0) == len(chs) and d['n_stuck'] == 0 and d['n_incomplete'] == 0
                              and all(d['rhat_' + n_] < 1.1 for n_ in q) and min(d['ess_' + n_] for n_ in q) >= 100)
        d['frac_final_pass'] = float(np.mean([np.mean([p.get('final_status') == 'PASS' for p in c['series']]) for c in chs]))   # stored states that pass vacuum+collider+requested cutoff
        for X in (10.0, 11.0, 12.0):                      # DIAGNOSTIC fraction of visited stored states with certified scale >= X (5-variant minimum), full gate not required
            per = [np.mean([p['cert']['min'] >= X for p in c['series']]) for c in chs]
            bs = [np.mean(rng.choice(per, len(per))) for _ in range(300)]
            d['frac_cert_ge_%g' % X] = dict(mean=float(np.mean(per)), lo=float(np.percentile(bs, 16)), hi=float(np.percentile(bs, 84)))
        res['basins']['logL=%g|start=%d' % (L_, st)] = d
    # verification
    import accept as AC, cert as CE
    def verify(group, p, logL, prov=None):
        """full gate + fine-grid certification.  meets_cutoff: logL > 0: vacuum/collider/low-energy PASS and a POSITIVE valid certified reach >= logL
        (a -1 start-failure sentinel can never satisfy a cutoff); logL = -1 (nominal baseline): low-energy PASS only; logL = 0 (rescaled reference):
        PASS and a positive valid reach.  The fields separate nominal acceptance from the rescaled-variant result."""
        x, eta = np.array(p['x']), np.array(p['eta'])
        r = AC.accept(x, eta, cmin, logL=None, collider=True, n_vac=200); cs, cd = CE.cert(x, eta, CE.FULL)
        nominal = (r['status'] == 'PASS'); reach_valid = bool(np.isfinite(cs) and cs > 0); start_fail = any(str(v['fail']).startswith('start') for v in cd.values())
        num_stop = sorted({v['fail'] for v in cd.values() if v['fail'] in ('integration', 'nonfinite')})
        ok = nominal and (True if logL < -0.5 else (reach_valid and not start_fail and (logL < 0.5 or cs >= logL)))
        return dict(group=group, logL=logL, status=r['status'], failed=r.get('failed_cuts'), certified13=(float(cs) if reach_valid else None), nominal_pass=bool(nominal), reach_valid=reach_valid, start_fail=bool(start_fail),
                    numerical_stops=num_stop, meets_cutoff=bool(ok), chi2=r['chi2'], provenance=prov,
                    obs={**{k: r['obs'][k] for k in ('mu_e_g', 'tau_mu_g', 'tau_e_g')}, 'h_tau_mu': r.get('collider', {}).get('BR_h_tau_mu', r['obs']['h_tau_mu'])}, x=list(map(float, x)), eta=list(map(float, eta)))
    seen = set(); key_of = lambda p: tuple(np.round(np.asarray(p['x'], float), 12)) + tuple(np.round(np.asarray(p['eta'], float), 12))
    cands = collections.defaultdict(list)
    for j in J:                                           # best AND the archive of the last accepted improvements: a candidate whose final revalidation fails is replaced by its predecessors
        grp = '%s|%g' % (j['mode'], j['logL'])
        prov = dict(run=j['manifest'].get('run_id'), seed=j['seed'], start=j['start'], margin=j.get('margin'), logL=j['logL'])
        if j['best'] or j.get('best_archive'):
            for b in ([j['best']] if j['best'] else []) + list(j.get('best_archive') or []):
                if key_of(b) not in seen: seen.add(key_of(b)); cands[grp].append((b, j['logL'], prov))     # deduplicated in full parameter coordinates
        if j.get('best_unresolved'): res.setdefault('unresolved_bests', []).append(dict(group=grp, obs=j['best_unresolved']['obs']))
    for k, L_ in cands.items():
        key = k.split('|')[0].split(':')[1]
        for b, lg, pv in sorted(L_, key=lambda t: -t[0]['obs'][key])[:TOPK*3]: res['verified'].append(verify(k, b, lg, pv))
    pool = [(p, j['logL']) for j in J if j['mode'] == 'sample' for p in j['series']]; pool = list({key_of(p): (p, l_) for p, l_ in pool}.values())       # distinct states (dwell weights stay in the chains)
    for i in (rng.choice(len(pool), size=min(NS, len(pool)), replace=False) if pool else []): res['verified'].append(verify('sample', *pool[i], prov=dict(kind='visited-state sample')))
    res['n_verified'] = len(res['verified']); res['n_pass'] = sum(v['meets_cutoff'] for v in res['verified'])
    good = [v for v in res['verified'] if v['meets_cutoff'] and v['group'] != 'sample']
    fin = sorted([v for v in good if v['certified13'] is not None], key=lambda v: -v['certified13'])[:max(NF//2, 1)]      # highest demonstrated scales ...
    for obs_ in ('mu_e_g', 'tau_mu_g', 'tau_e_g', 'h_tau_mu'):                      # ... the best verified witness of each observable, and of each observable at EACH requested cutoff (Pareto witnesses)
        for lg in sorted({v['logL'] for v in good if v['logL'] > 0}) + [None]:
            w = sorted([v for v in good if v['obs'][obs_] is not None and (lg is None or v['logL'] == lg)], key=lambda v: -v['obs'][obs_])[:1]
            fin += [v for v in w if v not in fin]
    from campaign import clean
    json.dump(clean(res), open(os.path.join(ROOT, 'results', 'campaign_summary.json'), 'w'), indent=1, allow_nan=False)
    json.dump({'C%d' % i: dict(x=v['x'], eta=v['eta'], origin='campaign ' + v['group'], provenance=v.get('provenance')) for i, v in enumerate(fin)}, open(os.path.join(ROOT, 'results', 'campaign_finalists.json'), 'w'), indent=1)
    if os.environ.get('PLOTDIR'): make_verified_plot(res['verified'], os.environ['PLOTDIR'])
    print(json.dumps(res['groups'], indent=0)[:1500]); print('verified', res['n_verified'], 'meet cutoff', res['n_pass'], 'finalists', len(fin))


if __name__ == '__main__': main()
