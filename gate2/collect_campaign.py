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


def reach_class(cs, variants):
    """classify a stored/verified reach: 'boundary_failure' (some variant violates a condition at the matching scale: no scale), 'numerical_stop' (solver stop or
    non-finite coupling: a demonstrated PREFIX, not a first-failure scale), 'ceiling' (no failure up to 1e16 GeV), 'physical_failure' (first crossing of BFB/unitarity/perturbativity
    found by bisection).  Only the last two are plotted on a scale axis; the sentinel -1 is never a scale."""
    fl = [str(v['fail']) if v['fail'] is not None else None for v in variants.values()]
    if any(f is not None and f.startswith('start') for f in fl): return 'boundary_failure'
    if any(f in ('integration', 'nonfinite') for f in fl): return 'numerical_stop'
    if all(f is None for f in fl): return 'ceiling'
    return 'physical_failure'


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
    allp = [(j['logL'], p) for j in J if j['mode'] == 'sample' for p in j['series'] if p.get('cert') and (not only_pass or p.get('final_status') == 'PASS')]
    cls = collections.Counter(reach_class(p['cert']['min'], p['cert']['variants']) for _, p in allp)
    pts = [(l_, p) for l_, p in allp if reach_class(p['cert']['min'], p['cert']['variants']) in ('physical_failure', 'ceiling')]     # boundary failures and numerical stops are NOT scales
    if not pts: return
    L = np.array([a for a, _ in pts]); cert = np.array([p['cert']['min'] for _, p in pts]); tmg = np.log10([p['obs']['tau_mu_g'] for _, p in pts])
    meg = np.log10([max(p['obs']['mu_e_g'], 1e-30) for _, p in pts]); m1 = np.array([p['masses']['m1'] for _, p in pts]); cba = np.array([abs(p['masses']['cba']) for _, p in pts])
    cos2 = np.cos([p['masses']['phi'] for _, p in pts])**2; what = 'gate-passing stored states' if only_pass else 'all stored states (diagnostic)'
    xl = r'screening reach $\log_{10}\Lambda$ (coarse 5-variant minimum)'
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
    ax[0].scatter(cert, tmg, s=5, c=L, cmap='viridis'); ax[0].axhline(-9, c='gray', ls='--', lw=0.8); ax[0].set_xlabel(xl); ax[0].set_ylabel(r'$\log_{10}\mathrm{Br}(\tau\to\mu\gamma)$')
    ax[1].scatter(cert, meg, s=5, c=L, cmap='viridis'); ax[1].axhline(np.log10(6e-14), c='gray', ls='--', lw=0.8); ax[1].set_xlabel(xl); ax[1].set_ylabel(r'$\log_{10}\mathrm{Br}(\mu\to e\gamma)$')
    sc = ax[2].scatter(cba, cert, s=5, c=L, cmap='viridis'); ax[2].set_xlabel(r'$|c_{\beta-\alpha}|$'); ax[2].set_ylabel(xl); fig.colorbar(sc, ax=ax[2], label='logL of the chain (-1, 0: baselines)')
    fig.suptitle(what + '; not on the axis: ' + ', '.join('%s %d' % (k, v) for k, v in sorted(cls.items()) if k in ('boundary_failure', 'numerical_stop')), fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(outdir, pre + 'scan_lfv_cutoff.pdf')); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for lo, hi, c, lab in CATS:
        m = (L >= lo) & (L <= hi)
        if m.any(): ax[0].scatter(m1[m], cos2[m], s=4, c=c, label=lab.split(':')[0] if lab.startswith('logL') else 'logL>=6', alpha=0.6); ax[1].hist(cert[m], bins=30, color=c, alpha=0.5)
    ax[0].set_xlabel(r'$m_{h_1^+}$ [GeV]'); ax[0].set_ylabel(r'doublet fraction $\cos^2\varphi$ of $h_1^+$'); ax[0].legend(fontsize=6, frameon=False); ax[1].set_xlabel(xl)
    fig.suptitle(what + ': local exploration from the declared starts (reach classes: ' + ', '.join('%s %d' % kv for kv in sorted(cls.items())) + ')', fontsize=6); fig.tight_layout(); fig.savefig(os.path.join(outdir, pre + 'scan_baseline_vs_filtered.pdf')); plt.close(fig)
    if not only_pass:
        fails = collections.Counter(v['fail'] or 'none' for _, p in pts for v in p['cert']['variants'].values())
        fig, ax = plt.subplots(figsize=(4.2, 2.8)); ax.bar(list(fails), list(fails.values())); ax.set_ylabel('variant evaluations'); ax.set_title('first-failure mechanism (screening variants)', fontsize=9); plt.xticks(rotation=30, fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, 'scan_failure_mechanisms.pdf')); plt.close(fig)


def make_verified_plot(verified, outdir):
    from zee import hgmodel as H
    """Re-verified points only (accept(collider=True, n_vac=200) and the 13-variant certification on the fine grid): the quantity quoted in results"""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    good = [v for v in verified if v['meets_cutoff'] and v['screen13'] is not None and v['screen13'] > 0 and v['reach_class'] in ('physical_failure', 'ceiling')]
    if not good: return
    os.makedirs(outdir, exist_ok=True); fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for a, (k, lim) in zip(ax, (('mu_e_g', 1.5e-13), ('tau_mu_g', 4.2e-8), ('h_tau_mu', H.H_TAU_MU_LIMIT))):
        a.scatter([v['screen13'] for v in good], np.log10([max(v['obs'][k], 1e-30) for v in good]), s=8, c=[v['logL'] for v in good], cmap='viridis'); a.axhline(np.log10(lim), c='gray', ls='--', lw=0.8)
        a.set_xlabel(r'screening reach $\log_{10}\Lambda$ (10/decade, 13 variants)'); a.set_ylabel(r'$\log_{10}$ ' + k)
    fig.suptitle('re-verified points: full low-energy gate, 13-variant SCREENING reach (final scales come from benchmarks.py)', fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(outdir, 'verified_lfv_cutoff.pdf')); plt.close(fig)


def main():
    # all outputs go to ZEE_OUT (default results/ of the checkout); a frozen read-only checkout needs ZEE_OUT (and ZEE_CACHE for benchmarks.py) set to a writable directory
    global OUT
    OUT = os.environ.get('ZEE_OUT', os.path.join(ROOT, 'results'))
    try: os.makedirs(OUT, exist_ok=True); _t = os.path.join(OUT, '.write_test'); open(_t, 'w').close(); os.remove(_t)
    except OSError as e_: sys.exit('output directory %s is not writable (%s): set ZEE_OUT to a writable directory (the frozen checkout is read-only)' % (OUT, e_))
    if os.environ.get('PLOTDIR'):
        try: os.makedirs(os.environ['PLOTDIR'], exist_ok=True)
        except OSError as e_: sys.exit('PLOTDIR not writable: %s' % e_)
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
    exp = json.load(open(os.environ['EXPECT_FILE'])) if os.environ.get('EXPECT_FILE') else None
    if exp is not None:                                   # exact expected IDs (mode, logL, seed): missing outputs need accounting from the Condor logs, they are not evidence of 'no physical points'
        want = {(e_['mode'], float(e_['logL']), int(e_['seed'])) for e_ in exp}; have = {(j['mode'], float(j['logL']), int(j['seed'])) for j in J}
        res['missing_jobs'] = sorted(map(list, want - have)); res['unexpected_jobs'] = sorted(map(list, have - want))
        if (want - have or have - want) and not os.environ.get('ALLOW_MISSING'): sys.exit('job inventory mismatch: %d missing, %d unexpected (ALLOW_MISSING=1 overrides, e.g. to inspect a partial run); missing e.g. %s' % (len(want - have), len(have - want), sorted(want - have)[:3]))
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
        if j['mode'] == 'sample': bas[(j['logL'], j['start'])].append(j)          # ALL sample chains enter the inventory, including empty ones
    rng = np.random.default_rng(0)
    for (L_, st), chs in sorted(bas.items()):
        q = {'log_tmg': lambda p: np.log10(p['obs']['tau_mu_g']), 'log_m1': lambda p: np.log10(p['masses']['m1']), 'cert_pos': lambda p: max(p['cert']['min'], 0.0),
             'log_maxY2': lambda p: np.log10(max(np.abs(p['x'][2:11]).max(), 1e-12)), 'tanb': lambda p: float(np.exp(p['x'][11])), 'cba': lambda p: p['masses']['cba'],
             'lam4': lambda p: p['x'][18], 'lam6': lambda p: p['x'][17], 'log_meg': lambda p: np.log10(max(p['obs']['mu_e_g'], 1e-30))}
        n_exp = sum(1 for e_ in (exp or []) if e_['mode'] == 'sample' and float(e_['logL']) == L_ and e_['start'] == st) or len(chs)
        chs_all = chs; chs = [c for c in chs_all if len(c['series']) >= 8]              # diagnostics (R-hat/ESS) need >= 8 stored states; the others stay in the inventory and block the converged label
        cls = collections.Counter(classify(c) for c in chs_all)
        if not chs:
            res['basins']['logL=%g|start=%d' % (L_, st)] = dict(n_expected=n_exp, n_found=len(chs_all), n_missing=n_exp - len(chs_all), n_chains=0, classes=dict(cls), converged=False, note='no chain with >= 8 stored states')
            continue
        d = dict(n_expected=n_exp, n_found=len(chs_all), n_missing=n_exp - len(chs_all), n_empty_or_short=len(chs_all) - len(chs), n_chains=len(chs), n_points=sum(len(c['series']) for c in chs), n_stuck=sum(len({p['id'] for p in c['series']}) < 2 for c in chs), n_incomplete=sum(not c.get('complete') for c in chs), classes=dict(cls),
                 n_enter=[c.get('n_enter') for c in chs])
        for nm, f in q.items():
            ser = [[f(p) for p in c['series']] for c in chs]; zs = rank_norm(ser)         # all chains enter (none is discarded); the rank-normalised values give bulk R-hat/ESS
            d['rhat_' + nm] = rhat(zs); d['ess_' + nm] = float(sum(len(s_)/act(s_) for s_ in zs))
        d['converged'] = bool(len(chs) >= 4 and n_exp == len(chs_all) == len(chs) and cls.get('COMPLETE_TARGET_SAMPLED', 0) == n_exp and d['n_stuck'] == 0 and d['n_incomplete'] == 0
                              and all(d['rhat_' + n_] < 1.1 for n_ in q) and min(d['ess_' + n_] for n_ in q) >= 100)
        d['frac_final_pass'] = float(np.mean([np.mean([p.get('final_status') == 'PASS' for p in c['series']]) for c in chs]))   # stored states that pass vacuum+collider+requested cutoff
        for X in (10.0, 11.0, 12.0):                      # DIAGNOSTIC fraction of visited stored states with certified scale >= X (5-variant minimum), full gate not required
            per = [np.mean([p['cert']['min'] >= X for p in c['series']]) for c in chs]
            bs = [np.mean(rng.choice(per, len(per))) for _ in range(300)]
            d['frac_cert_ge_%g' % X] = dict(mean=float(np.mean(per)), lo=float(np.percentile(bs, 16)), hi=float(np.percentile(bs, 84)))
        res['basins']['logL=%g|start=%d' % (L_, st)] = d
    # verification
    import accept as AC, cert as CE
    key_of = lambda p: tuple(np.round(np.asarray(p['x'], float), 12)) + tuple(np.round(np.asarray(p['eta'], float), 12))
    cache = {}                                            # one verification per distinct input (exact full parameter vector); group/cutoff memberships are kept separately

    def raw_verify(p):
        k = key_of(p)
        if k not in cache:
            x, eta = np.array(p['x']), np.array(p['eta'])
            r = AC.accept(x, eta, cmin, logL=None, collider=True, n_vac=200); cs, cd = CE.cert(x, eta, CE.FULL); rc = reach_class(cs, cd)
            cache[k] = dict(status=r['status'], failed=r.get('failed_cuts'), nominal_pass=(r['status'] == 'PASS'), reach_class=rc, chi2=r['chi2'],
                            reach_valid=bool(np.isfinite(cs) and cs > 0 and rc in ('physical_failure', 'ceiling')), cs=float(cs), x=list(map(float, x)), eta=list(map(float, eta)),
                            numerical_stops=sorted({v['fail'] for v in cd.values() if v['fail'] in ('integration', 'nonfinite')}),
                            obs={**{kk: r['obs'][kk] for kk in ('mu_e_g', 'tau_mu_g', 'tau_e_g')}, 'h_tau_mu': r.get('collider', {}).get('BR_h_tau_mu', r['obs']['h_tau_mu'])})
        return cache[k]

    def member(raw, group, logL, prov=None):
        """eligibility of a verified point for the cutoff of THIS membership: logL > 0: nominal PASS and a valid (physical/ceiling) screening reach >= logL; logL = -1 (nominal baseline):
        nominal PASS only; logL = 0: PASS and a valid reach.  Numerical stops and boundary failures never count as a demonstrated scale (the stop prefix is reported separately)."""
        ok = raw['nominal_pass'] and (True if logL < -0.5 else (raw['reach_valid'] and (logL < 0.5 or raw['cs'] >= logL)))
        return dict(group=group, logL=logL, status=raw['status'], failed=raw['failed'], screen13=(raw['cs'] if raw['reach_valid'] else None), prefix_screen13=(raw['cs'] if raw['reach_class'] == 'numerical_stop' and raw['cs'] > 0 else None),
                    reach_class=raw['reach_class'], nominal_pass=raw['nominal_pass'], reach_valid=raw['reach_valid'], start_fail=(raw['reach_class'] == 'boundary_failure'), numerical_stops=raw['numerical_stops'],
                    meets_cutoff=bool(ok), chi2=raw['chi2'], provenance=prov, obs=raw['obs'], x=raw['x'], eta=raw['eta'])
    members = collections.defaultdict(list)               # group -> [(point, logL, provenance)]; the same input may belong to several groups
    for j in J:                                           # best AND the archive of the last accepted improvements: a candidate whose final revalidation fails is replaced by its predecessors
        grp = '%s|%g' % (j['mode'], j['logL'])
        prov = dict(run=j['manifest'].get('run_id'), seed=j['seed'], start=j['start'], margin=j.get('margin'), logL=j['logL'], mode=j['mode'])
        if j['best'] or j.get('best_archive'):
            seen_g = set()
            for b in ([j['best']] if j['best'] else []) + list(j.get('best_archive') or []):
                if key_of(b) not in seen_g: seen_g.add(key_of(b)); members[grp].append((b, j['logL'], prov))       # deduplicate within a job only
        if j.get('best_unresolved'): res.setdefault('unresolved_bests', []).append(dict(group=grp, obs=j['best_unresolved']['obs']))
    for k, L_ in members.items():
        key = k.split('|')[0].split(':')[1]
        for b, lg, pv in sorted(L_, key=lambda t: -t[0]['obs'][key])[:TOPK*3]: res['verified'].append(member(raw_verify(b), k, lg, pv))
    pool = {}                                             # distinct visited states of the sample chains: all memberships (logL of every chain that visited them) are retained
    for j in J:
        if j['mode'] != 'sample': continue
        for p in j['series']: pool.setdefault(key_of(p), (p, set()))[1].add(j['logL'])
    states = list(pool.values())
    chosen = {}                                           # purposeful extremes: best rate per observable among gate-passing stored states, highest screening reach, plus random representatives
    for obs_ in ('mu_e_g', 'tau_mu_g', 'tau_e_g', 'h_tau_mu'):
        for p, ls in sorted([t for t in states if t[0].get('final_status') == 'PASS'], key=lambda t: -t[0]['obs'][obs_])[:5]: chosen[key_of(p)] = (p, ls, 'sample-extreme:' + obs_)
    for p, ls in sorted([t for t in states if t[0].get('cert')], key=lambda t: -t[0]['cert']['min'])[:5]: chosen.setdefault(key_of(p), (p, ls, 'sample-extreme:reach'))
    for i in (rng.choice(len(states), size=min(NS, len(states)), replace=False) if states else []): chosen.setdefault(key_of(states[i][0]), (states[i][0], states[i][1], 'sample-random'))
    for p, ls, why in chosen.values():
        for lg in sorted(ls): res['verified'].append(member(raw_verify(p), 'sample', lg, dict(kind=why)))
    res['n_verified_distinct'] = len(cache); res['n_verified'] = len(res['verified']); res['n_pass'] = sum(v['meets_cutoff'] for v in res['verified'])
    # finalists: eligibility is judged on the verified scale of each DISTINCT point (valid for every cutoff it reaches), not on the cutoff of the job that produced it
    good = {}
    for k_, raw in cache.items():
        if raw['nominal_pass'] and raw['reach_valid']: good[k_] = raw
    fin_raw = sorted(good.values(), key=lambda v: -v['cs'])[:max(NF//2, 1)]      # highest demonstrated (screening) scales ...
    fin = [dict(x=v['x'], eta=v['eta'], obs=v['obs'], logL=None, group='top-reach', provenance=None) for v in fin_raw]
    pm = collections.defaultdict(list)
    for v in res['verified']: pm[tuple(np.round(v['x'], 12))].append(v)
    for obs_ in ('mu_e_g', 'tau_mu_g', 'tau_e_g', 'h_tau_mu'):     # ... the best verified witness of each observable, and of each observable at EACH requested cutoff (Pareto witnesses)
        for lg in sorted({v['logL'] for v in res['verified'] if v['logL'] > 0}) + [None]:
            w = sorted([r for r in good.values() if r['obs'][obs_] is not None and (lg is None or r['cs'] >= lg)], key=lambda r: -r['obs'][obs_])[:1]
            fin += [dict(x=r['x'], eta=r['eta'], obs=r['obs'], logL=lg, group='best-%s@%s' % (obs_, lg), provenance=[m['provenance'] for m in pm.get(tuple(np.round(r['x'], 12)), [])][:4]) for r in w]
    seen_f = set(); fin = [f for f in fin if not (tuple(np.round(f['x'], 12)) in seen_f or seen_f.add(tuple(np.round(f['x'], 12))))]
    from campaign import clean
    json.dump(clean(res), open(os.path.join(OUT, 'campaign_summary.json'), 'w'), indent=1, allow_nan=False)
    json.dump({'C%d' % i: dict(x=v['x'], eta=v['eta'], origin='campaign ' + v['group'], provenance=v.get('provenance')) for i, v in enumerate(fin)}, open(os.path.join(OUT, 'campaign_finalists.json'), 'w'), indent=1)
    if os.environ.get('PLOTDIR'): make_verified_plot(res['verified'], os.environ['PLOTDIR'])
    print(json.dumps(res['groups'], indent=0)[:1500]); print('verified', res['n_verified'], 'meet cutoff', res['n_pass'], 'finalists', len(fin))


OUT = None
if __name__ == '__main__': main()
