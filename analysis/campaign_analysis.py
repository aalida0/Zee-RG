"""Analysis of a collected campaign (referee plan, sections 3 and 5): physical correlations with controls, per start, never pooled across starts.

Input: the collector output `campaign_summary.json` (verified, distinct full-vector points with group, logL, provenance incl. start/seed) and optionally `results/validation/validate_*.json`.
Output (directory OUT): points.csv (derived quantities of every distinct verified PASS point), report.md, fig_rates_vs_reach.pdf, fig_spectrum.pdf, fig_lambda6.pdf, fig_baseline_vs_highscale.pdf,
fig_matching.pdf (if validation records are given), and analysis.json.

Rules built into the analysis (the campaign is a set of local explorations around a few starts, not a scan of the region):
 * statistics are computed per start and are never pooled across starts; a start with fewer than NMIN distinct points or NSEED seeds is reported as 'insufficient';
 * point densities are not probabilities (the chains are Metropolis walks / maximisers with different measures); correlations are rank correlations of DISTINCT points, with a seed-blocked bootstrap
   interval, and partial correlations control for the quantities the tree-level formulas say matter (heavy mass, Yukawa size, alignment);
 * the `reach` plotted is the verified 13-variant screening scale of the collector (not a certified bound); points with undefined reach are excluded from reach statistics, not set to a number;
 * a correlation that does not survive the controls is reported as such.
Usage: ZEE_INPUTS=modern python3 -W ignore analysis/campaign_analysis.py SUMMARY.json OUT_DIR [--validation DIR]"""
import sys, os, json, argparse, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
os.environ.setdefault('ZEE_INPUTS', 'modern')
NMIN, NSEED, NBOOT = 30, 3, 400
CHANNELS = (('mu_e_g', r'BR($\mu\to e\gamma$)'), ('tau_mu_g', r'BR($\tau\to\mu\gamma$)'), ('h_tau_mu', r'BR($h\to\tau\mu$)'))


def rank(a):
    a = np.asarray(a, float); o = np.argsort(a, kind='mergesort'); r = np.empty(len(a)); r[o] = np.arange(len(a))
    for v in np.unique(a):                                                            # average ranks for ties
        m = a == v
        if m.sum() > 1: r[m] = r[m].mean()
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0: return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def partial_spearman(a, b, controls):
    """rank partial correlation of a and b given the columns of `controls` (linear residualisation of the ranks)"""
    ra, rb = rank(a), rank(b); C = np.column_stack([np.ones(len(ra))] + [rank(c) for c in controls])
    ea = ra - C @ np.linalg.lstsq(C, ra, rcond=None)[0]; eb = rb - C @ np.linalg.lstsq(C, rb, rcond=None)[0]
    if ea.std() < 1e-12 or eb.std() < 1e-12: return float('nan')
    return float(np.corrcoef(ea, eb)[0, 1])


def block_bootstrap(fun, seeds, rng, n=NBOOT):
    """bootstrap over SEEDS (chains of one seed are correlated): resample whole seed blocks; fun(index_array) -> statistic"""
    seeds = np.asarray(seeds); ids = np.unique(seeds); out = []
    for _ in range(n):
        pick = rng.choice(ids, size=len(ids), replace=True); idx = np.concatenate([np.where(seeds == s)[0] for s in pick])
        v = fun(idx)
        if np.isfinite(v): out.append(v)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if len(out) > 20 else (float('nan'), float('nan'))


def derive(rec):
    """derived physical quantities of a verified record"""
    import bench
    from zee import hgmodel as H
    x, eta = np.array(rec['x'], float), np.array(rec['eta'], float)
    sc, c = bench.coup(x, eta); e = H.evaluate(list(x), 'NO', eta=list(eta)); nm = dict(zip(H.NAMES, x))
    y2 = max(abs(nm['Y_et']), abs(nm['Y_tt_re'] + 1j*nm['Y_tt_im']), abs(nm['Y_tm_re'] + 1j*nm['Y_tm_im']), abs(nm['Y_te_re'] + 1j*nm['Y_te_im']), abs(nm['Y_mt_re'] + 1j*nm['Y_mt_im']))
    vb = float(bench.margins(c)[0]); lh = float(c[10])
    d = dict(mH=sc['mH'], mA=sc['mA'], m1=sc['m1'], m2=sc['m2'], tb=-sc['tb'] if sc['tb'] < 0 else sc['tb'], cba=sc['cba'], phi=sc['phi'], mu_v=sc['mu']/246.22,
             split_HA=sc['mH'] - sc['mA'], split_Hp1=sc['m1'] - sc['mH'], split_Hp2=sc['m2'] - sc['mH'], mH_A_mean=0.5*(sc['mH'] + sc['mA']),
             lam6=c[5], lam4=c[3], lam5=c[4], lam1=c[0], lam2=c[1], lam3=c[2], max_quartic=float(np.abs(c).max()), vred=vb, lam_h=lh, margin=min(vb, lh), maxY2=float(y2), chi2=rec['chi2'])
    for k in ('mu_e_g', 'tau_mu_g', 'tau_e_g', 'h_tau_mu'): d[k] = rec['obs'][k]
    d['reach'] = rec['screen13'] if 'screen13' in rec else rec.get('certified13'); d['reach_class'] = rec.get('reach_class'); d['group'] = rec['group']; d['logL'] = rec['logL']
    pv = rec.get('provenance') or {}
    st = pv.get('start', None) if pv.get('start', None) is not None else (pv.get('starts') or [None])[0]
    sd = pv.get('seed', None) if pv.get('seed', None) is not None else (pv.get('seeds') or [None])[0]
    d['start'] = st; d['seed'] = sd; d['_x'] = x; d['_eta'] = eta
    return d


def load_points(summary_path):
    S = json.load(open(summary_path)); seen = {}; rows = []
    for r in S['verified']:
        if r.get('status') != 'PASS': continue
        k = tuple(r['x']) + tuple(r['eta']) + (r['group'], r['logL'])
        if k in seen: continue
        seen[k] = 1
        try: rows.append(derive(r))
        except Exception as ex: rows.append(dict(error=repr(ex), group=r['group'], logL=r['logL']))
    return [r for r in rows if 'error' not in r], [r for r in rows if 'error' in r], S


def per_start(rows):
    """group distinct points by (start); nothing is pooled across starts"""
    out = {}
    for r in rows:
        if r['start'] is None: continue
        out.setdefault(r['start'], []).append(r)
    return out


def lambda6_study(pts, rng):
    """does lambda_6 (through c_{b-a}) explain BR(h->tau mu), and does the reach depend on it, after controlling for heavy mass, Yukawa size and alignment?  rank correlations per start"""
    seeds = np.array([p['seed'] if p['seed'] is not None else -1 for p in pts]); res = {}
    la = np.array([p['lam6'] for p in pts]); br = np.log10([max(p['h_tau_mu'], 1e-300) for p in pts])
    ctrl = [np.log([p['mH_A_mean'] for p in pts]), np.log([max(p['maxY2'], 1e-12) for p in pts])]
    res['n'] = len(pts); res['n_seeds'] = int(len(np.unique(seeds)))
    if res['n'] < NMIN or res['n_seeds'] < NSEED: res['status'] = 'insufficient (n<%d or seeds<%d)' % (NMIN, NSEED); return res
    f1 = lambda i: spearman(la[i], br[i]); f2 = lambda i: partial_spearman(la[i], br[i], [c[i] for c in ctrl])
    allv = np.arange(len(pts)); res['status'] = 'ok'
    res['spearman_lam6_logBRhtm'] = dict(value=f1(allv), ci=block_bootstrap(f1, seeds, rng)); res['partial_given_mH_Y2'] = dict(value=f2(allv), ci=block_bootstrap(f2, seeds, rng))
    ok = [i for i, p in enumerate(pts) if p['reach'] is not None]
    if len(ok) >= NMIN:
        reach = np.array([pts[i]['reach'] for i in ok]); l6 = np.abs(la[ok]); s_ = seeds[ok]
        g = lambda i: spearman(l6[i], reach[i]); res['spearman_abs_lam6_reach'] = dict(value=g(np.arange(len(ok))), ci=block_bootstrap(g, s_, rng), n=len(ok))
    return res


def make_figures(rows, out, val_dir=None):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ok = [r for r in rows if r['reach'] is not None]; lo = sorted({r['logL'] for r in rows})
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a, (k, lab) in zip(ax, CHANNELS):
        for L in lo:
            q = [r for r in ok if r['logL'] == L]
            if q: a.scatter([r['reach'] for r in q], [r[k] for r in q], s=10, label='requirement %g' % L, alpha=.7)
        a.set_yscale('log'); a.set_xlabel(r'verified screening scale $\log_{10}(\Lambda/{\rm GeV})$'); a.set_ylabel(lab)
    ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(out, 'fig_rates_vs_reach.pdf')); plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a, (xk, lab) in zip(ax, (('split_HA', r'$m_H-m_A$ [GeV]'), ('split_Hp1', r'$m_{h_1^+}-m_H$ [GeV]'), ('split_Hp2', r'$m_{h_2^+}-m_H$ [GeV]'))):
        sc_ = a.scatter([r[xk] for r in ok], [r['reach'] for r in ok], c=[np.log10(max(r['mu_e_g'], 1e-30)) for r in ok], s=10); a.set_xlabel(lab); a.set_ylabel('verified screening scale')
    fig.colorbar(sc_, ax=ax[-1], label=r'$\log_{10}$BR($\mu\to e\gamma$)'); fig.tight_layout(); fig.savefig(os.path.join(out, 'fig_spectrum.pdf')); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    if ok:
        s1 = ax[0].scatter([r['lam6'] for r in ok], [r['h_tau_mu'] for r in ok], c=[r['reach'] for r in ok], s=10); ax[0].set_yscale('log'); ax[0].set_xlabel(r'$\lambda_6$'); ax[0].set_ylabel(r'BR($h\to\tau\mu$)'); fig.colorbar(s1, ax=ax[0], label='verified scale')
        ax[1].scatter([abs(r['cba']) for r in ok], [r['h_tau_mu'] for r in ok], s=10); ax[1].set_yscale('log'); ax[1].set_xscale('log'); ax[1].set_xlabel(r'$|c_{\beta-\alpha}|$')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'fig_lambda6.pdf')); plt.close(fig)
    base = [r for r in rows if r['logL'] < -0.5]; high = [r for r in rows if r['logL'] >= 8]
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))
    for a, (k, lab) in zip(ax, (('max_quartic', r'$\max|\lambda_i|$'), ('margin', 'conservative BFB margin'), ('mH_A_mean', r'$(m_H+m_A)/2$ [GeV]'), ('maxY2', r'$\max|Y_2|$'))):
        for name, q in (('low-energy baseline (req. -1)', base), ('high-scale requirement >= 8', high)):
            if q: a.hist([r[k] for r in q], bins=15, histtype='step', label=name, density=True)
        a.set_xlabel(lab)
    ax[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(os.path.join(out, 'fig_baseline_vs_highscale.pdf')); plt.close(fig)
    if val_dir and os.path.isdir(val_dir):
        fig, ax = plt.subplots(figsize=(5, 4)); n = 0
        for f in sorted(os.listdir(val_dir)):
            if not f.startswith('validate_'): continue
            v = json.load(open(os.path.join(val_dir, f)))
            for mode, m in v['matching'].items():
                if m['min_stress'] is not None: ax.scatter(v['tree']['min'], m['min_stress'], s=18, marker='o' if m['complete'] else 'x'); n += 1
        if n: ax.plot([10, 17], [10, 17], 'k:', lw=.6); ax.set_xlabel('tree-level boundary scale'); ax.set_ylabel('matched scale (x: undefined variants)'); fig.tight_layout(); fig.savefig(os.path.join(out, 'fig_matching.pdf'))
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('summary'); ap.add_argument('out'); ap.add_argument('--validation', default=None); ap.add_argument('--first-failure', action='store_true', help='classify the first RG failure (default boundary) of every point'); ar = ap.parse_args()
    os.makedirs(ar.out, exist_ok=True); rng = np.random.default_rng(1)
    rows, bad, S = load_points(ar.summary)
    import csv
    keys = sorted({k for r in rows for k in r if not k.startswith('_')}); w = csv.DictWriter(open(os.path.join(ar.out, 'points.csv'), 'w', newline=''), fieldnames=keys, extrasaction='ignore'); w.writeheader(); [w.writerow(r) for r in rows]
    ps = per_start(rows); res = dict(n_points=len(rows), n_derive_errors=len(bad), per_start={})
    for st, pts in sorted(ps.items(), key=lambda t: str(t[0])): res['per_start'][str(st)] = dict(lambda6=lambda6_study(pts, rng), n_points=len(pts), groups=sorted({p['group'] for p in pts}))
    if ar.first_failure:
        import cert as CE, collections
        for r in rows:
            try: r['first_failure'] = CE.cert(r['_x'], r['_eta'], ('default',))[1]['default']['fail']
            except Exception as ex: r['first_failure'] = 'error'
        for st, pts in ps.items(): res['per_start'][str(st)]['first_failure'] = dict(collections.Counter(p.get('first_failure') for p in pts))
        res['first_failure_all_starts_listed_separately'] = True
    make_figures(rows, ar.out, ar.validation)
    L = ['# Campaign analysis', '', 'Local explorations around the starts; statistics per start, never pooled; reach = verified 13-variant screening scale (not a bound); densities are not probabilities.', '',
         '%d distinct verified PASS points (%d could not be analysed).' % (len(rows), len(bad)), '']
    for st, d in res['per_start'].items():
        L.append('## start %s: %d points' % (st, d['n_points'])); l6 = d['lambda6']
        if l6['status'] != 'ok': L.append('lambda_6 study: ' + l6['status'] + ' (n=%d, seeds=%d)' % (l6['n'], l6['n_seeds']))
        else: L += ['lambda_6 vs log BR(h->tau mu): Spearman %.2f %s; partial (given m_H, max|Y2|) %.2f %s; |lambda_6| vs reach: %s' % (
            l6['spearman_lam6_logBRhtm']['value'], l6['spearman_lam6_logBRhtm']['ci'], l6['partial_given_mH_Y2']['value'], l6['partial_given_mH_Y2']['ci'], l6.get('spearman_abs_lam6_reach'))]
        L.append('')
    open(os.path.join(ar.out, 'report.md'), 'w').write('\n'.join(L)); json.dump(res, open(os.path.join(ar.out, 'analysis.json'), 'w'), indent=1, default=float)
    print('\n'.join(L))


if __name__ == '__main__': main()
