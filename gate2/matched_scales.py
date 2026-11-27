"""One-loop threshold matching of the boundary quartics of a benchmark and the resulting scalar-consistency scale (zee/matching.py, zee/matching_full.py).
For each prescribed boundary variant (the 13 of gate2/cert.FULL) the quartics lambda_1, lambda_4, lambda_5, lambda_6 are re-matched at that variant's mu0 with that variant's SM couplings;
the scale is then obtained with the production RGEs (bench.trajectory, 100 scales per decade).  The 'qscale' variants are NOT re-matched: they multiply the matched default quartics (a stress test of the
matched point, not an independent matching) and are reported separately.
Fail-closed semantics (report v11): a variant whose matching did not converge (match_ok False, residual > MATCH_TOL, non-finite quartics/parameters, or the light state cannot be identified) is recorded as
fail='match_failed' with no scale, and a boundary that is unbounded / non-perturbative at the start is 'start:...' with no scale; neither is ever turned into a number.  The file reports
  min_rematched   minimum scale over the re-matched variants that have a defined scale   (None if none)
  min_stress      the same including the qscale stress variants
  complete        True only if every variant has a defined scale from a physical first failure or a ceiling; n_match_failed, n_start_fail, n_numerical count the others
                  (numerical stops and borderline BFB keep their finite prefix in numerical_prefix_min and never become a scale).
The matched quartics, mass parameters, target (input) spectrum and the off-diagonal pole residual are stored for each variant.
Usage: ZEE_INPUTS=modern python3 -W ignore gate2/matched_scales.py NAME [KAPPA] [MODE=ep|full-{sym,min,max}|full-hpole-{sym,min,max}] -> $ZEE_OUT/matching_<NAME>_k<KAPPA>.json  (NAME in benchmarks_input.json or S<i> of campaign_starts.json)"""
import sys, os, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
os.environ.setdefault('ZEE_INPUTS', 'modern')
import bench, cert as CE
from zee import matching as M, matching_full as MF

G0 = np.array([0.36, 0.64, 1.05, 0.88]); MATCH_TOL = 1e-6; _CACHE = {}


def _key(x, eta, mode, kappa, mu0, g0):
    return (np.asarray(x, float).tobytes(), np.asarray(eta, float).tobytes(), mode, float(kappa), float(np.round(mu0, 9)), tuple(np.round(np.asarray(g0, float), 12)))


def do_match(c, sc, x, eta, mu0, g0, r, mode, kappa, cache=None):
    """matching at (mu0, g0); cached under an immutable key containing the point, mode, kappa, mu0 and the SM couplings"""
    cache = _CACHE if cache is None else cache; key = _key(x, eta, mode, kappa, mu0, g0)
    if key not in cache:
        try:
            if mode == 'ep': m = M.match(c, sc, mu0, tuple(g0[:3]), float(g0[3]), r, kappa=kappa)
            else: m = MF.match_full(c, sc, mu0, tuple(g0[:3]), float(g0[3]), r, prescription=mode.split('-', 1)[1], kappa=kappa)
            m['error'] = None
        except Exception as e:                                         # includes the light-state identification check
            m = dict(c=None, P=None, delta=None, ok=False, resid=float('nan'), error=repr(e))
        cache[key] = m
    return cache[key]


def match_valid(m):
    """None if the matching is usable, else the reason"""
    if m.get('error'): return m['error']
    if not m['ok']: return 'root finder did not converge'
    if not (np.isfinite(m['resid']) and m['resid'] <= MATCH_TOL): return 'residual %.3g > %.1g' % (m['resid'], MATCH_TOL)
    if not (np.all(np.isfinite(m['c'])) and all(np.isfinite(v) for v in m['P'].values())): return 'non-finite matched parameters'
    return None


def piece_matched(x, eta, p, kappa=1.0, cache=None, mode='ep'):
    sc, c = bench.coup(x, eta); r = -sc['tb']; ml = min(sc['m1'], sc['mA'], sc['mH']); mu0, g0, scale = ml, G0, 1.0
    if p.startswith('match_'): mu0 = max(172.5, ml*float(p.split('_x')[1])); g0 = bench.sm_run(mu0)
    elif p.startswith('matchH'): mu0 = max(172.5, max(sc['mH'], sc['mA'], sc['m2'])*float(p.split('_x')[1])); g0 = bench.sm_run(mu0)
    elif p.startswith('qscale'): scale = float(p.split('_')[1])
    elif p.startswith('sm_'):
        kind, k = p.split('_')[1], float(p.split('_')[2]); g = np.array(bench.SM_MT, float); g[3 if kind == 'yt' else 2] *= k; g0 = bench.sm_run(max(172.5, ml), start=tuple(g))
    base = dict(mu0=float(mu0), stress=p.startswith('qscale'), g0=[float(v) for v in g0])
    # qscale variants reuse the matching of 'default' at its own (mu0, g0): the cache key contains mu0 and g0, so this is the same entry
    m = do_match(c, sc, x, eta, mu0, g0, r, mode, kappa, cache); why = match_valid(m)
    if why is not None:
        return dict(base, log10Lambda=None, fail='match_failed', match_error=why, match_ok=bool(m['ok']), match_resid=float(m['resid']))
    cm = m['c']*scale
    out = bench.trajectory(cm, np.log(mu0), r, g0)
    vb, u, mx = bench.margins(cm)
    start = str(out['fail']).startswith('start')
    rec = dict(base, log10Lambda=None if start else float(out['log10Lambda']), fail=out['fail'], delta=[float(v) for v in m['delta']], bfb_margin=float(vb), max_unit=float(u), max_lam=float(mx),
               match_ok=bool(m['ok']), match_resid=float(m['resid']), matched_c=[float(v) for v in m['c']], matched_P={k: float(v) for k, v in m['P'].items()})
    if 'target_masses' in m: rec.update(target_masses=m['target_masses'], mh_state=m['mh_state'])
    if mode.startswith('full') and p in ('default',):
        try: rec['pole_residual'] = MF.pole_residual(m, c, sc, mu0, tuple(g0[:3]), float(g0[3]), r)
        except Exception as e: rec['pole_residual'] = None
    return rec


def summarize(res):
    """fail-closed summary: only variants whose result is a physical first failure or a ceiling (cert.variant_status) carry a scale.  Start failures, failed matchings, numerical stops
    (integration/nonfinite) and borderline BFB are counted separately; the finite prefix of a numerical stop is reported as numerical_prefix_min and never enters min_rematched/min_stress/complete."""
    st = {p: CE.variant_status(v['fail']) for p, v in res.items()}
    ok = {p: v['log10Lambda'] for p, v in res.items() if CE.eligible(st[p]) and v['log10Lambda'] is not None}
    rem = [v for p, v in ok.items() if not res[p]['stress']]
    num = [res[p]['log10Lambda'] for p in res if st[p] in ('numerical_stop', 'unresolved') and res[p]['log10Lambda'] is not None]
    return dict(min_rematched=min(rem) if rem else None, min_stress=min(ok.values()) if ok else None,
                complete=len(ok) == len(res), n_defined=len(ok), n_variants=len(res),
                n_match_failed=sum(s == 'match_failed' for s in st.values()), n_start_fail=sum(s == 'boundary_failure' for s in st.values()),
                n_numerical=sum(s in ('numerical_stop', 'unresolved') for s in st.values()), numerical_prefix_min=min(num) if num else None, variant_status=st)


if __name__ == '__main__':
    name = sys.argv[1]; kappa = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0; mode = sys.argv[3] if len(sys.argv) > 3 else 'ep'   # mode: ep | full-sym | full-min | full-max | full-hpole-X
    inp = json.load(open(os.path.join(ROOT, 'results', 'benchmarks_input.json')))
    st = {('S%d' % i): d for i, d in enumerate(json.load(open(os.path.join(ROOT, 'results', 'campaign_starts.json'))))}
    d = inp[name] if name in inp else st[name]; x, eta = np.array(d['x'], float), np.array(d['eta'], float)
    res = {p: piece_matched(x, eta, p, kappa, mode=mode) for p in CE.FULL}
    S = summarize(res); S['min'] = S['min_stress']         # 'min' kept for older readers: minimum over the variants with a defined scale (see 'complete')
    sc, c = bench.coup(x, eta); tree = bench.margins(c)[0]
    outdir = os.environ.get('ZEE_OUT', os.path.join(ROOT, 'results', 'matching'))
    try: os.makedirs(outdir, exist_ok=True)
    except OSError: outdir = '/tmp'
    json.dump(dict(name=name, kappa=kappa, mode=mode, tree_bfb_margin=float(tree), match_tol=MATCH_TOL, **S, variants=res),
              open(os.path.join(outdir, ('matching_%s_k%g.json' % (name, kappa)) if mode == 'ep' else ('matching_%s_%s.json' % (name, mode))), 'w'), indent=1)
    print(name, mode, 'kappa', kappa, {k: v for k, v in S.items()}, {p: (None if r['log10Lambda'] is None else round(r['log10Lambda'], 2), r['fail']) for p, r in res.items()})
