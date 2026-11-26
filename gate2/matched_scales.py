"""One-loop (effective-potential) threshold matching of the boundary quartics of a benchmark and the resulting scalar-consistency scale (zee/matching.py).
For each prescribed boundary variant (the 13 of gate2/cert.FULL) the quartics lambda_1, lambda_4, lambda_5, lambda_6 are re-matched at that variant's mu0 with that variant's
SM couplings; qscale variants multiply the matched default quartics.  The scale is then obtained with the production RGEs (bench.trajectory, 100 scales per decade).
Usage: ZEE_INPUTS=modern python3 -W ignore gate2/matched_scales.py NAME [KAPPA] -> $ZEE_OUT/matching_<NAME>_k<KAPPA>.json  (NAME in benchmarks_input.json or S<i> of campaign_starts.json)"""
import sys, os, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
os.environ.setdefault('ZEE_INPUTS', 'modern')
import bench, cert as CE
from zee import matching as M

G0 = np.array([0.36, 0.64, 1.05, 0.88])


def piece_matched(x, eta, p, kappa=1.0, cache={}):
    sc, c = bench.coup(x, eta); r = -sc['tb']; ml = min(sc['m1'], sc['mA'], sc['mH']); t0 = np.log(ml); mu0, g0, scale = ml, G0, 1.0
    if p.startswith('match_'): mu0 = max(172.5, ml*float(p.split('_x')[1])); g0 = bench.sm_run(mu0)
    elif p.startswith('matchH'): mu0 = max(172.5, max(sc['mH'], sc['mA'], sc['m2'])*float(p.split('_x')[1])); g0 = bench.sm_run(mu0)
    elif p.startswith('qscale'): scale = float(p.split('_')[1])
    elif p.startswith('sm_'):
        kind, k = p.split('_')[1], float(p.split('_')[2]); g = np.array(bench.SM_MT, float); g[3 if kind == 'yt' else 2] *= k; g0 = bench.sm_run(max(172.5, ml), start=tuple(g))
    key = (p.split('_')[0] if p.startswith('qscale') else p)       # qscale reuses the default matching
    key = 'default' if p.startswith('qscale') else p
    if key not in cache:
        cache[key] = M.match(c, sc, mu0, tuple(g0[:3]), float(g0[3]), r, kappa=kappa)
    m = cache[key]; cm = m['c']*scale
    out = bench.trajectory(cm, np.log(mu0), r, g0)
    vb, u, mx = bench.margins(cm)
    return dict(log10Lambda=float(out['log10Lambda']), fail=out['fail'], mu0=float(mu0), delta=[float(v) for v in m['delta']], bfb_margin=float(vb), max_unit=float(u), max_lam=float(mx),
                match_ok=m['ok'], match_resid=m['resid'])


if __name__ == '__main__':
    name = sys.argv[1]; kappa = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    inp = json.load(open(os.path.join(ROOT, 'results', 'benchmarks_input.json')))
    st = {('S%d' % i): d for i, d in enumerate(json.load(open(os.path.join(ROOT, 'results', 'campaign_starts.json'))))}
    d = inp[name] if name in inp else st[name]; x, eta = np.array(d['x'], float), np.array(d['eta'], float)
    res = {p: piece_matched(x, eta, p, kappa) for p in CE.FULL}
    low = min(r['log10Lambda'] for r in res.values())
    sc, c = bench.coup(x, eta); tree = bench.margins(c)[0]
    outdir = os.environ.get('ZEE_OUT', os.path.join(ROOT, 'results', 'matching'))
    try: os.makedirs(outdir, exist_ok=True)
    except OSError: outdir = '/tmp'
    json.dump(dict(name=name, kappa=kappa, min=low, tree_bfb_margin=float(tree), variants=res), open(os.path.join(outdir, 'matching_%s_k%g.json' % (name, kappa)), 'w'), indent=1)
    print(name, 'kappa', kappa, 'min matched %.2f' % low, {p: (round(r['log10Lambda'], 2), r['fail']) for p, r in res.items()})
