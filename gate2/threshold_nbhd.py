"""Sensitivity of the one-loop scalar-consistency scale to INDEPENDENT shifts of the boundary quartics at fixed low-energy point
(a stand-in for unknown one-loop threshold corrections; the shifted points are not calculated corrections and give no uncertainty interval).
(i) one coupling at a time, Delta in SHIFTS; (ii) random neighbourhood: all 11 quartics shifted independently by U(-d, d), d = 0.05, 0.10.
Default matching prescription, screening grid of gate2/cert.run.  Usage: ZEE_INPUTS=modern python3 -W ignore gate2/threshold_nbhd.py [BM1 BM2 ...]
-> results/threshold_nbhd.json"""
import sys, os, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
os.environ.setdefault('ZEE_INPUTS', 'modern')
import bench, cert as CE
NAMES = ['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9', 'l10', 'lh']
SHIFTS = (-0.2, -0.1, -0.05, 0.05, 0.1, 0.2)


def scale(c, sc):
    v, why = CE.run(c, np.log(min(sc['m1'], sc['mA'], sc['mH'])), -sc['tb'], CE.G0); return float(v), str(why)


if __name__ == '__main__':
    inp = json.load(open(os.environ.get('BM_INPUT', os.path.join(ROOT, 'results', 'benchmarks_input.json')))); names = [n for n in inp if len(sys.argv) < 2 or n in sys.argv[1:]]; out = {}
    for n in names:
        x, eta = np.array(inp[n]['x']), np.array(inp[n]['eta']); sc, c0 = bench.coup(x, eta); base = scale(c0, sc)
        one = {NAMES[i]: {('%+.2f' % d): scale(c0 + d*np.eye(11)[i], sc) for d in SHIFTS} for i in range(11)}
        rng = np.random.default_rng(7); rnd = {}
        for d in (0.05, 0.10):
            res = [scale(c0 + rng.uniform(-d, d, 11), sc) for _ in range(200)]; L = np.array([r[0] for r in res])
            rnd['%.2f' % d] = dict(n=len(L), start_fail=int(np.sum(L < 0)), frac_ge_10=float(np.mean(L >= 10)), frac_ge_11=float(np.mean(L >= 11)), frac_ge_12=float(np.mean(L >= 12)),
                                   median=float(np.median(L)), p16=float(np.percentile(L, 16)), p84=float(np.percentile(L, 84)))
        out[n] = dict(base=base, boundary_quartics=dict(zip(NAMES, map(float, c0))), one_at_a_time=one, random=rnd)
        print(n, 'base %.2f' % base[0], {k: rnd[k] for k in rnd}, flush=True)
    json.dump(out, open(os.environ.get('BM_OUTPUT', os.path.join(ROOT, 'results', 'threshold_nbhd.json')), 'w'), indent=1)
