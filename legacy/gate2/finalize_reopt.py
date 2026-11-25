"""Re-gate the saved re-optimisation candidates (results/reopt_cands.pkl) with the collider cuts, pick per benchmark the accepted
candidate with the highest RG reach (ties: larger tau->mu gamma), and build the full records -> results/benchmarks_reopt.json"""
import sys, os, json, pickle, numpy as np
os.environ.setdefault('ZEE_INPUTS', 'modern')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
from multiprocessing import Pool
import accept as AC, benchmarks as B
RES = os.path.join(ROOT, 'results'); CMIN = float(pickle.load(open(os.path.join(RES, 'refit_corrected.pkl'), 'rb'))['NO'][0])
def g(a):
    n, c, x, e = a[:4]
    try: r = AC.accept(x, e, CMIN, n_vac=80); return n, c, x, e, bool(r['accepted']), r['Lf'], r['obs']['tau_mu_g'], {k: bool(v) for k, v in r['cuts'].items()}
    except Exception as ex: return n, c, x, e, False, -1.0, 0.0, {'error': str(ex)}
if __name__ == '__main__':
    D = pickle.load(open(os.path.join(RES, 'reopt_cands.pkl'), 'rb'))['gate']
    with Pool(int(sys.argv[1])) as P: G = P.map(g, D, chunksize=1)
    out = {}
    for n in sorted({a[0] for a in G}):
        L = [a for a in G if a[0] == n]; ok = sorted([a for a in L if a[4]], key=lambda a: (-round(a[5], 1), -a[6]))
        print(n, 'gated', len(L), 'accepted', len(ok), 'reach', [round(a[5], 2) for a in ok[:6]], 'failed cuts', {k for a in L if not a[4] for k, v in a[7].items() if not v})
        if ok: out[n] = B.clean(B.record(n, ok[0][2], ok[0][3], CMIN))
    json.dump(out, open(os.path.join(RES, 'benchmarks_reopt.json'), 'w'), indent=1, allow_nan=False)
    for n, r in out.items():
        print(n, 'chi2 %.2f' % r['chi2'], {k: '%.3g' % r['observables'][k] for k in ('mu_e_g', 'tau_mu_g', 'tau_e_g', 'd_e', 'sum_m')}, 'BR h->tau mu %.2g' % r['collider']['BR_h_tau_mu'])
        print('   ', {p: (round(v['log10Lambda'], 2), v['fail']) for p, v in r['robustness'].items() if p != 'vacuum'}, 'vacuum', r['robustness']['vacuum']['status'])
