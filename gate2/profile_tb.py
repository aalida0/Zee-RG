"""Profile of the production chi2 in tan(beta): ln tan(beta) fixed on a grid, all other parameters (and lambda7,8,10 in [-3,3]) refitted
from the global minimum and perturbed starts.  Usage: ZEE_INPUTS=modern python3 -W ignore gate2/profile_tb.py NPROC -> results/profile_tb.json"""
import sys, os, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
os.environ.setdefault('ZEE_INPUTS', 'modern')
from multiprocessing import Pool
from scipy.optimize import least_squares
from zee import hgmodel as H
import refit_hw as R
GRID = [0.3, 0.4, 0.6, 1.0, 2.0, 4.0, 10.0]


def run(a):
    tbv, k = a; best = json.load(open(os.path.join(ROOT, 'results', 'refit_hw.json')))['best']
    x0 = np.array(best['x']); x0[11] = np.log(tbv); z0 = R.tz(x0); w0 = np.array(best['eta'])[[1, 2, 4]]
    rng = np.random.default_rng(100 * k + int(10 * tbv))
    if k: z0 = z0 + rng.normal(0, 1, 19) * R.XS * 0.15; z0[11] = np.log(tbv); w0 = rng.uniform(-3, 3, 3)
    lo = np.concatenate([R.LO, [-3] * 3]); hi = np.concatenate([R.HI, [3] * 3]); lo[11] = np.log(tbv) - 1e-9; hi[11] = np.log(tbv) + 1e-9
    z0 = np.clip(z0, lo[:19], hi[:19])
    def resid(p):
        z = p[:19]; return H.evaluate(list(R.tx(z)), 'NO', eta=R.eta_of(p[19:]))['res']
    try:
        r = least_squares(resid, np.concatenate([z0, w0]), bounds=(lo, hi), max_nfev=300, x_scale=np.concatenate([R.XS, [0.5] * 3]))
        x = R.tx(r.x[:19]); e = H.evaluate(list(x), 'NO', eta=R.eta_of(r.x[19:]))
        return dict(tb=tbv, chi2=float(e['chi2']), x=x.tolist(), eta=R.eta_of(r.x[19:]).tolist(), parts={k_: float(v) for k_, v in e.get('parts', {}).items()} if isinstance(e.get('parts'), dict) else None)
    except Exception as ex:
        return dict(tb=tbv, chi2=float('inf'), err=str(ex))


if __name__ == '__main__':
    NP = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    with Pool(NP) as P: out = P.map(run, [(t, k) for t in GRID for k in range(3)])
    best = {}
    for r in out:
        if r['tb'] not in best or r['chi2'] < best[r['tb']]['chi2']: best[r['tb']] = r
    json.dump(list(best.values()), open(os.path.join(ROOT, 'results', 'profile_tb.json'), 'w'), indent=1)
    for t in GRID: print('tanb %5.2f  chi2 %.2f' % (t, best[t]['chi2']), flush=True)
