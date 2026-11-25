"""Polish candidates from search_bench.py: all parameters free (including MHp, M33, u) with the production statistic, RG-friendliness
penalties, tau->mu gamma >= TM_MIN (one-sided, 'observable') and mu->e gamma <= 1e-13, light charged scalar >= M1MIN.
Usage: ZEE_INPUTS=modern python3 -W ignore gate2/polish_bench.py NPROC CANDS.json [N=6] [TM_MIN=1.5e-9] [M1MIN=460] -> results/polished.json"""
import sys, os, json, time, numpy as np
os.environ.setdefault('ZEE_INPUTS', 'modern')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
from multiprocessing import Pool
from scipy.optimize import least_squares
from zee import hgmodel as H
import accept as AC, refit_hw as R, bench
QB = float(os.environ.get('PQB', 1.5)); PMX = float(os.environ.get('PMX', 1.0)); ELO = np.array([0.02, -QB, -QB, -QB, -QB, 0.02])
TM_MIN = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5e-9; M1MIN = float(sys.argv[5]) if len(sys.argv) > 5 else 460.


def job(a):
    x0, e0 = np.array(a['x']), np.array(a['eta']); z0 = R.tz(x0)
    def unpack(p): return R.tx(p[:19]), np.asarray(p[19:], float)
    def resid(p):
        x, eta = unpack(p); e = H.evaluate(list(x), 'NO', eta=eta); o = e['obs']; sc = e['sc']
        c = AC.quartics(sc, eta); vb, u, mx = bench.margins(c)
        pen = [max(0.0, 0.05 - vb)/0.02, max(0.0, u - 12.0)/3.0, max(0.0, mx - PMX)/0.25, max(0.0, M1MIN - sc['m1'])/20.0]
        return np.concatenate([e['res'], pen, [max(0.0, np.log(TM_MIN/max(o['tau_mu_g'], 1e-30)))/0.2, max(0.0, np.log(max(o['mu_e_g'], 1e-30)/1.0e-13))/0.2]])
    p0 = np.concatenate([z0, np.clip(e0, -QB, QB)]); lo = np.concatenate([R.LO, ELO]); hi = np.concatenate([R.HI, [QB]*6])
    try:
        r = least_squares(resid, np.clip(p0, lo, hi), bounds=(lo, hi), max_nfev=300, x_scale=np.concatenate([R.XS, [0.3]*6]))
        x, eta = unpack(r.x); e = H.evaluate(list(x), 'NO', eta=eta)
        return dict(chi2=float(e['chi2']), x=x.tolist(), eta=eta.tolist(), m1=float(e['sc']['m1']), tm=float(e['obs']['tau_mu_g']), me=float(e['obs']['mu_e_g']))
    except Exception:
        return None


if __name__ == '__main__':
    NP = int(sys.argv[1]); C = json.load(open(sys.argv[2])); N = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    t0 = time.time()
    with Pool(NP) as P: out = [r for r in P.map(job, C[:N]) if r]
    out.sort(key=lambda r: r['chi2']); json.dump(out, open(os.path.join(ROOT, 'results', 'polished.json'), 'w'), indent=1)
    print('done %.0fs' % (time.time() - t0), [(round(r['chi2'], 2), round(r['m1']), '%.1e' % r['tm']) for r in out], flush=True)
