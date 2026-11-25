"""Heavy-singlet deformation search for benchmark candidates that satisfy the full gate (Drell-Yan, HiggsTools, B->Xs gamma, RG).
Starting points: the common-refit minimum and the earlier benchmarks.  M33 and u (x[13], x[14]) are drawn so that the light charged scalar is
heavy (m1 in [M1MIN, M1MAX]); then Yukawas, tan(beta), the doublet mass and the quartics lambda3-6 and lambda7,8,10 are re-fitted with the
production statistic, keeping tau->mu gamma at TARGET_TM (two-sided 25%) and mu->e gamma below its limit.
Usage: ZEE_INPUTS=modern python3 -W ignore gate2/search_bench.py NPROC NDRAW [TARGET_TM=3e-9] [M1MIN=460] -> results/search_cands.json"""
import sys, os, json, pickle, time, numpy as np
os.environ.setdefault('ZEE_INPUTS', 'modern')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
from multiprocessing import Pool
from scipy.optimize import least_squares
from zee import hgmodel as H
import accept as AC, refit_hw as R, bench
QB = 1.5; ELO = np.array([0.02, -QB, -QB, -QB, -QB, 0.02])    # lambda2, lh > 0 (BFB); lambda7, 8, 9, 10 in [-1.5, 1.5]
NP = int(sys.argv[1]); ND = int(sys.argv[2]); TM = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-9; M1MIN = float(sys.argv[4]) if len(sys.argv) > 4 else 460.
FIX = [12, 13, 14]                        # MHp, M33, u fixed by the deformation
FREE = [i for i in range(19) if i not in FIX]


def starts():
    d = json.load(open(os.path.join(ROOT, 'results', 'refit_hw.json')))
    inp = json.load(open(os.path.join(ROOT, 'results', 'benchmarks_input.json')))
    return [(np.array(v['x']), np.array(v['eta'])) for v in inp.values()]      # small-quartic starting points (the fit minimum sits at the quartic box edge)


def job(a):
    seed, x0, e0 = a; rng = np.random.default_rng(seed); x0 = x0.copy()
    for _ in range(200):                  # draw a deformation with a heavy light-charged state
        x0[13] = rng.uniform(2.65, 3.4); x0[14] = rng.uniform(0.05, 0.95); x0[12] = x0[12] + rng.normal(0, 0.05)
        sc = H.scalar_sector(x0)
        if sc is not None and M1MIN <= sc['m1'] <= 2500: break
    else: return None
    w0 = np.clip(e0, -QB, QB)
    z0 = R.tz(x0)
    def unpack(p):
        z = z0.copy(); z[FREE] = p[:len(FREE)]; return R.tx(z), np.asarray(p[len(FREE):], float)
    def resid(p):
        x, eta = unpack(p); e = H.evaluate(list(x), 'NO', eta=eta); o = e['obs']; sc = e['sc']
        c = AC.quartics(sc, eta); vb, u, mx = bench.margins(c)       # RG-friendliness: BFB margin, unitarity, size of the quartics
        pen = [max(0.0, 0.05 - vb)/0.02, max(0.0, u - 12.0)/3.0, max(0.0, mx - 1.0)/0.25]
        return np.concatenate([e['res'], pen, [np.log(max(o['tau_mu_g'], 1e-30)/TM)/0.25, max(0.0, np.log(max(o['mu_e_g'], 1e-30)/1.0e-13))/0.2]])
    p0 = np.concatenate([z0[FREE], w0]); lo = np.concatenate([R.LO[FREE], ELO]); hi = np.concatenate([R.HI[FREE], [QB]*6])
    try:
        r = least_squares(resid, np.clip(p0, lo, hi), bounds=(lo, hi), max_nfev=250, x_scale=np.concatenate([R.XS[FREE], [0.3]*6]))
        x, eta = unpack(r.x); e = H.evaluate(list(x), 'NO', eta=eta)
        return dict(seed=seed, chi2=float(e['chi2']), x=x.tolist(), eta=eta.tolist(), m1=float(e['sc']['m1']), tm=float(e['obs']['tau_mu_g']), me=float(e['obs']['mu_e_g']))
    except Exception as ex:
        return None


if __name__ == '__main__':
    S = starts(); S0_ = int(os.environ.get('SEED0', 0)); jobs = [(i + S0_, *S[i % len(S)]) for i in range(ND)]; t0 = time.time()
    with Pool(NP) as P: out = [r for r in P.imap_unordered(job, jobs) if r]
    out.sort(key=lambda r: r['chi2'])
    json.dump(out, open(os.path.join(ROOT, 'results', os.environ.get('SEARCH_OUT', 'search_cands.json')), 'w'), indent=1)
    print('done %.0fs' % (time.time() - t0), 'n', len(out), 'chi2 top', [round(r['chi2'], 2) for r in out[:8]], 'm1', [round(r['m1']) for r in out[:8]], flush=True)
