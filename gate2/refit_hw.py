"""Common refit with the production statistic hgmodel.evaluate (H+-W-gamma set included): Yukawas |f|,|Y2| <= 0.1, the three free
quartics lambda7, lambda8, lambda10 in [-3, 3] (lambda2 = lambda9 = lh = 0), lambda4 free.  Starts: the stored corrected NO fits and
random perturbations of them.  Output: results/refit_hw.json (minimum, best point, all local minima).
Usage: ZEE_INPUTS=modern python3 -W ignore gate2/refit_hw.py NPROC [NPERT=2]"""
import sys, os, json, pickle, time, numpy as np
os.environ.setdefault('ZEE_INPUTS', 'modern')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
from multiprocessing import Pool
from scipy.optimize import least_squares
from zee import hgmodel as H
import accept as AC
ORD = os.environ.get('ORD', 'NO')   # mass ordering of the fit (NO default; IO for the ordering difference)
S0 = 1e-6; ZM = np.arcsinh(AC.FMAX/S0); SQ = np.sqrt(4*np.pi); EB = AC.ETA_MAX
LO = np.append(H.LO, -SQ)[:19].copy(); HI = np.append(H.HI, SQ)[:19].copy(); LO[:11] = -ZM; HI[:11] = ZM
tx = lambda z: np.concatenate([S0*np.sinh(z[:11]), z[11:19]])
tz = lambda x: np.concatenate([np.arcsinh(np.clip(np.asarray(x[:11], float), -AC.FMAX, AC.FMAX)/S0), np.asarray(x[11:19], float)])
XS = np.maximum(HI - LO, 1e-3)*0.1; XS[:11] = 1.0
eta_of = lambda w: np.array([0.0, w[0], w[1], 0.0, w[2], 0.0])


def run(a):
    seed, z0, w0 = a
    def resid(p):
        e = H.evaluate(list(tx(p[:19])), ORD, eta=eta_of(p[19:])); return e['res']
    try:
        r = least_squares(resid, np.concatenate([z0, w0]), bounds=(np.concatenate([LO, [-EB]*3]), np.concatenate([HI, [EB]*3])),
                          max_nfev=400, x_scale=np.concatenate([XS, [0.5]*3]))
        x = tx(r.x[:19]); w = r.x[19:]; c = float(H.evaluate(list(x), ORD, eta=eta_of(w))['chi2'])
        return dict(seed=seed, chi2=c, x=x.tolist(), eta=eta_of(w).tolist())
    except Exception as ex:
        return dict(seed=seed, chi2=float('inf'), err=str(ex))


if __name__ == '__main__':
    NP = int(sys.argv[1]); NPT = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    d = pickle.load(open(os.path.join(ROOT, 'results', 'refit_corrected.pkl'), 'rb'))
    starts = [np.asarray(d[ORD][1], float)] + [np.asarray(a[1], float) for a in d[ORD + '_all']]
    rng = np.random.default_rng(1); jobs = []
    for i, x in enumerate(starts):
        z = np.clip(tz(x), LO, HI); w = np.clip(np.array(d['eta_min'])[[1, 2, 4]], -EB, EB)
        jobs.append((len(jobs), z, w))
        for _ in range(NPT): jobs.append((len(jobs), np.clip(z + rng.normal(0, 1, 19)*XS*0.15, LO, HI), rng.uniform(-EB, EB, 3)))
    t0 = time.time(); print('jobs', len(jobs), flush=True)
    with Pool(NP) as P: R = list(P.imap_unordered(run, jobs))
    R = sorted([r for r in R if np.isfinite(r['chi2'])], key=lambda r: r['chi2'])
    json.dump(dict(best=R[0], all=R[:50], note='production statistic with H+-W-gamma set, |x|<=0.1, lambda7,8,10 in [-3,3]'),
              open(os.path.join(ROOT, 'results', 'refit_hw.json' if ORD == 'NO' else 'refit_hw_%s.json' % ORD), 'w'), indent=1)
    print('done %.0fs' % (time.time() - t0), 'best chi2', R[0]['chi2'], 'eta', R[0]['eta'], 'top5', [round(r['chi2'], 3) for r in R[:5]], flush=True)
