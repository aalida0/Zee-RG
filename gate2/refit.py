"""Laboratory minimum of the corrected chi2 (lambda4 free, 19 parameters) for NO and IO.
Seeds: the three published benchmarks, the previous best fits, the 40 best stored sample points, and for each of these
(a) copies with f scaled up by 10, 100, 1000 and Y2 scaled down by the same factor, and (b) 'small-mixing' copies with
f scaled up by k = 1e2..1e6 and the trilinear u scaled down by k (sin 2phi * f, hence M_nu, roughly unchanged), since the
corrected universality term allows f_mu_tau ~ 0.1 for light charged scalars. |f| <= FMAX (default 1).
Option SEEDS=prev also starts from the 20 best points of a previous results/refit_corrected.pkl.
Usage: ZEE_INPUTS=modern python gate2/refit.py NPROC [NSEED] [FMAX] -> results/refit_corrected.pkl"""
import sys, os, json, pickle, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from multiprocessing import Pool
from scipy.optimize import least_squares
from zee import hgmodel as H

NPROC = int(sys.argv[1]); NS = int(sys.argv[2]) if len(sys.argv) > 2 else 40; FMAX = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
S0 = 1e-6; ZM = np.arcsinh(FMAX/S0); SQ = np.sqrt(4*np.pi)
LO = np.append(H.LO, -SQ); HI = np.append(H.HI, SQ); LO[:11] = -ZM; HI[:11] = ZM
xs = np.maximum(HI - LO, 1e-3)*0.1; xs[:11] = 1.0
tx = lambda z: np.concatenate([S0*np.sinh(z[:11]), z[11:]])
tz = lambda x: np.concatenate([np.arcsinh(np.clip(np.asarray(x[:11], float), -FMAX, FMAX)/S0), np.asarray(list(x[11:19]) + [0.0]*(19 - len(x)), float)])


def fit(a):
    o, x = a
    try:
        r = least_squares(lambda z: H.evaluate(tx(z), o)['res'], np.clip(tz(x), LO, HI), bounds=(LO, HI), max_nfev=400, x_scale=xs)
        xb = tx(r.x); return float(H.evaluate(list(xb), o)['chi2']), xb
    except Exception as ex:
        return np.inf, None


def seeds(o):
    S = []
    bm = os.path.join(ROOT, 'results', 'benchmarks.json')
    if os.path.exists(bm): S += [np.array(r['x']) for r in json.load(open(bm)).values()]
    for f in ('/tmp/refit19.pkl',):
        try: S.append(np.asarray(pickle.load(open(f, 'rb'))[o][1]))
        except Exception: pass
    for f in (f'/tmp/mbest_{o}.pkl', f'/tmp/best_{o}.pkl'):
        try: S += [np.asarray(x) for _, x in pickle.load(open(f, 'rb'))[:NS // 4]]
        except Exception: pass
    prev = os.path.join(ROOT, 'results', 'refit_corrected.pkl')
    if os.environ.get('SEEDS') == 'prev' and os.path.exists(prev):
        S += [np.asarray(x) for _, x in pickle.load(open(prev, 'rb'))[o + '_all']]
    if o == 'NO':
        try:
            P = pickle.load(open('/tmp/mfilt4w_NO.pkl', 'rb'))['S']
            ch = sorted([(H.evaluate(list(x), 'NO')['chi2'], x) for _, x in P], key=lambda t: t[0])
            S += [np.asarray(x) for _, x in ch[:NS]]
        except Exception: pass
    out = []
    for x in S:
        x = np.asarray(x, float).copy()
        if len(x) < 19: x = np.append(x, [0.0]*(19 - len(x)))
        out.append(x)
        for k in (10.0, 100.0, 1000.0):
            y = x.copy(); y[:2] *= k; y[2:11] /= k
            if np.abs(y[:2]).max() <= FMAX: out.append(y)
        for k in (1e2, 1e3, 1e4, 1e5, 1e6):
            for both in (True, False):
                y = x.copy(); y[1] *= k; y[14] /= k
                if both: y[0] *= k
                if np.abs(y[:2]).max() <= FMAX: out.append(y)
    return out


if __name__ == '__main__':
    res = {}
    with Pool(NPROC) as P:
        for o in ('NO', 'IO'):
            J = [(o, x) for x in seeds(o)]
            R = sorted([r for r in P.map(fit, J, chunksize=1) if r[1] is not None], key=lambda t: t[0])
            res[o] = R[0]; res[o + '_all'] = R[:20]
            print(o, 'n_seeds', len(J), 'chi2min %.3f' % R[0][0], 'next', [round(r[0], 2) for r in R[1:6]], flush=True)
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    pickle.dump(res, open(os.path.join(ROOT, 'results', 'refit_corrected.pkl'), 'wb'))
