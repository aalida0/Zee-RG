"""Re-optimise the benchmarks with the free quartics (l7,l8,l10 in [-3,3]) so that chi2 <= chi2_min + 3.84, keeping the
tau->mu gamma rate (two-sided, 20%) and mu->e gamma (upper, +20%) at the benchmark values; then apply the shared gate
(gate2/accept.py) and rebuild the full robustness record for the best candidate per benchmark.
Usage: python3 -W ignore gate2/reopt_bench.py NPROC [NSEED=48] [BM1 BM2 BM3]  -> results/benchmarks_reopt.json, results/reopt_cands.pkl"""
import sys, os, json, pickle, time, numpy as np
os.environ.setdefault('ZEE_INPUTS', 'modern')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
from multiprocessing import Pool
from scipy.optimize import least_squares
import accept as AC, benchmarks as B
from zee import hgmodel as H
try:
    import hcoll; COLL = True
except Exception as ex:
    COLL = False; print('WARNING: collider module unavailable, gate runs without HB/HS cuts:', ex)
RES = os.path.join(ROOT, 'results'); NPROC = int(sys.argv[1]); NS = int(sys.argv[2]) if len(sys.argv) > 2 else 48
inp = json.load(open(os.path.join(RES, 'benchmarks_input.json'))); CMIN = float(pickle.load(open(os.path.join(RES, 'refit_corrected.pkl'), 'rb'))['NO'][0])
names = [a for a in sys.argv[3:] if a in inp] or list(inp)
S0 = 1e-6; ZM = np.arcsinh(1.0/S0); SQ = np.sqrt(4*np.pi)
LO = np.append(H.LO, -SQ)[:19]; HI = np.append(H.HI, SQ)[:19]; LO[:11] = -ZM; HI[:11] = ZM
xs = np.maximum(HI - LO, 1e-3)*0.1; xs[:11] = 1.0
tx = lambda z: np.concatenate([S0*np.sinh(z[:11]), z[11:19]])
tz = lambda x: np.concatenate([np.arcsinh(np.clip(x[:11], -1, 1)/S0), x[11:19]])
EB = 3.0


def fit(a):
    name, seed = a; r0 = inp[name]; x0 = np.asarray(r0['x'], float); e0 = np.asarray(r0['eta'], float)
    fac = (1.0, 0.85, 0.7, 0.55)[seed % 4]; t = {k: v*fac for k, v in TARGET[name].items()}; rng = np.random.default_rng(seed)
    z0 = np.clip(tz(x0) + (seed >= 4)*rng.normal(0, 1, 19)*xs*0.15, LO, HI)
    w0 = np.clip(e0[[1, 2, 4]] if seed < 4 else rng.uniform(-EB, EB, 3), -EB, EB)
    def eta_of(w): e = e0.copy(); e[[1, 2, 4]] = w; return e
    def resid(p):
        e = H.evaluate(list(tx(p[:19])), 'NO', eta=eta_of(p[19:])); o = e['obs']
        return np.concatenate([e['res'], [np.log(max(o['tau_mu_g'], 1e-30)/t['tm'])/0.2, max(0.0, np.log(max(o['mu_e_g'], 1e-30)/t['me']))/0.2]])
    try:
        r = least_squares(resid, np.concatenate([z0, w0]), bounds=(np.concatenate([LO, [-EB]*3]), np.concatenate([HI, [EB]*3])),
                          max_nfev=300, x_scale=np.concatenate([xs, [0.5]*3]))
        x = tx(r.x[:19]); e = eta_of(r.x[19:]); return name, float(H.evaluate(list(x), 'NO', eta=e)['chi2']), x, e, fac
    except Exception:
        return name, np.inf, None, None


def gate(a):
    name, c, x, e, fac = a
    try:
        r = AC.accept(x, e, CMIN, collider=COLL, n_vac=80, stop_early=True)
        return name, c, x, e, bool(r['accepted']), r.get('Lf', -1.0), {k: bool(v) for k, v in r['cuts'].items()}
    except Exception as ex:
        return name, c, x, e, False, -1.0, {'error': str(ex)}


TARGET = {}
if __name__ == '__main__':
    t0 = time.time()
    for n in names:
        o = H.evaluate(list(inp[n]['x']), 'NO', eta=inp[n]['eta'])['obs']; TARGET[n] = dict(tm=o['tau_mu_g'], me=o['mu_e_g'])
    print('chi2_min', CMIN, 'limit', CMIN + 3.84, 'targets', TARGET, flush=True)
    with Pool(NPROC) as P:
        fits = [r for r in P.imap_unordered(fit, [(n, s) for n in names for s in range(NS)]) if r[2] is not None]
        print('stage1 done %.0fs, fits %d' % (time.time() - t0, len(fits)), flush=True)
        cand = []
        for n in names:
            F = sorted([f for f in fits if f[0] == n and f[1] <= CMIN + 3.84], key=lambda f: f[1])[:3*NPROC]
            print(n, 'chi2 of best fits', [round(f[1], 2) for f in sorted([f for f in fits if f[0] == n], key=lambda f: f[1])[:5]], 'n_inside', len(F), flush=True)
            cand += F
        G = P.map(gate, cand, chunksize=1); print('stage2 done %.0fs' % (time.time() - t0), flush=True)
    pickle.dump(dict(fits=fits, gate=G), open(os.path.join(RES, 'reopt_cands.pkl'), 'wb'))
    out = {}
    for n in names:
        ok = sorted([g for g in G if g[0] == n and g[4]], key=lambda g: -g[5])
        print(n, 'accepted by the gate:', len(ok), 'of', len([g for g in G if g[0] == n]), 'best Lf', [round(g[5], 2) for g in ok[:5]], flush=True)
        if ok:
            g = ok[0]; out[n] = B.clean(B.record(n, g[2], g[3], CMIN)); print(n, 'record built', flush=True)
        else:
            bad = [g for g in G if g[0] == n]; print(n, 'no accepted candidate; failed cuts of the best:', bad[0][6] if bad else None)
    json.dump(out, open(os.path.join(RES, 'benchmarks_reopt.json'), 'w'), indent=1, allow_nan=False)
    for n, r in out.items():
        print(n, 'chi2 %.2f' % r['chi2'], {k: '%.3g' % r['observables'][k] for k in ('mu_e_g', 'tau_mu_g', 'd_e')}, {p: (round(v['log10Lambda'], 2), v['fail']) for p, v in r['robustness'].items() if p != 'vacuum'}, r['robustness']['vacuum']['status'])
    print('total %.0fs' % (time.time() - t0))
