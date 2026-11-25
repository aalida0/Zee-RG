"""Diverse valid starting points for the campaign chains.  Draws the doublet-sector scales (tan beta, M_Hp, M33, u) broadly, refits Yukawas,
lambda3-6 and the free quartics with the production statistic plus RG-friendliness penalties and a random tau->mu gamma / mu->e gamma target
(so the starts spread in rate as well as in mass), and keeps those that pass the staged gate of gate2/campaign.py (everything but the cutoff).
Usage: ZEE_INPUTS=modern python3 -W ignore gate2/make_starts.py NPROC NDRAW SEED0 -> results/starts_<SEED0>.json"""
import sys, os, json, time, numpy as np
os.environ.setdefault('ZEE_INPUTS', 'modern')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
from multiprocessing import Pool
from scipy.optimize import least_squares
from zee import hgmodel as H
import accept as AC, refit_hw as R, bench, campaign as C
QB = 1.5; ELO = np.array([0.02, -QB, -QB, -QB, -QB, 0.02])
FIX = [12, 13, 14]; FREE = [i for i in range(19) if i not in FIX]
BASE = [(np.array(v['x']), np.array(v['eta'])) for v in json.load(open(os.path.join(ROOT, 'results', 'benchmarks_input.json'))).values()]


def job(a):
    seed = a; rng = np.random.default_rng(seed); x0, e0 = BASE[seed % len(BASE)]; x0 = x0.copy()
    for _ in range(300):
        x0[11] = x0[11] + rng.normal(0, 0.1); x0[12] = x0[12] + rng.normal(0, 0.25); x0[13] = rng.uniform(2.65, 3.6); x0[14] = rng.uniform(0.03, 0.97)
        sc = H.scalar_sector(x0)
        if sc is not None and sc['m1'] >= 150: break
    else: return None
    TM = 10**rng.uniform(-10, np.log10(2e-8)); ME = 10**rng.uniform(-15.5, -13.2); z0 = R.tz(x0); w0 = np.clip(e0, -QB, QB)
    def unpack(p):
        z = z0.copy(); z[FREE] = p[:len(FREE)]; return R.tx(z), np.asarray(p[len(FREE):], float)
    def resid(p):
        x, eta = unpack(p); e = H.evaluate(list(x), 'NO', eta=eta); o = e['obs']; sc = e['sc']
        vb, u, mx = bench.margins(AC.quartics(sc, eta))
        pen = [max(0.0, 0.05 - vb)/0.02, max(0.0, u - 12.0)/3.0, max(0.0, mx - 1.0)/0.25]
        return np.concatenate([e['res'], pen, [np.log(max(o['tau_mu_g'], 1e-30)/TM)/0.5, np.log(max(o['mu_e_g'], 1e-30)/ME)/0.5]])
    p0 = np.concatenate([z0[FREE], w0]); lo = np.concatenate([R.LO[FREE], ELO]); hi = np.concatenate([R.HI[FREE], [QB]*6])
    try:
        r = least_squares(resid, np.clip(p0, lo, hi), bounds=(lo, hi), max_nfev=120, x_scale=np.concatenate([R.XS[FREE], [0.3]*6]))
        x, eta = unpack(r.x); ok, info, why = C.stage_gate(x, eta, -1.0)
        if not ok: return dict(ok=False, why=why, seed=seed, chi2=float(info['e']['chi2']), m1=float(info['e']['sc']['m1']))
        sc = info['e']['sc']
        return dict(ok=True, seed=seed, x=x.tolist(), eta=eta.tolist(), chi2=float(info['e']['chi2']), m1=float(sc['m1']), mA=float(sc['mA']), mH=float(sc['mH']), tb=float(sc['tb']),
                    tm=float(info['e']['obs']['tau_mu_g']), me=float(info['e']['obs']['mu_e_g']), Lf=info['Lf'])
    except Exception as ex:
        return dict(ok=False, why='err ' + type(ex).__name__, seed=seed)


if __name__ == '__main__':
    NP, ND, S0_ = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); t0 = time.time()
    with Pool(NP) as P: out = [r for r in P.imap_unordered(job, range(S0_, S0_ + ND)) if r]
    good = [r for r in out if r['ok']]
    import collections; print('done %.0fs' % (time.time() - t0), 'draws', len(out), 'valid', len(good), collections.Counter(r.get('why') for r in out if not r['ok']), [(round(r['chi2'],1), round(r['m1'])) for r in out if not r['ok'] and 'chi2' in r][:12], flush=True)
    json.dump(good, open(os.path.join(ROOT, 'results', 'starts_%d.json' % S0_), 'w'), indent=1)
