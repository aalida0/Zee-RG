"""Screen every stored witness (sample + dedicated searches) with the corrected code: continuous likelihood, modern EDM bound,
lambda_h<0 boundedness, same-Lagrangian HiggsTools (gate2/hcoll.py), fine-grid cutoff (10/decade) and global minimum.
Usage: ZEE_INPUTS=modern python gate2/screen.py NPROC OUT.pkl"""
import sys, glob, json, pickle, numpy as np
sys.path.insert(0, '/home/claude/zee_fit'); sys.path.insert(0, '/home/claude/zee_fit/gate1'); sys.path.insert(0, '/home/claude/zee_fit/gate2')
from multiprocessing import Pool
import lmax2 as L, m_filter_l4w as M, hcoll
from zee import vac
from m_filter_l4w import H
L.SC = np.minimum(np.log(10.0**np.linspace(3, 16, 131)), np.log(1e16))
KEYS = ['tau_mu_g', 'tau_e_g', 'mu_e_g', 'tau_3mu', 'tau_mumue', 'Cr_Au', 'Cr_Al', 'mu_3e', 'h_tau_mu', 'h_tau_e', 'd_e', 'd_mu', 'da_e', 'das_mu', 'm_ee', 'm_beta', 'sum_m']
def one(a):
    src, x, eta = a
    try:
        x = np.asarray(x, float); eta = np.asarray(eta, float)
        col, e = hcoll.run(x, eta); sc = e['sc']; o = e['obs']
        c = np.zeros(11); c[0], c[2], c[3], c[4], c[5] = sc['lam1'], sc['lam3'], sc['lam4'], sc['lam5'], sc['lam6']; c[1], c[6:] = eta[0], eta[1:]
        dgg = ((e['mus']['gg']*col['rgg']/e['rgg'] - 1.14)/0.185)**2 - ((e['mus']['gg'] - 1.14)/0.185)**2
        hard = all(o[k] <= lim for k, (lim, z) in H.BOUNDS.items()) and col['BR_h_tau_mu'] <= 0.0018 and col['BR_h_tau_e'] <= 0.0020
        t0 = np.log(min(sc['m1'], sc['mA'], sc['mH'])); Lf = float(L.lam_valid(c, t0, -sc['tb']))
        g = bool(vac.global_ok(c, sc, nstart=80, rng=np.random.default_rng(11))[0]) if Lf >= 3 else False
        return dict(src=src, x=x, eta=eta, chi2=float(e['chi2'] + dgg), hard=hard, Lf=Lf, glob=g, **{k: float(o[k]) for k in KEYS},
                    **col, tb=sc['tb'], mA=sc['mA'], mH=sc['mH'], m1=sc['m1'], m2=sc['m2'], cba=sc['cba'], mu=sc['mu'], phi=sc['phi'], lam6=sc['lam6'])
    except Exception as ex: return dict(src=src, err=str(ex))
if __name__ == '__main__':
    J = []; S = pickle.load(open('/tmp/mfilt4w_NO.pkl', 'rb'))['S']
    for t in pickle.load(open('/tmp/lmax_final2.pkl', 'rb')):
        if t[2] is not None: J.append((('sample', t[0]), S[t[0]][1], t[6]))
    for d in ('/tmp/envres/out', '/tmp/envres2/out2'):
        for f in glob.glob(d + '/*.json'):
            b = json.load(open(f))
            if 'best' in b: J.append((('env', f), b['best']['x'], b['best']['eta']))
    import os; D = sys.argv[2] + '_parts'; os.makedirs(D, exist_ok=True); CH = 40   # resumable in chunks
    with Pool(int(sys.argv[1])) as P:
        for k in range(0, len(J), CH):
            fn = f'{D}/{k:05d}.pkl'
            if os.path.exists(fn): continue
            pickle.dump(P.map(one, J[k:k + CH], chunksize=2), open(fn, 'wb'))
    R = sum((pickle.load(open(f'{D}/{k:05d}.pkl', 'rb')) for k in range(0, len(J), CH)), [])
    pickle.dump(R, open(sys.argv[2], 'wb')); print('done', len(R), sum('err' in r for r in R))
