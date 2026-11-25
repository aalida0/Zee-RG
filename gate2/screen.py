"""Screen every stored witness (sample + dedicated searches) with the corrected code: continuous likelihood, modern EDM bound,
lambda_h<0 boundedness, same-Lagrangian HiggsTools (gate2/hcoll.py), fine-grid cutoff (10/decade) and global minimum.
Usage: ZEE_INPUTS=modern python gate2/screen.py NPROC OUT.pkl"""
import sys, glob, json, pickle, numpy as np
_R = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, _R + '/gate1'); sys.path.insert(0, _R + '/gate2')
from multiprocessing import Pool
import lmax2 as L, m_filter_l4w as M, hcoll, accept as AC
from zee import vac
from m_filter_l4w import H
L.SC = np.minimum(np.log(10.0**np.linspace(3, 16, 131)), np.log(1e16))
KEYS = ['tau_mu_g', 'tau_e_g', 'mu_e_g', 'tau_3mu', 'tau_mumue', 'Cr_Au', 'Cr_Al', 'mu_3e', 'h_tau_mu', 'h_tau_e', 'd_e', 'd_mu', 'da_e', 'das_mu', 'm_ee', 'm_beta', 'sum_m']
CMIN = None   # set by the caller (laboratory minimum, results/refit_corrected.pkl)
def one(a):
    """shared acceptance gate (gate2/accept.py) with a cutoff target 10^3 for the record; returns a flat dict"""
    src, x, eta = a
    try:
        r = AC.accept(x, eta, CMIN if CMIN is not None else np.inf, logL=3.0); sc = r['sc']; o = r['obs']; col = r['collider']
        return dict(src=src, x=np.asarray(x, float), eta=np.asarray(eta, float), chi2=r['chi2'], accepted=r['accepted'],
                    hard=all(v for k, v in r['cuts'].items() if k.startswith('lim_')), Lf=r['Lf'], Lf_status=r['Lf_status'],
                    glob=r['vacuum']['status'] == 'ok', cuts=r['cuts'], **{k: float(o[k]) for k in KEYS}, **col,
                    tb=sc['tb'], mA=sc['mA'], mH=sc['mH'], m1=sc['m1'], m2=sc['m2'], cba=sc['cba'], mu=sc['mu'], phi=sc['phi'], lam6=sc['lam6'])
    except Exception as ex: return dict(src=src, err=repr(ex))
if __name__ == '__main__':
    import os; CMIN = pickle.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'refit_corrected.pkl'), 'rb'))['NO'][0]
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
