"""Recompute the cutoff of every stored witness on a fine scale grid (10 per decade): sample (lmax_final2.pkl) and
verified envelope points. Usage: ZEE_INPUTS=modern python gate1/fine_lmax.py ENV_DF.pkl OUT.pkl"""
import sys, json, pickle, numpy as np
_R = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, _R + '/gate1')
from multiprocessing import Pool
import lmax2 as L
from m_filter_l4w import H
L.SC = np.minimum(np.log(10.0**np.linspace(3, 16, 131)), np.log(1e16))
def lv(a):
    key, x, eta = a
    try:
        sc = H.evaluate(list(x), 'NO')['sc']; c = np.zeros(11)
        c[0], c[2], c[3], c[4], c[5] = sc['lam1'], sc['lam3'], sc['lam4'], sc['lam5'], sc['lam6']; c[1], c[6:] = eta[0], eta[1:]
        return key, float(L.lam_valid(c, np.log(min(sc['m1'], sc['mA'], sc['mH'])), -sc['tb']))
    except Exception: return key, -9.0
if __name__ == '__main__':
    S = pickle.load(open('/tmp/mfilt4w_NO.pkl', 'rb'))['S']; jobs = []
    for t in pickle.load(open('/tmp/lmax_final2.pkl', 'rb')):
        if t[2] is not None and t[-1]: jobs.append((('s', t[0]), S[t[0]][1], np.array(t[6])))
    df = pickle.load(open(sys.argv[1], 'rb'))
    for _, r in df[df.chiok & df.hbok & (df.dhs <= 3.84) & df.gok].iterrows():
        b = json.load(open(r.f))['best']; jobs.append((('e', r.f), np.array(b['x']), np.array(b['eta'])))
    with Pool(2) as P: R = P.map(lv, jobs, chunksize=4)
    pickle.dump(dict(R), open(sys.argv[2], 'wb')); print('done', len(R))
