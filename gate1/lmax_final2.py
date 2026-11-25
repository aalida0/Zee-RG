"""Lambda_max (optimised and random-stage) for every point of the final NO sample (mfilt4w), with the HiggsTools keep flags.
Usage: ZEE_INPUTS=modern python lmax_final.py NPROC"""
import sys, pickle, numpy as np
_R = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, _R + '/gate1')
from multiprocessing import Pool
import lmax2 as L, ht
def job(a):
    i, x = a
    r = L.job((i, x)); Lr, Lo = r[1], r[2]; ex = r[3:] if len(r) > 3 else (None, None, None); chi, ok, _ = ht.run(list(x))
    return i, Lr, Lo, chi, ok, *ex
if __name__ == '__main__':
    d = pickle.load(open('/tmp/mfilt4w_NO.pkl', 'rb')); S = d['S']
    best = pickle.load(open('/tmp/mbest_NO.pkl', 'rb'))[0][0]; L.M.LIM = best + 3.84
    pts = [(i, S[i][1]) for i, r in d['res']]
    with Pool(int(sys.argv[1])) as P: out = P.map(job, pts, chunksize=2)
    pickle.dump(out, open('/tmp/lmax_final2.pkl', 'wb')); print('done', len(out))
