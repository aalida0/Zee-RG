"""Effect of the second-doublet top Yukawa rho_t = -yt tan(beta) (Phi2 free of quark couplings in the generic basis, HG Sec. 2.5)
on the stable-to-Lambda fractions. Usage: ZEE_INPUTS=modern python rg_rho.py STRIDE NPROC"""
import sys, pickle, numpy as np
_R = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, _R + '/gate1')
from multiprocessing import Pool
from zee import rgfast, rgfast2
import m_filter_l4w as M
from m_filter_l4w import H
R_ = 0.0; _TB = 0.0
def run_r(c0, t0, t1=np.log(1e19), **kw):
    kw.pop('g', None); kw.pop('yt', None)
    return rgfast2.run(c0, t0, t1=t1, r=-_TB*R_, **kw)
rgfast.run = run_r
def job(a):
    global R_, _TB
    R_, (i, x) = a; _TB = H.scalar_sector(x)['tb']
    return R_, _TB, M.one((i, x))[1]
if __name__ == '__main__':
    stride, npr = int(sys.argv[1]), int(sys.argv[2])
    S = pickle.load(open('/tmp/lam4wide_NO.pkl', 'rb')); best = pickle.load(open('/tmp/mbest_NO.pkl', 'rb'))[0][0]; M.LIM = best + 3.84
    pts = [(i, S[i][1]) for i in range(0, len(S), stride) if S[i][0] < best + 3.84]
    jobs = [(r, p) for p in pts for r in (0.0, 1.0)]
    with Pool(npr) as P: out = P.map(job, jobs, chunksize=1)
    pickle.dump(out, open('/tmp/rgrho.pkl', 'wb'))
    for r in (0.0, 1.0):
        o = [x[2] for x in out if x[0] == r]
        print('rho/yt=-tanb*', r, len(o), {L: round(np.mean([a[L] for a in o]), 3) for L in (1e4, 1e8, 1e16)}, 'ew', round(np.mean([a['ew'] for a in o]), 3))
