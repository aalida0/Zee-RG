import sys, os, pickle, numpy as np
_R = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
sys.path.insert(0, _R); os.environ['ZEE_INPUTS'] = 'modern'
import dynesty
from dynesty import NestedSampler
from multiprocessing import Pool
from zee import hgmodel as H
O = sys.argv[1]; NL = int(sys.argv[2]); CK = f'/tmp/ns_{O}.save'
LO, HI = H.LO, H.HI
def prior(u):
    x = np.empty(18)
    v = 2 * u[:11] - 1; x[:11] = np.sign(v) * 10 ** (-12 + 11 * np.abs(v))
    x[11] = LO[11] + u[11] * (HI[11] - LO[11])
    x[12:] = LO[12:] + u[12:] * (HI[12:] - LO[12:])
    return x
def logl(x):
    try: return -0.5 * H.evaluate(x, O)['chi2']
    except Exception: return -1e6
if __name__ == '__main__':
    with Pool(2) as p:
        if os.path.exists(CK): s = NestedSampler.restore(CK, pool=p)
        else: s = NestedSampler(logl, prior, 18, nlive=NL, sample='rslice', slices=8, pool=p, queue_size=2)
        s.run_nested(dlogz=float(sys.argv[3]), checkpoint_file=CK, checkpoint_every=120, resume=os.path.exists(CK), print_progress=False, maxcall=int(float(sys.argv[4])))
    r = s.results; pickle.dump(dict(samples=r.samples, logwt=r.logwt, logl=r.logl, logz=r.logz[-1]), open(f'/tmp/nsres_{O}.pkl', 'wb')); print('done', r.niter, r.logz[-1], flush=True)
