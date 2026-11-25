"""Final NO sample with the corrected likelihood: thin the walk19 chains, apply HiggsTools, search for a stable choice of the
free quartics (lmax2: 100 random draws + 150 local steps, unitarity, BFB incl. lambda_h<0, global minimum), recheck HiggsTools
with that choice, hard cuts, fine-grid cutoff. Resumable in chunks. Usage: ZEE_INPUTS=modern python gate2/sample2.py NPT NPROC"""
import sys, os, glob, pickle, numpy as np
sys.path.insert(0, '/home/claude/zee_fit'); sys.path.insert(0, '/home/claude/zee_fit/gate1'); sys.path.insert(0, '/home/claude/zee_fit/gate2')
from multiprocessing import Pool
import lmax2 as L, m_filter_l4w as M, hcoll, screen
from zee import hgmodel as H
CMIN = pickle.load(open('/tmp/refit19.pkl', 'rb'))['NO'][0]; M.LIM = CMIN + 3.84; L.NN = (60, 80)
def one(a):
    i, x = a
    try:
        col, e = hcoll.run(x)
        base = dict(idx=i, x=np.asarray(x), chi2_base=float(e['chi2']), HS0=col['HS_chi2'], HB0=col['HB_allowed'])
        r = L.job((i, x))
        if r[2] is None: return dict(base, Lsearch=None)
        out = screen.one((('sample2', i), x, np.array(r[4]))); out.update(base, Lsearch=float(r[2]), gok80=bool(r[5])); return out
    except Exception as ex: return dict(idx=i, err=str(ex))
if __name__ == '__main__':
    W = sum((pickle.load(open(f, 'rb')) for f in sorted(glob.glob('/tmp/walk19/*.pkl'))), [])
    npt = int(sys.argv[1]); sel = np.linspace(0, len(W) - 1, npt).astype(int); J = [(int(k), W[k][1]) for k in sel]
    D = '/tmp/sample2_parts'; os.makedirs(D, exist_ok=True); CH = 10
    with Pool(int(sys.argv[2])) as P:
        ks = list(range(0, len(J), CH)); ks = ks[::2] + ks[1::2]        # even chunks first: spread over all chains
        for k in ks:
            fn = f'{D}/{k:05d}.pkl'
            if os.path.exists(fn): continue
            pickle.dump(P.map(one, J[k:k + CH], chunksize=1), open(fn, 'wb')); print('chunk', k, flush=True)
    print('done', len(os.listdir(D)))
