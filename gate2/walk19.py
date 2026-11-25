"""Random-walk sampling of the NO region (Delta chi2 <= 3.84, 19 parameters, lambda4 free) with the corrected likelihood.
Resumable: one file per chain. Usage: ZEE_INPUTS=modern python gate2/walk19.py NCHAIN NIT NPROC"""
import sys, os, pickle, numpy as np
_R = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
sys.path.insert(0, _R)
from multiprocessing import Pool
from zee import hgmodel as H
S0 = 1e-6; ZM = np.arcsinh(0.1/S0); SQ = np.sqrt(4*np.pi)
LO = np.append(H.LO, -SQ); HI = np.append(H.HI, SQ); LO[:11] = -ZM; HI[:11] = ZM; span = HI - LO; span[:11] = 3.0
tx = lambda z: np.concatenate([S0*np.sinh(z[:11]), z[11:]]); tz = lambda x: np.concatenate([np.arcsinh(np.asarray(x[:11], float)/S0), np.asarray(x[11:19], float)])
CMIN = pickle.load(open('/tmp/refit19.pkl', 'rb'))['NO'][0]; LIM = CMIN + 3.84; D = '/tmp/walk19'
def chain(a):
    k, x0, nit = a; fn = f'{D}/{k:03d}.pkl'
    if os.path.exists(fn): return
    rng = np.random.default_rng(700 + k); z = np.clip(tz(x0), LO, HI); step = 0.02; out = []; acc = n = 0
    for it in range(nit):
        y = np.clip(z + rng.normal(size=19)*span*step*(rng.random(19) < 0.3), LO, HI)
        try: e = H.evaluate(tx(y), 'NO')
        except Exception: continue
        n += 1
        if e['pen'] == 0 and e['chi2'] <= LIM: z = y; acc += 1
        if it % 100 == 99: step = float(np.clip(step*(1.3 if acc/max(n, 1) > .3 else .7), 1e-4, .3)); acc = n = 0
        if it >= nit//6 and it % 15 == 0: x = tx(z); out.append((float(H.evaluate(x, 'NO')['chi2']), x))
    pickle.dump(out, open(fn, 'wb'))
if __name__ == '__main__':
    os.makedirs(D, exist_ok=True); nc, nit = int(sys.argv[1]), int(sys.argv[2])
    df, cmin, hsmin = pickle.load(open('/tmp/screen_df.pkl', 'rb'))
    seeds = [pickle.load(open('/tmp/refit19.pkl', 'rb'))['NO'][1]]
    ok = df[(df.chi2 <= LIM)].sort_values('chi2')
    seeds += [x for x in ok.x][:nc - 1]
    with Pool(int(sys.argv[3])) as P: P.map(chain, [(k, seeds[k % len(seeds)], nit) for k in range(nc)], chunksize=1)
    print('chains', len(os.listdir(D)))
