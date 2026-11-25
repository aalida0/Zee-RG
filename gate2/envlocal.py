"""Targeted local search (corrected likelihood): maximise observable OBS among points with a quartic choice stable to 10^LOGL.
Starts from the best validated point with cutoff >= LOGL; a step is accepted if chi2 <= min+3.84, all hard limits hold, OBS increases,
and the current quartic choice (or a new one from a short search) stays stable to 10^LOGL; HiggsTools is checked on acceptance.
Usage: ZEE_INPUTS=modern python gate2/envlocal.py OBS LOGL SEED WALL OUTDIR"""
import sys, os, json, time, pickle, numpy as np
_R = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, _R + '/gate1'); sys.path.insert(0, _R + '/gate2')
import lmax2 as L, m_filter_l4w as M, hcoll, accept as AC
from zee import hgmodel as H, vac
obs, logL, seed, wall, out = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMIN = pickle.load(open(os.path.join(ROOT, 'results', 'refit_corrected.pkl'), 'rb'))['NO'][0]; LIM = CMIN + 3.84; M.LIM = LIM; HSMIN = 151.65
L.SC = np.minimum(np.log(10.0**np.linspace(3, 16, 131)), np.log(1e16))   # fine grid
rng = np.random.default_rng(31 + 97*seed); S0 = 1e-6; ZM = np.arcsinh(0.1/S0); SQ = np.sqrt(4*np.pi)
LO = np.append(H.LO, -SQ); HI = np.append(H.HI, SQ); LO[:11] = -ZM; HI[:11] = ZM; span = HI - LO; span[:11] = 3.0
tx = lambda z: np.concatenate([S0*np.sinh(z[:11]), z[11:]]); tz = lambda x: np.concatenate([np.arcsinh(np.asarray(x[:11], float)/S0), np.asarray(x[11:19], float)])
key = 'BR_h_tau_mu' if obs == 'h_tau_mu' else obs
import pandas as pd
A = pickle.load(open('/tmp/allvalid.pkl', 'rb'))
if os.path.exists('/tmp/loc_valid.pkl'):
    V = pd.DataFrame([r for r in pickle.load(open('/tmp/loc_valid.pkl', 'rb')) if 'err' not in r]); V = V[(V.chi2 <= LIM) & (V.HS_chi2 - HSMIN <= 3.84) & V.HB_allowed & V.hard & V.glob]; A = pd.concat([A, V])
A = A[A.Lf >= logL].sort_values(key, ascending=False)
if not len(A): sys.exit('no start')
st = A.iloc[seed % min(len(A), 4)]; x = np.asarray(st.x, float); eta = np.asarray(st.eta, float); z = tz(x)
def val(e, col=None): return (col['BR_h_tau_mu'] if col else e['obs']['h_tau_mu']) if obs == 'h_tau_mu' else e['obs'][obs]
cur = float(st[key]); best = dict(val=cur, x=x.tolist(), eta=eta.tolist()); t0 = time.time(); step = 0.01; n = acc = 0
while time.time() - t0 < wall:
    y = np.clip(z + rng.normal(size=19)*span*step*(rng.random(19) < 0.3), LO, HI); xn = tx(y); n += 1
    try:
        e = H.evaluate(list(xn), 'NO', eta=eta)        # cheap pre-check with the same statistic
        if e['pen'] != 0 or e['chi2'] > LIM or val(e) <= cur: step *= 0.995; continue
        r = AC.accept(xn, eta, CMIN, logL=logL, n_vac=20, seed=int(rng.integers(1 << 30)), stop_early=True)
        if not r['accepted']: continue
        v = val(e, r['collider'])
        if v <= cur: continue
    except Exception: continue
    z = y; cur = v; acc += 1; step = min(step*1.2, 0.1); best = dict(val=cur, x=xn.tolist(), eta=eta.tolist())
    json.dump(dict(obs=obs, logL=logL, seed=seed, n=n, acc=acc, best=best), open(f'{out}/loc_{obs}_{logL:g}_{seed}.json', 'w'))
json.dump(dict(obs=obs, logL=logL, seed=seed, n=n, acc=acc, best=best, done=True), open(f'{out}/loc_{obs}_{logL:g}_{seed}.json', 'w'))
print(obs, logL, seed, 'start %.2e -> %.2e' % (float(st[key]), cur), 'acc', acc, 'n', n)
