"""Dedicated search for the largest rate of one observable among points that admit a quartic sector stable up to Lambda_min.
Walk in the 19 model parameters; a step is accepted only if chi2 <= LIM, the HiggsTools filter passes, the observable increases,
and a witness of the free quartics (eta) with Lambda_max >= Lambda_min is found.
Usage: ZEE_INPUTS=modern python envopt.py OBS LOGL SEED WALL_SECONDS OUTDIR
  OBS in {tau_mu_g, mu_e_g, Cr_Al, h_tau_mu};  LOGL = log10(Lambda_min/GeV);  data pickles in $ZEE_DATA (default /tmp)."""
import sys, os, json, time, pickle, numpy as np
D = os.environ.get('ZEE_DATA', '/tmp'); R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R); sys.path.insert(0, os.path.join(R, 'gate1'))
import lmax2 as L, m_filter_l4w as M, ht
from zee import vac
from m_filter_l4w import H
obs, logL, seed, wall, out = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
rng = np.random.default_rng(777 + 1000 * seed)
d = pickle.load(open(D + '/mfilt4w_NO.pkl', 'rb')); S = d['S']; lm = {t[0]: t for t in pickle.load(open(D + '/lmax_final2.pkl', 'rb'))}
best = pickle.load(open(os.path.join(R, 'results', 'refit_corrected.pkl'), 'rb'))['NO'][0]; M.LIM = best + 3.84
HSMIN = min(t[3] for t in lm.values()); S0 = 1e-6
lo = np.append(H.LO, -np.sqrt(4*np.pi)); hi = np.append(H.HI, np.sqrt(4*np.pi)); ZM = np.arcsinh(0.1/S0); lo[:11] = -ZM; hi[:11] = ZM
span = hi - lo; span[:11] = 3.0
def to_z(x): z = np.array(x, float); z[:11] = np.arcsinh(z[:11]/S0); return z
def to_x(z): x = z.copy(); x[:11] = S0*np.sinh(z[:11]); return x
def lam_test(x, target, n0=60, nit=120):
    """search for eta with Lambda_max >= target; returns (log10 Lambda reached, eta or None)"""
    e = H.evaluate(list(x), 'NO'); sc = e['sc']; t0 = np.log(min(sc['m1'], sc['mA'], sc['mH'])); r = -sc['tb']
    mu_gg = e['mus']['gg']; g0 = ((mu_gg - 1.14)/0.185)**2; rgg0 = e['rgg']
    def build(eta):
        c = np.zeros(11); c[0], c[2], c[3], c[4], c[5] = sc['lam1'], sc['lam3'], sc['lam4'], sc['lam5'], sc['lam6']
        c[1], c[6], c[7], c[8], c[9], c[10] = eta; return c
    ok = lambda c: H.evaluate(list(x), 'NO', eta=[c[1], c[6], c[7], c[8], c[9], c[10]])['chi2'] <= M.LIM   # same statistic as gate2/accept.py
    elo = np.array([0.02, -1.5, -1.5, -1.5, -1.5, 0.0]); ehi = np.array([3.5449, 1.5, 1.5, 1.5, 1.5, 3.0])
    draw = lambda: np.array([np.exp(rng.uniform(np.log(.02), np.log(3.5449))), rng.uniform(-1.5, 1.5), *(rng.uniform(-1.5, 1.5, 3)*(rng.random(3) < .6)), rng.uniform(0, 3)])
    pool = []
    for k in range(n0):
        eta = draw(); c = build(eta)
        if ok(c):
            v = L.lam_valid(c, t0, r)
            if v >= target - 1e-9 and vac.global_ok(c, sc, nstart=40, rng=rng)[0]: return v, eta
            pool.append((v, eta))
    if not pool: return -1.0, None
    pool.sort(key=lambda t: -t[0]); cur, eta = pool[0]; step = 0.3*(ehi - elo)
    for it in range(nit):
        cand = np.clip(eta + step*rng.normal(size=6)*(rng.random(6) < .5), elo, ehi); c = build(cand)
        if not ok(c): continue
        v = L.lam_valid(c, t0, r)
        if v >= target - 1e-9 and vac.global_ok(c, sc, nstart=40, rng=rng)[0]: return v, cand
        if v >= cur: cur, eta = v, cand
        else: step *= 0.97
    return cur, None
def val(x): return H.evaluate(list(x), 'NO')['obs'][obs]
# starting points: kept sample points with Lambda_max >= logL, ordered by the observable
cand = [(i, S[i][1]) for i in lm if lm[i][4] and lm[i][2] is not None and lm[i][2] >= logL - 1e-9]
if not cand: cand = sorted([(i, S[i][1]) for i in lm if lm[i][4] and lm[i][2] is not None], key=lambda t: -lm[t[0]][2])[:5]
cand.sort(key=lambda t: -val(t[1])); k0 = seed % len(cand)
t_start = time.time(); log = []
x = None
ST = pickle.load(open(D + '/env_starts.pkl', 'rb')).get((obs, logL), []) if os.path.exists(D + '/env_starts.pkl') and os.environ.get('ZEE_RESTART') else []
if ST:   # second round: restart from a verified best point of the previous round (stored witness eta)
    v0, xs, eta0 = ST[seed % len(ST)]; x = np.array(xs, float); z = to_z(x); cur = val(x); eta0 = np.array(eta0)
    e0 = H.evaluate(list(x), 'NO')['sc']; c0 = np.zeros(11); c0[0], c0[2], c0[3], c0[4], c0[5] = e0['lam1'], e0['lam3'], e0['lam4'], e0['lam5'], e0['lam6']; c0[1], c0[6:] = eta0[0], eta0[1:]
    v = L.lam_valid(c0, np.log(min(e0['m1'], e0['mA'], e0['mH'])), -e0['tb'])
for j in range(len(cand) if x is None else 0):
    i, xs = cand[(k0 + j) % len(cand)]; v, eta = lam_test(xs, logL, 100, 150)
    if eta is not None: x = np.array(xs, float); z = to_z(x); cur = val(x); eta0 = eta; break
if x is None: json.dump({'obs': obs, 'logL': logL, 'seed': seed, 'failed': 'no start'}, open(f'{out}/env_{obs}_{logL:g}_{seed}.json', 'w')); sys.exit()
bestrec = dict(val=cur, x=x.tolist(), eta=list(map(float, eta0)), logLmax=float(v)); n = nacc = 0; step = 0.03
while time.time() - t_start < wall:
    y = np.clip(z + rng.normal(size=19)*span*step*(rng.random(19) < 0.3), lo, hi); xn = to_x(y); n += 1
    try:
        e = H.evaluate(list(xn), 'NO')
        if e['pen'] != 0 or e['chi2'] > M.LIM or e['obs'][obs] <= cur: continue
        chi, okb, _ = ht.run(list(xn))
        if not okb or chi - HSMIN > 3.84: continue
        v, eta = lam_test(xn, logL)
    except Exception: continue
    if eta is None: continue
    z = y; cur = e['obs'][obs]; nacc += 1; bestrec = dict(val=cur, x=xn.tolist(), eta=list(map(float, eta)), logLmax=float(v), chi2=float(e['chi2']), hs=float(chi))
    if True: json.dump(dict(obs=obs, logL=logL, seed=seed, n=n, nacc=nacc, best=bestrec), open(f'{out}/env_{obs}_{logL:g}_{seed}.json', 'w'))
    if nacc % 20 == 0: step = float(np.clip(step*(1.2 if nacc/n > .05 else 0.8), 3e-3, .2))
json.dump(dict(obs=obs, logL=logL, seed=seed, n=n, nacc=nacc, best=bestrec, done=True), open(f'{out}/env_{obs}_{logL:g}_{seed}.json', 'w'))
print('done', obs, logL, seed, n, nacc, bestrec['val'])
