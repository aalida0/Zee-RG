import sys, pickle, numpy as np
sys.path.insert(0, '/home/claude/zee_fit')
from scipy.optimize import least_squares
from zee import hgmodel as H, neutrino as N
ordering, seed, nstart, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
rng = np.random.default_rng(seed)
V = 246.21965
def mat(y):
    x = np.zeros(18); x[:11] = y[:11]
    f, Y = H.unpack(x); tb = np.exp(y[11]); sb = tb / np.sqrt(1 + tb**2)
    mE = np.diag(H.M_LEP)
    # eq. 2.30 with A = 1 (overall scale carried by kappa = exp(y[12]))
    t1 = f @ mE @ mE + mE @ mE @ f.T; t2 = f @ mE @ Y + Y.T @ mE @ f.T
    return np.exp(y[12]) * (t1 - V / (np.sqrt(2) * sb) * t2)
def res(y):
    o = N.oscillation_observables(mat(y), ordering); t = H.OSC[ordering]
    dd = (o["delta_deg"] - t["d"][0] + 180.) % 360. - 180.
    return np.array([(o["s12sq"]-t["s12"][0])/t["s12"][1], (o["s13sq"]-t["s13"][0])/t["s13"][1], (o["s23sq"]-t["s23"][0])/t["s23"][1],
                     (o["dm21_eV2"]-t["dm21"][0])/t["dm21"][1], (o["dm3l_eV2"]-t["dm3"][0])/t["dm3"][1], dd/t["d"][1]])
lo = np.array([-.1]*11 + [np.log(.3), -80.]); hi = np.array([.1]*11 + [np.log(50.), 0.])
best = []
for n in range(nstart):
    y = np.zeros(13)
    y[:3] = rng.choice([-1,1],3)*10**rng.uniform(-4,-1,3)
    for k in range(4):
        r = 10**rng.uniform(-4,-1); th = rng.uniform(0,2*np.pi); y[3+2*k], y[4+2*k] = r*np.cos(th), r*np.sin(th)
    y[11] = rng.uniform(np.log(.3), np.log(50.))
    M = mat(np.r_[y[:12], 0.0]); sv = np.linalg.svd(M, compute_uv=False)[0]
    y[12] = np.log(0.05e-9 / sv) if sv > 0 else 0.0
    y[12] = min(y[12], 0.0)
    try:
        r = least_squares(res, np.clip(y, lo, hi), bounds=(lo, hi), max_nfev=300, x_scale=np.maximum(hi-lo,1e-3)*0.1)
        best.append((float(np.sum(r.fun**2)), r.x))
    except Exception: pass
    if (n+1) % 500 == 0:
        best.sort(key=lambda t: t[0]); best = best[:50]
        pickle.dump(best, open(out, 'wb')); print(n+1, [round(c,2) for c,_ in best[:6]], flush=True)
best.sort(key=lambda t: t[0]); pickle.dump(best[:50], open(out, 'wb')); print("done", flush=True)
