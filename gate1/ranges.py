"""Profile-like range search: extremise an observable inside the region chi2 <= chi2_min + dchi."""
import sys, pickle, time, numpy as np
sys.path.insert(0, '/home/claude/zee_fit')
from zee import hgmodel as H

OBS = ["tau_mu_g", "tau_e_g", "mu_e_g", "Cr_Au", "h_tau_mu", "h_tau_e", "m1", "mA", "mH", "sba"]


def getobs(e, k):
    if k in e['obs']:
        return np.log10(max(e['obs'][k], 1e-45))
    v = e['sc'][k]
    return np.log10(v) if k in ("m1", "mA", "mH") else v


def walk(x0, ordering, key, sign, lim, rng, nsteps=15000):
    x = x0.copy(); e = H.evaluate(x, ordering); g = sign * getobs(e, key)
    best = (g, x.copy()); sig = 0.05
    scale = np.maximum(np.abs(x), 1e-4)
    for n in range(nsteps):
        T = 0.3 * (1 - n / nsteps)
        y = x.copy()
        k = rng.integers(1, 5); idx = rng.choice(18, k, replace=False)
        s = 10 ** rng.uniform(-3, -0.5)
        for i in idx:
            if i < 11:
                y[i] += s * max(abs(x[i]), 1e-4) * rng.normal()
            else:
                y[i] += s * 0.5 * (H.HI[i] - H.LO[i]) * rng.normal()
        y = np.clip(y, H.LO, H.HI)
        try:
            ey = H.evaluate(y, ordering)
        except Exception:
            continue
        if ey['chi2'] > lim:
            continue
        gy = sign * getobs(ey, key)
        if gy <= g or rng.random() < np.exp(-(gy - g) / max(T, 1e-3)):
            x, g = y, gy
            if g < best[0]:
                best = (g, x.copy())
    return best


if __name__ == "__main__":
    ordering, pk, dchi, seed, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    pts = []
    for f in pk.split(','):
        pts += pickle.load(open(f, 'rb'))
    pts.sort(key=lambda t: t[0])
    cmin = pts[0][0]; lim = cmin + dchi
    starts = [x for c, x in pts if c <= lim][:6]
    print("chi2min", cmin, "n starts in region", len(starts), flush=True)
    rng = np.random.default_rng(seed)
    result = {}
    for key in OBS:
        lo = hi = None
        for sign in (+1, -1):
            vals = []
            for x0 in starts[:3]:
                g, x = walk(x0, ordering, key, sign, lim, rng)
                vals.append(sign * g)
            result[(key, sign)] = (min(vals) if sign > 0 else max(vals))
        print(key, "min %.3g max %.3g (log10 or value)" % (result[(key, 1)], result[(key, -1)]), flush=True)
        pickle.dump(dict(result=result, cmin=cmin), open(out, 'wb'))
