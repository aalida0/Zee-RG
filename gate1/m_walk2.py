import sys, pickle, numpy as np
sys.path.insert(0, '/home/claude/zee_fit')
from multiprocessing import Pool
from zee import hgmodel as H
ordering = sys.argv[1]; S0 = 1e-6
pts = pickle.load(open(f'/tmp/mbest_{ordering}.pkl', 'rb')); cmin = pts[0][0]; lim = cmin + 3.84
seeds = [(c, x) for c, x in pts if c <= lim]; seeds = seeds[::max(1, len(seeds) // 24)][:24]
LO, HI = H.LO.copy(), H.HI.copy(); ZM = np.arcsinh(0.1 / S0)
def chain(args):
    k, c0, x0 = args; rng = np.random.default_rng(100 + k)
    z = x0.copy(); z[:11] = np.arcsinh(x0[:11] / S0); lo = LO.copy(); hi = HI.copy(); lo[:11] = -ZM; hi[:11] = ZM
    span = hi - lo; span[:11] = 3.0
    step = 0.02; out = []; acc = n = 0
    for it in range(9000):
        y = np.clip(z + rng.normal(size=18) * span * step * (rng.random(18) < 0.3), lo, hi)
        x = y.copy(); x[:11] = S0 * np.sinh(y[:11])
        try: e = H.evaluate(x, ordering)
        except Exception: continue
        n += 1
        if e['pen'] == 0 and e['chi2'] <= lim: z = y; acc += 1
        if it % 100 == 99:
            step = float(np.clip(step * (1.3 if acc / max(n, 1) > 0.3 else 0.7), 1e-4, 0.3)); acc = n = 0
        if it >= 1500 and it % 15 == 0:
            xx = z.copy(); xx[:11] = S0 * np.sinh(z[:11]); out.append((float(H.evaluate(xx, ordering)['chi2']), xx))
    return out
if __name__ == '__main__':
    with Pool(2) as p: r = p.map(chain, [(k, c, x) for k, (c, x) in enumerate(seeds)])
    out = sum(r, []); pickle.dump(out, open(f'/tmp/msamples_{ordering}.pkl', 'wb')); print('done', len(out), len(seeds), flush=True)
