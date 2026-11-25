import sys, pickle, numpy as np
_R = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
sys.path.insert(0, _R)
from zee import hgmodel as H
pts = pickle.load(open('/tmp/s1_NO.pkl', 'rb')); cmin = pts[0][0]; lim = cmin + 3.84
seeds = [(c, x) for c, x in pts if c <= lim]
rng = np.random.default_rng(11); span = H.HI - H.LO
out = []
for k, (c, x) in enumerate(seeds):
    cur, cx = c, x.copy(); step = 0.01
    acc = 0; n = 0
    for it in range(6000):
        y = np.clip(cx + rng.normal(size=18) * span * step * (rng.random(18) < 0.3), H.LO, H.HI)
        try: e = H.evaluate(y, 'NO')
        except Exception: continue
        n += 1
        if e['pen'] == 0 and e['chi2'] <= lim:
            cx, cur = y, e['chi2']; acc += 1
        if it % 100 == 99:
            step = float(np.clip(step * (1.3 if acc / max(n, 1) > 0.3 else 0.7), 1e-4, 0.1)); acc = n = 0
        if it >= 1000 and it % 25 == 0:
            out.append((cur, cx.copy()))
    print(k, len(out), flush=True)
pickle.dump(out, open('/tmp/final_samples.pkl', 'wb'))
print('done', len(out))
