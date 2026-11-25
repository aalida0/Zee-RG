import sys, pickle, numpy as np
_R = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
sys.path.insert(0, _R)
from scipy.optimize import least_squares
from zee import hgmodel as H
o = sys.argv[1]; S0 = 1e-6; ZM = np.arcsinh(0.1 / S0)
LO, HI = H.LO.copy(), H.HI.copy(); LO[:11] = -ZM; HI[:11] = ZM
xs = np.maximum(HI - LO, 1e-3) * 0.1; xs[:11] = 1.0
def tx(z): x = z.copy(); x[:11] = S0 * np.sinh(z[:11]); return x
fun = lambda z: H.evaluate(tx(z), o)['res']
seeds = [x for _, x in pickle.load(open(f'/tmp/best_{o}.pkl', 'rb'))[:40]]
S = pickle.load(open(f'/tmp/samples_{o}.pkl', 'rb')); seeds += [S[i] if not isinstance(S[i], tuple) else S[i][-1] for i in range(0, len(S), max(1, len(S)//150))]
out = []
for x in seeds:
    z = np.array(x, float); z[:11] = np.arcsinh(np.clip(z[:11], -.1, .1) / S0)
    try:
        r = least_squares(fun, np.clip(z, LO, HI), bounds=(LO, HI), max_nfev=300, x_scale=xs)
        out.append((float(H.evaluate(tx(r.x), o)['chi2']), tx(r.x)))
    except Exception as e: pass
out.sort(key=lambda t: t[0]); pickle.dump(out, open(f'/tmp/mbest2_{o}.pkl', 'wb')); print(o, [round(c, 2) for c, _ in out[:6]], len(out))
