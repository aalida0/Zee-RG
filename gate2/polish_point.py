"""Local chi^2 polish of one stored parameter vector inside a trust box (V13 referee P0: the corrected oblique-T statistic moved the accepted region).
The box is +-FRAC of the refit scale XS (gate2/refit_hw.py) around the stored vector in the same variables (asinh Yukawas, ln tan(beta), log masses, quartics, eta); it minimises the
production chi^2 only, so the polished vector is a NEW parameter vector: every gate (accept.py) and the RG/matching validation must be rerun on it.
Usage: ZEE_INPUTS=modern [same ZEE_* variables as the reference] python3 -W ignore gate2/polish_point.py INPUT.json KEY CMIN OUT.json [FRAC ...]
(INPUT.json: results/prod1/benchmarks_input_prod1.json or results/benchmarks_input.json; KEY e.g. BP3)"""
import sys, os, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
import refit_hw as RF
from zee import hgmodel as H
from scipy.optimize import least_squares

if __name__ == '__main__':
    inp, key, cmin, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    fracs = [float(a) for a in sys.argv[5:]] or [0.01, 0.02, 0.05, 0.1]
    d = json.load(open(inp))[key]; x0 = np.array(d['x'], float); e0 = np.array(d['eta'], float)
    z0 = np.clip(RF.tz(x0), RF.LO, RF.HI); w0 = e0[[1, 2, 4]]; p0 = np.concatenate([z0, w0]); sc = np.concatenate([RF.XS, [0.5] * 3])
    def eta_of(w): e = e0.copy(); e[[1, 2, 4]] = w; return e      # the stored eta_1, eta_4, eta_6 are kept (the refit parameterisation would set them to 0)
    chi = lambda p: float(H.evaluate(list(RF.tx(p[:19])), 'NO', eta=eta_of(p[19:]))['chi2'])
    print('start chi2 %.4f  limit %.4f' % (chi(p0), cmin + 3.84), flush=True); best = None; recs = []
    for f in fracs:
        lo = np.maximum(np.concatenate([RF.LO, [-RF.EB] * 3]), p0 - f * sc); hi = np.minimum(np.concatenate([RF.HI, [RF.EB] * 3]), p0 + f * sc)
        r = least_squares(lambda p: H.evaluate(list(RF.tx(p[:19])), 'NO', eta=eta_of(p[19:]))['res'], p0, bounds=(lo, hi), max_nfev=300, x_scale=sc)
        c = chi(r.x); recs.append(dict(frac=f, chi2=c, x=RF.tx(r.x[:19]).tolist(), eta=eta_of(r.x[19:]).tolist(), max_shift_in_scale=float(np.max(np.abs(r.x - p0) / sc))))
        print('frac %.3f chi2 %.4f margin %+.4f max|shift|/scale %.3f' % (f, c, cmin + 3.84 - c, recs[-1]['max_shift_in_scale']), flush=True)
    json.dump(dict(key=key, cmin=cmin, start=dict(x=x0.tolist(), eta=e0.tolist(), chi2=chi(p0)), polished=recs, env={k: v for k, v in os.environ.items() if k.startswith('ZEE_')}), open(out, 'w'), indent=1)
