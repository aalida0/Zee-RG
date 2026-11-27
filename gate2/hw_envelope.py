"""Exact range of the CLFV rates over the overall multiplier s of the H+-W-+gamma Barr-Zee set, |s| <= 1 (report v11).
The amplitude vector is A + s B (A: everything else, B: the H+-W-+gamma set) and each rate is |A + s B|^2 = a + b s + c s^2 with c = |B|^2 >= 0, so the range over [-1, 1] is
[Br(s*), max(Br(-1), Br(1))] with s* = clip(-b/(2c), -1, 1) = clip(-Re(A^dagger B)/||B||^2, -1, 1); a, b, c follow from s = 0, +1, -1 (verified to be exactly quadratic with s = 0.5).
Usage: ZEE_INPUTS=modern python3 -W ignore gate2/hw_envelope.py [INPUT.json OUTPUT.json] -> results/hw_envelope.json"""
import os, sys, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path[:0] = [ROOT, ROOT + '/gate2']
os.environ.setdefault('ZEE_INPUTS', 'modern')
from zee import hgmodel as H, dipfull as DF
KEYS = ('mu_e_g', 'tau_mu_g', 'tau_e_g')


def rates(x, eta, s):
    DF.HW_SCALE = s
    try: o = H.evaluate(list(x), 'NO', eta=list(eta))['obs']
    finally: DF.HW_SCALE = 1.0
    return {k: float(o[k]) for k in KEYS}


def envelope(x, eta):
    r0, rp, rm = rates(x, eta, 0.0), rates(x, eta, 1.0), rates(x, eta, -1.0); rh = rates(x, eta, 0.5); out = {}
    for k in KEYS:
        a = r0[k]; b = 0.5*(rp[k] - rm[k]); c = 0.5*(rp[k] + rm[k]) - a
        ss = float(np.clip(-b/(2*c), -1, 1)) if c > 0 else (-1.0 if b > 0 else 1.0)
        out[k] = dict(s_min=ss, min=a + b*ss + c*ss**2, max=max(rp[k], rm[k]), nominal=rp[k], off=a, reversed=rm[k], quad_check=abs(a + 0.5*b + 0.25*c - rh[k])/max(rh[k], 1e-300))
    return out


if __name__ == '__main__':
    fin = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'results', 'benchmarks_input.json')               # optional: INPUT.json OUTPUT.json (e.g. results/corrected/benchmarks_input_corrected.json)
    fout = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'results', 'hw_envelope.json')
    inp = json.load(open(fin))
    res = {nm: envelope(np.array(d['x'], float), np.array(d['eta'], float)) for nm, d in inp.items()}
    json.dump(res, open(fout, 'w'), indent=1)
    for nm, r in res.items(): print(nm, {k: ('s*=%.2f min=%.2e max=%.2e nom=%.2e quad=%.1e' % (v['s_min'], v['min'], v['max'], v['nominal'], v['quad_check'])) for k, v in r.items()})
