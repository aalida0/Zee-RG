"""Profile of the production chi2 in tan(beta) with PHYSICAL starts: M_nu depends on (f*tan(beta), Y2/sin(beta)) only, so a point at
tan(beta)_0 is carried to tan(beta) by f -> f tb0/tb, Y2 -> Y2 sin(beta)/sin(beta0) (oscillation chi2 unchanged); all parameters are then refitted
with ln tan(beta) fixed (continuation in tan(beta), previous solution as next start).  Usage: ZEE_INPUTS=modern python3 -W ignore gate2/profile_tb.py
[TB list] -> results/profile_tb.json"""
import sys, os, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
os.environ.setdefault('ZEE_INPUTS', 'modern')
from scipy.optimize import least_squares
from zee import hgmodel as H
import refit_hw as R
sb = lambda t: t / np.sqrt(1 + t * t)
GRID = [float(g) for g in (sys.argv[1] if len(sys.argv) > 1 else '0.38,0.45,0.5,0.6,0.7,0.85,1.0,1.5,2.0').split(',')]


def fit(x1, w1, tbv):
    lo = np.concatenate([R.LO, [-3] * 3]); hi = np.concatenate([R.HI, [3] * 3]); lo[11] = np.log(tbv) - 1e-9; hi[11] = np.log(tbv) + 1e-9
    z0 = R.tz(x1); z0[11] = np.log(tbv); z0 = np.clip(z0, lo[:19], hi[:19])
    f = lambda p: H.evaluate(list(R.tx(p[:19])), 'NO', eta=R.eta_of(p[19:]))['res']
    r = least_squares(f, np.concatenate([z0, w1]), bounds=(lo, hi), max_nfev=400, x_scale=np.concatenate([R.XS, [0.5] * 3]))
    x = R.tx(r.x[:19]); eta = R.eta_of(r.x[19:]); e = H.evaluate(list(x), 'NO', eta=eta)
    return x, eta, e


if __name__ == '__main__':
    b = json.load(open(os.path.join(ROOT, 'results', 'refit_hw.json')))['best']; x, w, tb0, out = np.array(b['x'], float), np.array(b['eta'])[[1, 2, 4]], np.exp(b['x'][11]), []
    for tbv in GRID:
        x1 = x.copy(); x1[:3] *= tb0 / tbv; x1[3:11] *= sb(tbv) / sb(tb0)
        x, eta, e = fit(x1, w, tbv); w = eta[[1, 2, 4]]; tb0 = tbv; o = e['obs']
        parts = dict(osc=e['chi_osc'], univ=e['chi_univ'], stu=e['chi_stu'], higgs=e['chi_higgs'], **{k: float(v) for k, v in e['chi_bounds'].items() if v > 0.3})
        print('tanb %5.2f chi2 %8.2f  tau_mu_g %.1e mu_e_g %.1e h_tau_mu %.1e  %s' % (tbv, e['chi2'], o['tau_mu_g'], o['mu_e_g'], o['h_tau_mu'], {k: round(v, 1) for k, v in parts.items() if v > 0.9 and k not in ('univ', 'stu', 'higgs')}), flush=True)
        out.append(dict(tb=tbv, chi2=float(e['chi2']), x=x.tolist(), eta=eta.tolist(), parts=parts))
        json.dump(out, open(os.path.join(ROOT, 'results', 'profile_tb.json'), 'w'), indent=1)
