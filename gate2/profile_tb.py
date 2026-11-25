"""Targeted profile of the production chi2 in tan(beta) (NOT a global profile and not an exclusion: every value is an upper bound on the true
profiled minimum at that tan(beta), reached from the given start).  Leading M_nu depends on f*tan(beta) and Y2/sin(beta) only, so a point is
carried to a new tan(beta) with hgmodel.rescale_fixed_nu (full neutrino matrix invariant, test tests/test_profile_rescale.py) and all other
parameters are refitted with ln tan(beta) REMOVED from the optimiser's variables.  Continuation runs upward and downward from the start's own
tan(beta); the optimiser status, nfev, cost and optimality are recorded; points with chi2 <= chi2_min + 3.84 are passed through the low-energy
gate (campaign.stage_gate without the rescaled-boundary variants).
Starts (label): campaign_starts.json labels (BM1, BM2, start203, start229, start286), 'min' (refit_hw best), 'minK' (K-th distinct local minimum
of refit_hw 'all'), and 'heavyN' variants of 'min' (heavy scalar masses x N in M_H+ and the singlet mass, alternative Yukawa normalisation).
Usage: ZEE_INPUTS=modern [DIRECT=1] python3 -W ignore gate2/profile_tb.py START [TB list] -> results/profile_tb/START[_tb...].json"""
import sys, os, json, time, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
os.environ.setdefault('ZEE_INPUTS', 'modern')
from scipy.optimize import least_squares
from zee import hgmodel as H
import refit_hw as R
DEFAULT_GRID = '0.2,0.3,0.38,0.45,0.5,0.6,0.7,0.85,1.0,1.5,2.0,3.0'


def full_eta(parent, w):
    """complete quartic set of a candidate: the PARENT's (lambda2, lambda9, lambda_h) are kept and only the fitted spectators lambda7, lambda8, lambda10 (indices 1,2,4) are replaced
    (report v10: dropping lambda2/lambda9/lambda_h to zero builds a BFB failure into the completion when lambda10 != 0)."""
    e = np.array(parent, float).copy(); e[[1, 2, 4]] = w; return e


def complete_eta(x, eta, n=400, seed=0):
    """search the spectator quartics eta[[0,3,5]] = (lambda2, lambda9, lambda_h) for a completion that passes the low-energy gate (nominal boundary) before a point is called infeasible"""
    import campaign as C
    ok, info, why = C.stage_gate(x, eta, -1.0, rescaled=False)
    if ok: return eta, 'parent', why
    rng = np.random.default_rng(seed)
    for _ in range(n):
        e = np.array(eta, float).copy(); e[0] = rng.uniform(0, 1.5); e[3] = rng.uniform(0, 1.5); e[5] = rng.uniform(0, 1.0)
        ok, info, why2 = C.stage_gate(x, e, -1.0, rescaled=False)
        if ok: return e, 'searched', why2
        if why2 in ('chi2', 'limit', 'bsg', 'dy', 'higgs', 'yukawa'): return eta, 'none', why2      # failed for low-energy reasons: no completion of the spectators can help
    return eta, 'none', why


def start_point(label):
    if label.startswith('heavy'):                         # heavier scalars (LFV amplitudes ~ 1/m^2) and larger Yukawa normalisation: a seed, not the general solution
        x, e = start_point('min'); n = float(label[5:]); x = x.copy(); x[12] += np.log10(n); x[13] += np.log10(n); return x, e
    if label.startswith('min'):
        b = json.load(open(os.path.join(ROOT, 'results', 'refit_hw.json')))
        if label == 'min': r = b['best']
        else: r = b['all'][2*int(label[3:])]                 # distinct minima come in pairs (sign copies) in the stored list
        return np.array(r['x'], float), np.array(r['eta'], float)
    for s in json.load(open(os.path.join(ROOT, 'results', 'campaign_starts.json'))):
        if s['label'] == label: return np.array(s['x'], float), np.array(s['eta'], float)
    raise SystemExit('unknown start ' + label)


def fit(x1, w1, tbv, eta_parent):
    """refit all parameters except ln tan(beta) (removed from the variables, not pinned by a narrow box)"""
    free = [i for i in range(19) if i != 11]; lt = np.log(tbv)
    lo = np.concatenate([R.LO[free], [-3]*3]); hi = np.concatenate([R.HI[free], [3]*3]); xs = np.concatenate([R.XS[free], [0.5]*3])
    def full(p):
        z = np.empty(19); z[free] = p[:18]; z[11] = lt; return R.tx(z)
    f = lambda p: H.evaluate(list(full(p)), 'NO', eta=full_eta(eta_parent, p[18:]))['res']
    z0 = R.tz(x1); p0 = np.clip(np.concatenate([z0[free], w1]), lo, hi)
    r = None; hist = []
    for k in range(8):                                       # restart from the last point until chi2 stops improving (xtol terminations at a large gradient are not convergence)
        r1 = least_squares(f, p0, bounds=(lo, hi), max_nfev=400, x_scale=xs, xtol=1e-12, ftol=1e-12, gtol=1e-10)
        hist.append(2*float(r1.cost)); better = r is None or r1.cost < r.cost - 1e-4; r = r1 if (r is None or r1.cost <= r.cost) else r; p0 = r.x
        if not better: break
    r.hist = hist
    x = full(r.x); eta = full_eta(eta_parent, r.x[18:]); e = H.evaluate(list(x), 'NO', eta=eta)
    return x, eta, e, r


def record(x, eta, e, r, tbv, t0):
    o = e['obs']; parts = dict(osc=e['chi_osc'], univ=e['chi_univ'], stu=e['chi_stu'], higgs=e['chi_higgs'], **{k: float(v) for k, v in e['chi_bounds'].items()})
    d = dict(tb=tbv, chi2=float(e['chi2']), x=x.tolist(), eta=eta.tolist(), parts=parts, obs={k: float(o[k]) for k in ('tau_mu_g', 'mu_e_g', 'tau_e_g', 'h_tau_mu')},
             opt=dict(status=int(r.status), success=bool(r.success), nfev=int(r.nfev), cost=float(r.cost), optimality=float(r.optimality), message=r.message, chi2_restarts=r.hist), wall=time.time() - t0)
    if d['chi2'] <= CMIN + R.AC.DCHI2:                       # viable at the level of the statistic: run the low-energy gate (nominal boundary only)
        try:
            import campaign as C
            e2, how, why = complete_eta(x, eta); ok, info, why = C.stage_gate(x, e2, -1.0, rescaled=False)
            d['gate'] = dict(ok=bool(ok), reason=why, Lf=info.get('Lf'), completion=how, eta=e2.tolist())      # 'none' = no spectator completion found (a statement about this x, not about the tan(beta) value)
        except Exception as ex: d['gate'] = dict(ok=None, reason='error:' + type(ex).__name__)
    return d


CMIN = float(json.load(open(os.path.join(ROOT, 'results', 'refit_hw.json')))['best']['chi2'])

if __name__ == '__main__':
    label = sys.argv[1]; grid = sorted(float(g) for g in (sys.argv[2] if len(sys.argv) > 2 else DEFAULT_GRID).split(','))
    x0, eta_parent = start_point(label); w0 = eta_parent[[1, 2, 4]]; tb_s = float(np.exp(x0[11])); out = dict(start=label, tb_start=tb_s, chi2_min_ref=CMIN, results=[]); DIRECT = bool(os.environ.get('DIRECT'))
    fn = label + ('_tb' + '_'.join('%g' % g for g in grid) if DIRECT else '') + '.json'
    if label.startswith('min') or label.startswith('heavy'):         # fit-minimum starts carry no spectator completion: find one for the start itself first
        eta_parent, how, why = complete_eta(x0, eta_parent); out['start_completion'] = dict(how=how, reason=why, eta=eta_parent.tolist()); print(label, 'start completion', how, why, flush=True)
    out['eta_parent'] = eta_parent.tolist()
    seqs = [('direct', [g]) for g in grid] if DIRECT else [('up', [g for g in grid if g > tb_s]), ('down', [g for g in grid if g < tb_s][::-1])]   # DIRECT: every tan(beta) rescaled straight from the start (parallel jobs)
    for name, seq in seqs:
        x, w = x0.copy(), w0.copy()
        for tbv in seq:
            t0 = time.time(); x1 = H.rescale_fixed_nu(x, tbv)
            x, eta, e, r = fit(x1, w, tbv, eta_parent); w = eta[[1, 2, 4]]; d = record(x, eta, e, r, tbv, t0); d['direction'] = name; out['results'].append(d)
            print('%-8s %-6s tanb %5.2f chi2 %9.2f  tau_mu_g %.1e  status %d nfev %d opt %.1e %s' % (label, name, tbv, d['chi2'], d['obs']['tau_mu_g'], d['opt']['status'], d['opt']['nfev'], d['opt']['optimality'], d.get('gate', '')), flush=True)
            os.makedirs(os.path.join(os.environ.get('ZEE_OUT', os.path.join(ROOT, 'results')), 'profile_tb'), exist_ok=True)
            json.dump(out, open(os.path.join(os.environ.get('ZEE_OUT', os.path.join(ROOT, 'results')), 'profile_tb', fn), 'w'), indent=1)
        if DIRECT: x, w = x0.copy(), w0.copy()
