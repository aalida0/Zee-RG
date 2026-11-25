"""Robustness analysis of a benchmark (x, eta): RG trajectory on a fine grid with bracketing of the first failure, ODE tolerances,
matching-scale variation with SM running inputs, BFB margin and unitarity eigenvalues along the flow, deep global-minimum search,
amplitude decomposition. Usage (module): bench.analyse(x, eta) -> dict"""
import sys, numpy as np
_R = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, _R + '/gate1')
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, brentq
from zee import rgfast2 as R, bfb2, vac
from zee import hgmodel as H
CM = 4*np.pi; LOG16 = np.log(1e16)
SM_MT = (0.3583, 0.6478, 1.1666, 0.9369)   # MSbar g', g, g_s, y_t at m_t = 172.5 GeV (Buttazzo et al. 2013)
def sm_run(mu, start=None):
    """one-loop SM running of (g', g, g_s, y_t) from m_t to mu (start = alternative inputs at m_t for sensitivity checks)"""
    def f(t, z):
        g1, g2, g3, yt = z; k = 1/(16*np.pi**2)
        return [k*41/6*g1**3, -k*19/6*g2**3, -k*7*g3**3, k*yt*(4.5*yt**2 - 8*g3**2 - 2.25*g2**2 - 17/12*g1**2)]
    return solve_ivp(f, (np.log(172.5), np.log(mu)), SM_MT if start is None else start, rtol=1e-10, atol=1e-12).y[:, -1]
def coup(x, eta):
    sc = H.evaluate(list(x), 'NO')['sc']; c = np.zeros(11)
    c[0], c[2], c[3], c[4], c[5] = sc['lam1'], sc['lam3'], sc['lam4'], sc['lam5'], sc['lam6']; c[1], c[6:] = eta[0], eta[1:]
    return sc, c
def margins(c):
    return bfb2.min_vred(c, refine=3), vac.unit_max(c), np.abs(c).max()
BFB_TOL = 1e-6     # |V_red| below this is numerically unresolved, not "bounded"
def fails(c, with_margins=False):
    """first violated condition at c, or None. Non-finite input fails; a BFB margin within BFB_TOL of zero is 'BFB?'
    (unresolved) and counts as a failure for the cutoff (fail closed)."""
    if not np.all(np.isfinite(c)): return ('nonfinite', None) if with_margins else 'nonfinite'
    vb, u, mx = margins(c)
    why = ('BFB' if vb < -BFB_TOL else None) or ('BFB?' if vb < BFB_TOL else None) or ('unitarity' if u > 8*np.pi else None) or ('perturbativity' if mx > CM else None)
    return (why, (vb, u, mx)) if with_margins else why
def trajectory(c0, t0, r, g, n_per_dec=100, rtol=1e-8):
    z0 = np.concatenate([g, c0]); grid = np.append(np.arange(t0, LOG16, np.log(10)/n_per_dec), LOG16)   # endpoint included
    s = solve_ivp(R.rhs_r(r), (t0, LOG16), z0, dense_output=True, rtol=rtol, atol=rtol*1e-2)
    out = dict(min_bfb=np.inf, max_unit=0.0, max_lam=0.0, fail=None, Lambda=LOG16, solver_status=int(s.status))
    prev = t0
    for t in grid:
        if t > s.t[-1] + 1e-12: out.update(fail='integration', Lambda=prev); break   # solver stopped early
        c = s.sol(t)[4:]; why, mg = fails(c, True)
        if why is None: vb, u, mx = mg
        if why:
            lo, hi = prev, t      # bracket the first failure
            for _ in range(30):
                mid = 0.5*(lo + hi)
                if fails(s.sol(mid)[4:]): hi = mid
                else: lo = mid
            out.update(fail=why, Lambda=lo); break
        out['min_bfb'] = min(out['min_bfb'], vb); out['max_unit'] = max(out['max_unit'], u); out['max_lam'] = max(out['max_lam'], mx); prev = t
    if out['fail'] is None and prev < LOG16 - 1e-9: out.update(fail='integration', Lambda=prev)
    out['log10Lambda'] = out.pop('Lambda')/np.log(10); return out
def deep_vacuum(c, sc, n=2000, seed=0):
    """Deep global-minimum search (vac.global_search: analytic gradients, deterministic charge-breaking starts,
    fail closed). The claim is 'no deeper minimum found by this search', not a certificate."""
    r = vac.global_search(c, sc, nstart=n, rng=np.random.default_rng(seed))
    return dict(V_EW=r['V_EW'], deepest=r['deepest'], gap=r['deepest'] - r['V_EW'], status=r['status'], global_ok=r['status'] == 'ok',
                n_starts=r['n'], n_converged=r['n_conv'], n_failed=r['n_fail'], ew_gradient=r['ew_grad'])
def analyse(x, eta):
    x = np.asarray(x, float); eta = np.asarray(eta, float); sc, c = coup(x, eta); r = -sc['tb']
    mlight = min(sc['m1'], sc['mA'], sc['mH']); t0 = np.log(mlight); res = {}
    res['default'] = trajectory(c, t0, r, np.array([0.36, 0.64, 1.05, 0.88]))
    for lab, f in (('match_x0.5', 0.5), ('match_x1', 1.0), ('match_x2', 2.0)):
        mu0 = max(172.5, mlight*f); res[lab] = trajectory(c, np.log(mu0), r, sm_run(mu0))
    res['rtol_1e-6'] = trajectory(c, t0, r, np.array([0.36, 0.64, 1.05, 0.88]), rtol=1e-6)
    res['rtol_1e-10'] = trajectory(c, t0, r, np.array([0.36, 0.64, 1.05, 0.88]), rtol=1e-10)
    res['grid_300'] = trajectory(c, t0, r, np.array([0.36, 0.64, 1.05, 0.88]), n_per_dec=300)
    res['vacuum'] = deep_vacuum(c, sc)
    res['quartics_start'] = dict(zip(['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9', 'l10', 'lh'], map(float, c)))
    return res
