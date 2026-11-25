"""Robustness analysis of a benchmark (x, eta): RG trajectory on a fine grid with bracketing of the first failure, ODE tolerances,
matching-scale variation with SM running inputs, BFB margin and unitarity eigenvalues along the flow, deep global-minimum search,
amplitude decomposition. Usage (module): bench.analyse(x, eta) -> dict"""
import sys, numpy as np
sys.path.insert(0, '/home/claude/zee_fit'); sys.path.insert(0, '/home/claude/zee_fit/gate1')
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, brentq
from zee import rgfast2 as R, bfb2, vac
from zee import hgmodel as H
CM = 4*np.pi; LOG16 = np.log(1e16)
SM_MT = (0.3583, 0.6478, 1.1666, 0.9369)   # MSbar g', g, g_s, y_t at m_t = 172.5 GeV (Buttazzo et al. 2013)
def sm_run(mu):
    """one-loop SM running of (g', g, g_s, y_t) from m_t to mu"""
    def f(t, z):
        g1, g2, g3, yt = z; k = 1/(16*np.pi**2)
        return [k*41/6*g1**3, -k*19/6*g2**3, -k*7*g3**3, k*yt*(4.5*yt**2 - 8*g3**2 - 2.25*g2**2 - 17/12*g1**2)]
    return solve_ivp(f, (np.log(172.5), np.log(mu)), SM_MT, rtol=1e-10, atol=1e-12).y[:, -1]
def coup(x, eta):
    sc = H.evaluate(list(x), 'NO')['sc']; c = np.zeros(11)
    c[0], c[2], c[3], c[4], c[5] = sc['lam1'], sc['lam3'], sc['lam4'], sc['lam5'], sc['lam6']; c[1], c[6:] = eta[0], eta[1:]
    return sc, c
def margins(c):
    return bfb2.min_vred(c, refine=3), vac.unit_max(c), np.abs(c).max()
def fails(c):
    vb, u, mx = margins(c)
    return ('BFB' if vb < -1e-6 else None) or ('unitarity' if u > 8*np.pi else None) or ('perturbativity' if mx > CM else None)
def trajectory(c0, t0, r, g, n_per_dec=100, rtol=1e-8):
    z0 = np.concatenate([g, c0]); grid = np.arange(t0, LOG16, np.log(10)/n_per_dec)
    s = solve_ivp(R.rhs_r(r), (t0, LOG16), z0, dense_output=True, rtol=rtol, atol=rtol*1e-2)
    out = dict(min_bfb=np.inf, max_unit=0.0, max_lam=0.0, fail=None, Lambda=LOG16)
    prev = t0
    for t in grid:
        if t > s.t[-1]: out.update(fail='integration', Lambda=prev); break
        c = s.sol(t)[4:]; vb, u, mx = margins(c); why = fails(c)
        if why:
            lo, hi = prev, t      # bracket the first failure
            for _ in range(30):
                mid = 0.5*(lo + hi)
                if fails(s.sol(mid)[4:]): hi = mid
                else: lo = mid
            out.update(fail=why, Lambda=lo); break
        out['min_bfb'] = min(out['min_bfb'], vb); out['max_unit'] = max(out['max_unit'], u); out['max_lam'] = max(out['max_lam'], mx); prev = t
    out['log10Lambda'] = out.pop('Lambda')/np.log(10); return out
def deep_vacuum(c, sc, n=2000, seed=0):
    rng = np.random.default_rng(seed); V = H.V
    l1, l3, l6, l8 = c[0], c[2], c[5], c[7]
    args = (c, -l1/2, (sc['MHp']**2 - 0.5*V**2*l3)/V**2, l6/2, (sc['M33']**2 - 0.5*V**2*l8)/V**2, sc['mu']/V)
    f = lambda p: float(vac.vfull(p, *args)); vew = -l1/8; mins = []
    for k in range(n):
        p0 = rng.normal(size=10); p0 *= np.exp(rng.uniform(np.log(.01), np.log(100)))/np.linalg.norm(p0)
        with np.errstate(all='ignore'): rr = minimize(f, p0, method='L-BFGS-B', options={'maxiter': 2000})
        if np.isfinite(rr.fun): mins.append((rr.fun, np.linalg.norm(rr.x)))
    mins = np.array(mins); other = mins[np.abs(mins[:, 0] - vew) > 1e-6*abs(vew)]
    deepest = other[:, 0].min() if len(other) else np.nan
    return dict(V_EW=vew, deepest_other=float(deepest), gap=float(deepest - vew) if len(other) else np.nan, global_ok=bool(mins[:, 0].min() >= vew - 1e-6*abs(vew)), n_starts=n)
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
