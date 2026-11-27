"""Fast scale certification for campaign chains: the minimum over matching variants of the one-loop scale at which perturbativity,
tree-level unitarity or boundedness first fails (same variants and boundary conditions as gate2/benchmarks.py, coarser grid:
N_PER_DEC scales per decade with bisection of the first failure, coarse-then-refined BFB margin).  The chain value is a screening
estimate; the claims of the paper use benchmarks.py (100 scales per decade, 17 pieces).  Usage: cert(x, eta) -> (log10 scale, dict per variant)."""
import sys, os, numpy as np
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_R, _R + '/gate1', _R + '/gate2']
from scipy.integrate import solve_ivp
from zee import rgfast2 as R, bfb2, vac
import bench, lmax2 as L

N_PER_DEC = 10
VARIANTS = ('default', 'match_x0.5', 'matchH_x0.5', 'qscale_1.1', 'sm_yt_1.02')
FULL = ('default', 'match_x0.5', 'match_x1', 'match_x2', 'matchH_x0.5', 'matchH_x1', 'matchH_x2', 'qscale_1.1', 'qscale_0.9',
        'sm_yt_1.02', 'sm_yt_0.98', 'sm_gs_1.03', 'sm_gs_0.97')
SENS = ('spec_x1.01', 'spec_x0.99', 'spec_x1.02', 'spec_x0.98')   # threshold-like sensitivity of the boundary spectrum, reported, not part of the minimum
G0 = np.array([0.36, 0.64, 1.05, 0.88]); LOG16 = np.log(1e16)


def _bad(c):
    if not np.all(np.isfinite(c)): return 'nonfinite'
    if np.abs(c).max() > 4*np.pi: return 'perturbativity'
    if vac.unit_max(c) > 8*np.pi: return 'unitarity'
    v = L._bfb(c)
    return 'BFB' if v < -bench.BFB_TOL else ('BFB?' if v < bench.BFB_TOL else None)       # 'BFB?': margin within the numerical tolerance of zero = unresolved, not a known crossing


def reach_class(cs, variants):
    """single classifier of a certified reach, shared by campaign, collector, benchmarks and the matching drivers: 'boundary_failure' (some variant violates a condition at the matching scale: no scale),
    'unresolved' (borderline BFB margin within the numerical tolerance of zero), 'numerical_stop' (solver stop or non-finite coupling: a demonstrated PREFIX, not a first-failure scale),
    'match_failed' (the threshold-matching root was not converged), 'ceiling' (no failure up to 1e16 GeV, right-censored) or 'physical_failure' (first crossing of BFB/unitarity/perturbativity/Yukawa
    found by bisection).  Only the last two carry a scale; the sentinel -1 is never a scale."""
    fl = [str(v['fail']) if v.get('fail') is not None else None for v in variants.values()]
    if any(f is not None and f.startswith('start') for f in fl): return 'boundary_failure'
    if any(f == 'match_failed' for f in fl): return 'match_failed'
    if any(f in ('integration', 'nonfinite') for f in fl) or cs is None or not np.isfinite(cs): return 'numerical_stop'
    if any(f == 'BFB?' for f in fl): return 'unresolved'
    if all(f is None for f in fl): return 'ceiling'
    return 'physical_failure'


NUMERICAL_FAILS = ('integration', 'nonfinite')


def variant_status(fail):
    """classification of ONE variant from its first-failure string, with the same semantics as reach_class (single source of truth for validation drivers):
    'boundary_failure' ('start:*'), 'match_failed', 'numerical_stop' (integration/nonfinite: a demonstrated prefix, never a cutoff), 'unresolved' (borderline BFB 'BFB?'),
    'ceiling' (no failure up to 1e16 GeV) or 'physical_failure' (first crossing of BFB/unitarity/perturbativity/yukawa).  Only the last two carry a scale."""
    f = None if fail is None else str(fail)
    if f is None or f == 'None': return 'ceiling'
    if f.startswith('start'): return 'boundary_failure'
    if f == 'match_failed': return 'match_failed'
    if f in NUMERICAL_FAILS: return 'numerical_stop'
    if f == 'BFB?': return 'unresolved'
    return 'physical_failure'


def eligible(status):
    """a variant result may enter a minimum / robust scale only if it is a physical first failure or a ceiling"""
    return status in ('physical_failure', 'ceiling')


def status_fields(cs, rc, logL, base):
    """(rg_status, baseline, meets_requested_cutoff, final_status) of a campaign record from the certified minimum cs, its reach class rc, the requested cutoff logL and the low-energy status base.
    logL < 0 is the low-energy baseline with the nominal boundary only: no cutoff comparison is made (the start-failure sentinel -1 is not a scale), and only a boundary failure fails it.
    A numerical stop, borderline BFB or failed matching has no valid reach: it can never meet a cutoff, and a record that would otherwise PASS becomes UNRESOLVED."""
    valid = rc in ('physical_failure', 'ceiling')
    rg = 'not run' if cs is None else ('ok' if valid else ('start-fail' if rc == 'boundary_failure' else 'unresolved/' + str(rc)))
    baseline = logL is not None and logL < 0
    if logL is None or cs is None or baseline: meets = None
    else: meets = bool(valid and cs >= logL)
    if cs is None: fin = base
    elif rc == 'boundary_failure': fin = 'FAIL'
    elif baseline or meets is None: fin = base if valid else ('UNRESOLVED' if base == 'PASS' else base)
    elif meets: fin = base
    else: fin = 'FAIL' if valid else ('UNRESOLVED' if base == 'PASS' else base)
    return rg, bool(baseline), meets, fin


YMAX = np.sqrt(4*np.pi)


def _ybad(z4, r):
    """running top Yukawa y_t and the second-doublet top coupling rho_t = r*y_t (r = -tan(beta) fixed ratio) must stay perturbative; only these two Yukawas and the gauge couplings are evolved
    (lepton Yukawas, b and tau are not: see the manuscript scope), so this is a scalar-quartic + top-sector consistency test, not an all-Yukawa one"""
    return 'yukawa' if (not np.all(np.isfinite(z4)) or abs(z4[3]) > YMAX or abs(r*z4[3]) > YMAX) else None


def run(c, t0, r, g0):
    s = solve_ivp(R.rhs_r(r), (t0, LOG16), np.concatenate([g0, c]), dense_output=True, rtol=1e-7, atol=1e-9)
    grid = np.append(np.arange(t0, LOG16, np.log(10)/N_PER_DEC), LOG16); prev = t0
    for t in grid:
        if t > s.t[-1] + 1e-12: return prev/np.log(10), 'integration'
        zt = s.sol(t); why = _bad(zt[4:]) or _ybad(zt[:4], r)
        if why and prev == t0 and t == t0: return -1.0, 'start:' + why      # fails at the boundary itself: nothing was verified
        if why:
            lo, hi = prev, t
            for _ in range(10):
                mid = 0.5*(lo + hi)
                zm = s.sol(mid)
                if _bad(zm[4:]) or _ybad(zm[:4], r): hi = mid
                else: lo = mid
            return lo/np.log(10), why
        prev = t
    return LOG16/np.log(10), None


def cert(x, eta, variants=VARIANTS):
    sc, c = bench.coup(x, eta); r = -sc['tb']; ml = min(sc['m1'], sc['mA'], sc['mH']); t0 = np.log(ml); out = {}
    for p in variants:
        if p == 'default': v = run(c, t0, r, G0)
        elif p.startswith('match_'): mu0 = max(172.5, ml*float(p.split('_x')[1])); v = run(c, np.log(mu0), r, bench.sm_run(mu0))
        elif p.startswith('matchH'): mu0 = max(172.5, max(sc['mH'], sc['mA'], sc['m2'])*float(p.split('_x')[1])); v = run(c, np.log(mu0), r, bench.sm_run(mu0))
        elif p.startswith('qscale'): v = run(c*float(p.split('_')[1]), t0, r, G0)
        elif p.startswith('spec_'):      # physically consistent boundary shift (bench.spec_shift): M_H+ scaled by f at fixed m_H, m_A, m_h, lambda_4 re-solved
            f = float(p.split('_x')[1]); x2 = bench.spec_shift(x, f)
            sc2, c2 = bench.coup(x2, eta); v = run(c2, np.log(min(sc2['m1'], sc2['mA'], sc2['mH'])), -sc2['tb'], G0)
        elif p.startswith('sm_'):
            kind, k = p.split('_')[1], float(p.split('_')[2]); g = np.array(bench.SM_MT, float); g[3 if kind == 'yt' else 2] *= k
            v = run(c, t0, r, bench.sm_run(max(172.5, ml), start=tuple(g)))
        out[p] = dict(log10=float(v[0]), fail=v[1])
    return min(o['log10'] for o in out.values()), out
