"""Scalar-consistency scale of a benchmark with ALL fermion Yukawa couplings evolved (zee/rgfull.py): top and bottom with both doublets, diagonal lepton Yukawas
of H1, the 3x3 lepton matrix Y2 = Dd of H2, and f.  Modes: 'top' (the production RGEs re-run with the full code: consistency check), 'btau' (b and the
diagonal lepton Yukawas, Y2 only through its -tan(beta) m/v part), 'all' (full Dd and f).  The prescribed 13 boundary variants of gate2/cert.FULL are re-run;
the failure criteria are those of gate2/cert.run (perturbativity, unitarity, boundedness) plus |all Yukawas| <= sqrt(4 pi).
Usage: ZEE_INPUTS=modern python3 -W ignore gate2/yukawa_run.py NAME MODE [VARIANTS...]  -> $ZEE_OUT/yukawa_<NAME>_<MODE>.json"""
import sys, os, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
os.environ.setdefault('ZEE_INPUTS', 'modern')
from scipy.integrate import solve_ivp
import bench, cert as CE
from zee import rgfull as F, hgmodel as H
from zee.observables import M_LEP
V = 246.22; YMAX = np.sqrt(4*np.pi); LOG16 = np.log(1e16); YB = np.sqrt(2)*2.7/V


def yinit(x, sc, mode):
    """(yb, Y1, Y2, f) at the boundary scale"""
    Z = np.zeros((3, 3), complex)
    if mode == 'top': return 0.0, Z, Z, Z
    Y1 = np.diag(np.sqrt(2)*np.asarray(M_LEP, float)/V).astype(complex)
    f, Y = H.unpack(np.asarray(x, float)); Dd = H.lepton_couplings(Y, sc)[1]
    if mode == 'btau': return YB, Y1, -sc['tb']*Y1, Z               # Y2 = r Y1 only (no flavour-changing part), f dropped
    if mode == 'all': return YB, Y1, Dd.astype(complex), f
    if mode == 'all_T': return YB, Y1, Dd.T.astype(complex), f      # other index orientation of Y2 (sensitivity)
    if mode == 'all_dag': return YB, Y1, Dd.conj().T.astype(complex), f
    raise ValueError(mode)


def run_full(c, t0, r, g0, y0, rtol=1e-7, n_per_dec=100):
    yb, Y1, Y2, f = y0; yt = g0[3]; g = np.asarray(g0[:3], float)
    z0 = F.pack(g, yt, r*yt, yb, r*yb, Y1, Y2, f, c)
    s = solve_ivp(F.rhs, (t0, LOG16), z0, dense_output=True, rtol=rtol, atol=rtol*1e-2)
    def bad(z):
        if not np.all(np.isfinite(z)): return 'nonfinite'
        why = CE._bad(z[61:])
        if why is None and F.yukawa_max(z) > YMAX: why = 'yukawa'
        return why
    grid = np.append(np.arange(t0, LOG16, np.log(10)/n_per_dec), LOG16); prev = t0
    for t in grid:
        if t > s.t[-1] + 1e-12: return prev/np.log(10), 'integration', s
        why = bad(s.sol(t))
        if why and t <= t0 + 1e-12: return -1.0, 'start:' + why, s
        if why:
            lo, hi = prev, t
            for _ in range(25):
                mid = 0.5*(lo + hi)
                if bad(s.sol(mid)): hi = mid
                else: lo = mid
            return lo/np.log(10), why, s
        prev = t
    return LOG16/np.log(10), None, s


def cert_full(x, eta, mode, variants=CE.FULL, n_per_dec=100):
    sc, c = bench.coup(x, eta); r = -sc['tb']; ml = min(sc['m1'], sc['mA'], sc['mH']); t0 = np.log(ml); out = {}
    y0 = yinit(x, sc, mode)
    for p in variants:
        g0 = CE.G0
        if p == 'default': tt, cc = t0, c
        elif p.startswith('match_'): mu0 = max(172.5, ml*float(p.split('_x')[1])); tt, cc, g0 = np.log(mu0), c, np.append(bench.sm_run(mu0)[:3], bench.sm_run(mu0)[3])
        elif p.startswith('matchH'): mu0 = max(172.5, max(sc['mH'], sc['mA'], sc['m2'])*float(p.split('_x')[1])); tt, cc, g0 = np.log(mu0), c, bench.sm_run(mu0)
        elif p.startswith('qscale'): tt, cc = t0, c*float(p.split('_')[1])
        elif p.startswith('sm_'):
            kind, k = p.split('_')[1], float(p.split('_')[2]); g = np.array(bench.SM_MT, float); g[3 if kind == 'yt' else 2] *= k
            tt, cc, g0 = t0, c, bench.sm_run(max(172.5, ml), start=tuple(g))
        v = run_full(cc, tt, r, np.asarray(g0, float), y0, n_per_dec=n_per_dec)
        out[p] = dict(log10=float(v[0]), fail=v[1])
    return min(o['log10'] for o in out.values()), out


if __name__ == '__main__':
    name, mode = sys.argv[1], sys.argv[2]; variants = tuple(sys.argv[3:]) or CE.FULL
    inp = json.load(open(os.path.join(ROOT, 'results', 'benchmarks_input.json')))
    st = {('S%d' % i): d for i, d in enumerate(json.load(open(os.path.join(ROOT, 'results', 'campaign_starts.json'))))}
    d = inp[name] if name in inp else st[name]
    m, out = cert_full(d['x'], d['eta'], mode, variants)
    outdir = os.environ.get('ZEE_OUT', os.path.join(ROOT, 'results', 'yukawa'));
    try: os.makedirs(outdir, exist_ok=True)
    except OSError: outdir = '/tmp'
    json.dump(dict(name=name, mode=mode, min=m, variants=out), open(os.path.join(outdir, 'yukawa_%s_%s.json' % (name, mode)), 'w'), indent=1)
    print(name, mode, 'min %.2f' % m, {k: (round(v['log10'], 2), v['fail']) for k, v in out.items()})
