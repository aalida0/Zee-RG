"""Two-loop cross-check of the RG scale reach (referee plan, item 'targeted two-loop check via PyR@TE 3').

PyR@TE 3 generates the beta functions of the Zee model (validation/Zee.model, with the gauge-only SM fermions of
validation/Zee_full.model so that the two-loop gauge beta functions are complete): gauge couplings g1 (hypercharge, g'), g2, g3,
the top Yukawa y_t, the second-doublet top Yukawa rho, the 11 real quartics.  The same 13 prescribed variants, the same boundary
conditions and the same failure criteria as gate2/cert.py are run with
   (a) the generated ONE-loop functions (must reproduce the production solver rgfast2 -> validates this driver), and
   (b) the generated one- plus TWO-loop functions,
and the first-failure scale of every variant is compared.  This is a check of the ROBUSTNESS of the one-loop conclusion to the next order in the beta
functions (with the tree-level boundary values and without two-loop threshold corrections, so it is not a two-loop calculation of the matching): it
does not make the result two-loop accurate.  Scope: only top and rho Yukawas, no b/tau/lepton Yukawas (as the one-loop code); rho is evolved independently (the
ratio rho/y_t is RG invariant only at one loop).
Usage: ZEE_INPUTS=modern python3 -W ignore validation/two_loop_check.py RGEs.py NAME [NAME ...] [--out file.json]"""
import sys, os, json, inspect, importlib.util, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
os.environ.setdefault('ZEE_INPUTS', 'modern')
from scipy.integrate import solve_ivp
import cert as CE, bench, validate_finalist as VF

K = 1 / (16 * np.pi**2)
QN = ['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9', 'l10', 'lh']
LOG16 = CE.LOG16


def load_rges(path):
    spec = importlib.util.spec_from_file_location('rges', path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def make_rhs(mod, loops):
    names = ['g1', 'g2', 'g3', 'yt', 'rho'] + QN
    fns = []
    for n in names:
        f = getattr(mod, 'beta_' + n); fns.append((f, list(inspect.signature(f).parameters)[1:]))

    def rhs(t, z):
        env = dict(zip(names, z)); out = np.empty(len(z))
        for i, (f, args) in enumerate(fns):
            a = [env[k] for k in args]; v = f(1, *a) * K
            if loops >= 2: v += f(2, *a) * K * K
            out[i] = v
        return out
    return rhs


def run(mod, loops, c, t0, r, g0):
    z0 = np.concatenate([g0[:3], [g0[3], r * g0[3]], c])
    ev = lambda t, z: 50.0 - np.abs(z[5:]).max(); ev.terminal = True
    ey = lambda t, z: 50.0 - max(abs(z[3]), abs(z[4])); ey.terminal = True
    s = solve_ivp(make_rhs(mod, loops), (t0, LOG16), z0, dense_output=True, rtol=1e-8, atol=1e-10, events=[ev, ey])
    grid = np.append(np.arange(t0, LOG16, np.log(10) / CE.N_PER_DEC), LOG16); prev = t0

    def bad(z):
        return CE._bad(z[5:]) or ('yukawa' if (not np.all(np.isfinite(z[3:5])) or abs(z[3]) > CE.YMAX or abs(z[4]) > CE.YMAX) else None)
    for t in grid:
        if t > s.t[-1] + 1e-12:                    # solver stopped on a Landau-pole event: everything beyond is non-perturbative
            return prev / np.log(10), 'perturbativity(landau)'
        zt = s.sol(t); why = bad(zt)
        if why and prev == t0 and t == t0: return -1.0, 'start:' + why
        if why:
            lo, hi = prev, t
            for _ in range(10):
                mid = 0.5 * (lo + hi)
                if bad(s.sol(mid)): hi = mid
                else: lo = mid
            return lo / np.log(10), why
        prev = t
    return LOG16 / np.log(10), None


def cert(mod, loops, x, eta, variants=CE.FULL):
    sc, c = bench.coup(x, eta); r = -sc['tb']; ml = min(sc['m1'], sc['mA'], sc['mH']); t0 = np.log(ml); out = {}
    for p in variants:
        if p == 'default': v = run(mod, loops, c, t0, r, CE.G0)
        elif p.startswith('match_'): mu0 = max(172.5, ml * float(p.split('_x')[1])); v = run(mod, loops, c, np.log(mu0), r, bench.sm_run(mu0))
        elif p.startswith('matchH'): mu0 = max(172.5, max(sc['mH'], sc['mA'], sc['m2']) * float(p.split('_x')[1])); v = run(mod, loops, c, np.log(mu0), r, bench.sm_run(mu0))
        elif p.startswith('qscale'): v = run(mod, loops, c * float(p.split('_')[1]), t0, r, CE.G0)
        elif p.startswith('sm_'):
            kind, k = p.split('_')[1], float(p.split('_')[2]); g = np.array(bench.SM_MT, float); g[3 if kind == 'yt' else 2] *= k
            v = run(mod, loops, c, t0, r, bench.sm_run(max(172.5, ml), start=tuple(g)))
        out[p] = dict(log10=float(v[0]), fail=v[1])
    return min(o['log10'] for o in out.values()), out


if __name__ == '__main__':
    args = sys.argv[1:]; outp = None
    if '--out' in args: i = args.index('--out'); outp = args[i + 1]; del args[i:i + 2]
    mod = load_rges(args[0]); res = {}
    for name in args[1:]:
        x, eta = VF._load(name); rec = {}
        m0, o0 = CE.cert(x, eta, CE.FULL); rec['production_1loop'] = dict(min=m0, variants=o0)
        for loops in (1, 2):
            m, o = cert(mod, loops, x, eta); rec['pyrate_%dloop' % loops] = dict(min=m, variants=o)
        res[name] = rec
        print(name, 'production 1-loop %.2f | PyR@TE 1-loop %.2f | PyR@TE 2-loop %.2f' % (m0, rec['pyrate_1loop']['min'], rec['pyrate_2loop']['min']))
        for p in CE.FULL:
            print('   %-12s %6.2f (%s) %6.2f (%s) %6.2f (%s)' % (p, o0[p]['log10'], o0[p]['fail'], rec['pyrate_1loop']['variants'][p]['log10'], rec['pyrate_1loop']['variants'][p]['fail'],
                                                                rec['pyrate_2loop']['variants'][p]['log10'], rec['pyrate_2loop']['variants'][p]['fail']))
    if outp: json.dump(res, open(outp, 'w'), indent=1)
