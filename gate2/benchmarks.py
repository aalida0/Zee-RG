"""Clean benchmark reproduction (referee C06, C13): reads the committed inputs results/benchmarks_input.json (x, eta per
benchmark) and the laboratory minimum results/refit_corrected.pkl, applies the shared acceptance gate (gate2/accept.py),
computes the complex dipole components, neutrino quantities, collider predictions and the RG/vacuum robustness pieces,
and writes strict JSON results/benchmarks.json (no NaN). Robustness pieces are cached in results/bench_cache/ under a key
that hashes (x, eta, piece, physics source files), so a changed kernel never reuses stale pieces.
Usage: ZEE_INPUTS=modern python gate2/benchmarks.py [BM1 BM2 ...]      (corrected statistic: ZEE_INPUTS=modern2026 BM_CMIN_FILE=results/refit_hw_cT_all.json BM_INPUT=... BM_OUTPUT=...)"""
import sys, os, json, pickle, hashlib, glob, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
import accept as AC, bench
from zee import hgmodel as H, neutrino as N, dipfull as DF
RES = os.path.join(ROOT, 'results'); CACHE = os.environ.get('ZEE_CACHE', os.path.join(RES, 'bench_cache'))     # ZEE_CACHE / BM_OUTPUT: writable locations when the checkout is read-only
try: os.makedirs(CACHE, exist_ok=True)
except OSError as _e: raise SystemExit('benchmark cache %s is not writable (%s): set ZEE_CACHE to a writable directory' % (CACHE, _e))
G0 = np.array([0.36, 0.64, 1.05, 0.88])
SRC = sorted(glob.glob(os.path.join(ROOT, 'zee', '*.py'))) + [os.path.join(ROOT, 'gate2', f) for f in ('bench.py', 'benchmarks.py')]
CODE = hashlib.sha256(b''.join(open(f, 'rb').read() for f in SRC)).hexdigest()[:16]


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.bool_, bool)): return bool(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (float, np.floating)): return float(o) if np.isfinite(o) else None
    if isinstance(o, (complex, np.complexfloating)): return [clean(o.real), clean(o.imag)]
    if isinstance(o, np.ndarray): return clean(o.tolist())
    return o


def piece(x, eta, p):
    key = hashlib.sha256(json.dumps([list(map(float, x)), list(map(float, eta)), p, CODE]).encode()).hexdigest()[:20]
    fn = os.path.join(CACHE, key + '.pkl')
    if os.path.exists(fn): return pickle.load(open(fn, 'rb'))
    sc, c = bench.coup(x, eta); r = -sc['tb']; ml = min(sc['m1'], sc['mA'], sc['mH']); t0 = np.log(ml)
    if p == 'default': out = bench.trajectory(c, t0, r, G0)
    elif p.startswith('match_'):
        f = float(p.split('_x')[1]); mu0 = max(172.5, ml*f); out = dict(bench.trajectory(c, np.log(mu0), r, bench.sm_run(mu0)), mu0=mu0)
    elif p.startswith('matchH'):      # boundary at the heavy-doublet scale (x0.5, x1, x2 of the heaviest of mH, mA, m2): the light states decouple below it
        f = float(p.split('_x')[1]); mh = max(sc['mH'], sc['mA'], sc['m2']); mu0 = max(172.5, mh*f); out = dict(bench.trajectory(c, np.log(mu0), r, bench.sm_run(mu0)), mu0=mu0)
    elif p.startswith('qscale'):      # relative shift of all boundary quartics, a conservative proxy for one-loop threshold corrections
        k = float(p.split('_')[1]); out = bench.trajectory(c*k, t0, r, G0)
    elif p.startswith('spec_'):       # physically consistent shift (bench.spec_shift): M_H+ scaled by f at fixed m_H, m_A, m_h; lambda_4 re-solved from the mass relation
        f = float(p.split('_x')[1]); x2 = bench.spec_shift(x, f)
        sc2, c2 = bench.coup(x2, eta); out = bench.trajectory(c2, np.log(min(sc2['m1'], sc2['mA'], sc2['mH'])), -sc2['tb'], G0)
    elif p.startswith('sm_'):         # sensitivity to the SM inputs at m_t: y_t +-2%, g_s +-3%
        kind, k = p.split('_')[1], float(p.split('_')[2]); g = np.array(bench.SM_MT, float); g[3 if kind == 'yt' else 2] *= k
        out = bench.trajectory(c, t0, r, bench.sm_run(max(172.5, ml), start=tuple(g)))
    elif p == 'rtol_1e-6': out = bench.trajectory(c, t0, r, G0, rtol=1e-6)
    elif p == 'rtol_1e-10': out = bench.trajectory(c, t0, r, G0, rtol=1e-10)
    elif p == 'grid_300': out = bench.trajectory(c, t0, r, G0, n_per_dec=300)
    elif p == 'vacuum': out = bench.deep_vacuum(c, sc, n=1500)
    pickle.dump(out, open(fn, 'wb')); return out


PIECES = ['default', 'match_x0.5', 'match_x1', 'match_x2', 'matchH_x0.5', 'matchH_x1', 'matchH_x2', 'qscale_1.1', 'qscale_0.9', 'sm_yt_1.02', 'sm_yt_0.98', 'sm_gs_1.03', 'sm_gs_0.97', 'spec_x1.01', 'spec_x0.99', 'spec_x1.02', 'spec_x0.98',
          'rtol_1e-6', 'rtol_1e-10', 'grid_300', 'vacuum']
TRAJ = [p for p in PIECES if p != 'vacuum' and not p.startswith('spec_')]   # spec_* are reported as sensitivity (they move the boundary point), not part of the minimum


def rg_status_of(fails_, reach):
    """(start_fail, unresolved, label) of the RG part from the per-variant first-failure reasons.  'start:*' = a condition is violated at the matching
    scale itself (no scale is demonstrated, fail); 'integration' (solver stopped early) and 'nonfinite' (a coupling became non-finite) are NUMERICAL stops:
    the last good scale is a demonstrated prefix (a valid lower bound on the scale) but not a physical failure scale, so the record is UNRESOLVED
    (fail closed) and must not be quoted as a cutoff.  Anything else (unitarity, perturbativity, BFB, BFB?) is a physical first failure."""
    reasons = [str(v) for v in fails_.values()]
    start_fail = any(v.startswith('start') for v in reasons)
    borderline = any(v == 'BFB?' for v in reasons)       # BFB margin within the numerical tolerance of zero: unresolved, not a known physical crossing
    numerical = any(v in ('integration', 'nonfinite') for v in reasons) or not np.isfinite(reach) or borderline
    label = 'start-fail' if start_fail else (('unresolved (borderline BFB margin)' if borderline else 'unresolved (numerical stop: demonstrated prefix only)') if numerical else 'ok')
    return start_fail, numerical, label


def record(name, x, eta, cmin):
    x = np.asarray(x, float); eta = np.asarray(eta, float)
    a = AC.accept(x, eta, cmin, n_vac=80)
    sc = a['sc']; f, Y = H.unpack(x); Lam, Dd = H.lepton_couplings(Y, sc); mass = dict(h=H.MH, H=sc['mH'], A=sc['mA'])
    AL, AR, P = DF.amplitudes(Lam, mass, f, Dd, sc, DF.charged_trilinears(sc, eta[1], eta[2], eta[4]), parts=True)
    dip = {nm: {k: dict(A_L=v[0][i, j], A_R=v[1][i, j]) for k, v in P.items()} for nm, (i, j) in (('mu_e_g', (0, 1)), ('tau_mu_g', (1, 2)), ('tau_e_g', (0, 2)))}
    M = N.mnu_hg_leading(f, H.M_LEP, Y, sc['tb'], sc['m1'], sc['m2'], sc['mu']); osc = N.oscillation_observables(M, 'NO')
    U = osc['U'] if 'U' in osc else None
    nu = dict(masses_eV=osc['masses_eV'], sum_m=float(np.sum(osc['masses_eV'])), m_bb_eV=abs(M[0, 0])*1e9,
              m_beta_eV=a['obs']['m_beta'], s12sq=osc['s12sq'], s13sq=osc['s13sq'], s23sq=osc['s23sq'], delta_deg=osc['delta_deg'] % 360)
    rob = {p: piece(x, eta, p) for p in PIECES}
    _, c = bench.coup(x, eta)
    reach = min(rob[p]['log10Lambda'] for p in TRAJ); vac_ok = bool(rob['vacuum']['global_ok'])
    fails_ = {p: str(rob[p].get('fail')) for p in TRAJ}
    start_fail, unres_rg, rg_status = rg_status_of(fails_, reach)
    le_status = 'PASS' if (a['status'] == 'PASS' and vac_ok) else ('FAIL' if a['status'] == 'FAIL' or not vac_ok else 'UNRESOLVED')
    final = 'FAIL' if (le_status == 'FAIL' or start_fail) else ('UNRESOLVED' if (le_status == 'UNRESOLVED' or unres_rg) else 'PASS')
    return dict(x=x, eta=eta, accepted=(final == 'PASS'), status=final, low_energy_vacuum_status=le_status, rg_status=rg_status, certified_log10_scale=float(reach), scope='one-loop, tree-level-boundary scalar-consistency estimate (not a UV-completion scale)',
                a_status=a['status'], a_unresolved=a['unresolved'], a_failed_cuts=a.get('failed_cuts'), dy=a.get('DY'), bsgamma_e4=a.get('bsgamma_e4'), cuts=a['cuts'], chi2=a['chi2'], chi2_min=cmin, chi2_parts=a['chi2_parts'],
                Lf_scan=a['Lf'], Lf_scan_status=a['Lf_status'], vacuum_scan=a['vacuum'],
                masses=dict(h=H.MH, H=sc['mH'], A=sc['mA'], h1p=sc['m1'], h2p=sc['m2']), tan_beta=sc['tb'], c_beta_alpha=sc['cba'],
                phi=sc['phi'], mu_over_v=sc['mu']/H.V, quartics=dict(zip(['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9', 'l10', 'lh'], c)),
                f={'e_mu': f[0, 1].real, 'e_tau': f[0, 2].real, 'mu_tau': f[1, 2].real},
                Y2={f'{p}{q}': [Y[i, j].real, Y[i, j].imag] for i, p in enumerate('emt') for j, q in enumerate('emt') if abs(Y[i, j]) > 0},
                neutrino=nu, observables=a['obs'], collider=a.get('collider'), dipole_components=dip,
                robustness=rob, code_hash=CODE)


if __name__ == '__main__':
    inp = json.load(open(os.environ.get('BM_INPUT', os.path.join(RES, 'benchmarks_input.json'))))     # BM_INPUT/BM_OUTPUT: e.g. results/campaign_finalists.json
    cmin = float(json.load(open(os.environ.get('BM_CMIN_FILE', os.path.join(RES, 'refit_hw.json'))))['best']['chi2'])     # common refit with the production statistic (gate2/refit_hw.py); BM_CMIN_FILE: e.g. results/refit_hw_cT_all.json (corrected T, modern2026)
    names = [n for n in inp if len(sys.argv) == 1 or n in sys.argv[1:]]
    out = {}
    fn = os.environ.get('BM_OUTPUT', os.path.join(RES, 'benchmarks.json'))
    for n in names:
        out[n] = clean(record(n, inp[n]['x'], inp[n]['eta'], cmin)); print(n, 'done', flush=True)
    json.dump(out, open(fn, 'w'), indent=1, allow_nan=False)
    for n, r in out.items():
        rb = r['robustness']
        print(n, 'accepted', r['accepted'], 'chi2 %.2f (min %.2f)' % (r['chi2'], r['chi2_min']), 'failed cuts', [k for k, v in r['cuts'].items() if not v])
        print('   ', {k: '%.3g' % r['observables'][k] for k in ('mu_e_g', 'tau_mu_g', 'tau_e_g', 'Cr_Al', 'mu_3e', 'd_e', 'sum_m')},
              'BR h->tau mu %.2g' % r['collider']['BR_h_tau_mu'])
        print('    log10 Lambda', {p: (round(v['log10Lambda'], 2), v['fail']) for p, v in rb.items() if p != 'vacuum'}, 'vacuum', rb['vacuum']['status'])
