"""The single acceptance gate used by screen.py, envlocal.py, bm_records.py and figs2.py (referee C06).
The statistic is hgmodel.evaluate(x, eta)['chi2'] (the free quartics eta enter R_gamma_gamma and the Barr-Zee
charged-scalar loops), the same number is stored and reported. Every individual cut is recorded.
accept(x, eta, cmin, ...) -> dict(accepted=bool, chi2, cuts={...}, obs, collider, Lf, Lf_status, vacuum)"""
import sys, numpy as np
sys.path.insert(0, '/home/claude/zee_fit'); sys.path.insert(0, '/home/claude/zee_fit/gate1'); sys.path.insert(0, '/home/claude/zee_fit/gate2')
from zee import hgmodel as H, vac
import lmax2 as L

DCHI2 = 3.84; HSMIN = 151.65; H_TAU_MU, H_TAU_E = 0.0018, 0.0020


def quartics(sc, eta):
    c = np.zeros(11); c[0], c[2], c[3], c[4], c[5] = sc['lam1'], sc['lam3'], sc['lam4'], sc['lam5'], sc['lam6']
    c[1], c[6:] = eta[0], eta[1:]
    return c


def accept(x, eta, cmin, logL=None, collider=True, n_vac=80, seed=11, stop_early=False):
    x = np.asarray(x, float); eta = np.asarray(eta, float)
    e = H.evaluate(list(x), 'NO', eta=eta); sc = e['sc']; o = e['obs']
    cuts = {'chi2': e['chi2'] <= cmin + DCHI2, 'penalty': e['pen'] == 0}
    cuts.update({'lim_' + k: bool(o[k] <= lim) for k, (lim, z) in H.BOUNDS.items()})
    rec = dict(chi2=float(e['chi2']), chi2_parts=dict(osc=e['chi_osc'], univ=e['chi_univ'], stu=e['chi_stu'], higgs=e['chi_higgs'],
               **{k: float(v) for k, v in e['chi_bounds'].items()}), obs={k: float(v) for k, v in o.items()}, sc=sc, cuts=cuts)
    if stop_early and not all(cuts.values()): rec['accepted'] = False; return rec
    if collider:
        import hcoll
        col, _ = hcoll.run(x, eta); rec['collider'] = col
        cuts.update(HB=bool(col['HB_allowed']), HS=bool(col['HS_chi2'] - HSMIN <= DCHI2),
                    h_tau_mu=bool(col['BR_h_tau_mu'] <= H_TAU_MU), h_tau_e=bool(col['BR_h_tau_e'] <= H_TAU_E))
        if stop_early and not all(cuts.values()): rec['accepted'] = False; return rec
    c = quartics(sc, eta); t0 = np.log(min(sc['m1'], sc['mA'], sc['mH']))
    Lf, why = L.lam_valid(c, t0, -sc['tb'], status=True); rec.update(Lf=float(Lf), Lf_status=why)
    cuts['rg_start'] = Lf_ok = why not in ('nonfinite', 'unitarity@start', 'BFB@start')
    if logL is not None: cuts['cutoff'] = bool(Lf >= logL)
    g = vac.global_search(c, sc, nstart=n_vac, rng=np.random.default_rng(seed))
    rec['vacuum'] = {k: g[k] for k in ('status', 'n', 'n_conv', 'n_fail', 'deepest', 'V_EW', 'ew_grad')}
    cuts['vacuum'] = g['status'] == 'ok'
    rec['accepted'] = all(cuts.values())
    return rec
