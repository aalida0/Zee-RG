"""The single acceptance gate used by screen.py, envlocal.py, bm_records.py, reopt_bench.py and benchmarks.py.
The statistic is hgmodel.evaluate(x, eta)['chi2'] (the free quartics eta enter R_gamma_gamma and the Barr-Zee
charged-scalar loops); the same number is stored and reported. Every individual cut is recorded.

accept(x, eta, cmin, ...) -> dict(status in PASS/FAIL/UNRESOLVED, accepted (== PASS), chi2, cuts, unresolved, ...)
Fail-closed rules:
  * a point outside the declared parameter domain FAILS;
  * if the collider tests are not run (collider=False or the Higgs module is missing) the status is UNRESOLVED, never PASS;
  * the Drell-Yan screen returns FAIL / UNRESOLVED / PASS (zee/dy.py);
  * B -> X_s gamma is a hard cut (2 sigma around the HFLAV average, zee/flavour.py);
  * `accepted` is the conjunction of every cut that was evaluated and requires an empty `unresolved` list.
  * `validity`: the RG cutoff cut exists only if logL is given; without it the record states validity='not certified'."""
import sys, numpy as np
_R = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, _R + '/gate1'); sys.path.insert(0, _R + '/gate2')
from zee import hgmodel as H, vac, flavour as FL, chargedbr as CB, dy as DY
import lmax2 as L

DCHI2 = 3.84; HSMIN = 151.65; H_TAU_MU, H_TAU_E = 0.0018, 0.0020
SQ4PI = np.sqrt(4*np.pi)
FMAX = 0.1                                   # |f|, |Y2| components (declared domain of the paper)
ETA_MAX = 3.0                                # |free quartics|


def in_domain(x, eta):
    """Declared parameter domain; returns list of violated items (empty = inside)."""
    x = np.asarray(x, float); eta = np.asarray(eta, float); bad = []
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(eta)): return ['nonfinite']
    if np.abs(x[:11]).max() > FMAX*(1 + 1e-9): bad.append('yukawa>%g' % FMAX)
    if not (np.log(0.3) - 1e-9 <= x[11] <= np.log(50.) + 1e-9): bad.append('tanb')
    if not (2 - 1e-9 <= x[12] <= np.log10(2e4) + 1e-9): bad.append('MHp')
    if not (2 - 1e-9 <= x[13] <= np.log10(2e4) + 1e-9): bad.append('M33')
    if not (0 <= x[14] <= 0.999 + 1e-9): bad.append('u')
    if np.abs(x[15:19]).max() > SQ4PI*(1 + 1e-9): bad.append('lambda3-6>sqrt(4pi)')
    if np.abs(eta).max() > ETA_MAX*(1 + 1e-9): bad.append('eta>%g' % ETA_MAX)
    return bad


def quartics(sc, eta):
    c = np.zeros(11); c[0], c[2], c[3], c[4], c[5] = sc['lam1'], sc['lam3'], sc['lam4'], sc['lam5'], sc['lam6']
    c[1], c[6:] = eta[0], eta[1:]
    return c


def accept(x, eta, cmin, logL=None, collider=True, n_vac=80, seed=11, stop_early=False):
    x = np.asarray(x, float); eta = np.asarray(eta, float)
    unresolved = []; dom = in_domain(x, eta)
    cuts = {'domain': not dom}
    if dom:
        return dict(status='FAIL', accepted=False, chi2=float('nan'), cuts=cuts, domain_violations=dom, unresolved=unresolved)
    e = H.evaluate(list(x), 'NO', eta=eta); sc = e['sc']; o = e['obs']
    cuts.update({'chi2': e['chi2'] <= cmin + DCHI2, 'penalty': e['pen'] == 0})
    cuts.update({'lim_' + k: bool(o[k] <= lim) for k, (lim, z) in H.BOUNDS.items()})
    bpull = FL.bsg_pull(sc); cuts['bsgamma'] = bool(abs(bpull) <= FL.BSG_NSIGMA)
    rec = dict(chi2=float(e['chi2']), chi2_parts=dict(osc=e['chi_osc'], univ=e['chi_univ'], stu=e['chi_stu'], higgs=e['chi_higgs'],
               **{k: float(v) for k, v in e['chi_bounds'].items()}), obs={k: float(v) for k, v in o.items()}, sc=sc, cuts=cuts,
               bsgamma_e4=float(FL.bsg_e4(sc)), unresolved=unresolved, validity='not certified (no cutoff requested)' if logL is None else 'cutoff requested')
    def fin(status_override=None):
        failed = [k for k, v in cuts.items() if not v]
        rec['status'] = 'FAIL' if failed else ('UNRESOLVED' if unresolved else 'PASS')
        rec['accepted'] = rec['status'] == 'PASS'; rec['failed_cuts'] = failed
        return rec
    # Drell-Yan screen: needs only the charged-scalar branching ratios (zee/chargedbr.py), so it runs without HiggsTools
    cbr = CB.widths_and_brs(x, eta); br = cbr['BR']
    dyr = DY.decide(cbr['m'], cbr['d'], [b['enu'] for b in br], [b['munu'] for b in br], [b['taunu'] for b in br])
    rec['DY'] = dict(status=dyr['status'], S_fb=dyr['S_fb'], states=dyr['states'], widths=cbr['width'], BR=br)
    if dyr['status'] == 'UNRESOLVED': unresolved.append('DY screen unresolved')
    else: cuts['DY'] = dyr['status'] == 'PASS'
    if stop_early and not all(cuts.values()): return fin()
    if collider:
        try:
            import hcoll
        except Exception as ex:
            unresolved.append('collider module unavailable: %s' % type(ex).__name__); hcoll = None
        if hcoll is not None:
            col, _ = hcoll.run(x, eta); rec['collider'] = col
            cuts.update(HB=bool(col['HB_allowed']), HS=bool(col['HS_chi2'] - HSMIN <= DCHI2),
                        h_tau_mu=bool(col['BR_h_tau_mu'] <= H_TAU_MU), h_tau_e=bool(col['BR_h_tau_e'] <= H_TAU_E))
    else:
        unresolved.append('collider tests not run (HB/HS/Higgs LFV)')
    if stop_early and not all(cuts.values()): return fin()
    c = quartics(sc, eta); t0 = np.log(min(sc['m1'], sc['mA'], sc['mH']))
    Lf, why = L.lam_valid(c, t0, -sc['tb'], status=True); rec.update(Lf=float(Lf), Lf_status=why)
    cuts['rg_start'] = why not in ('nonfinite', 'unitarity@start', 'BFB@start')
    if logL is not None: cuts['cutoff'] = bool(Lf >= logL)
    g = vac.global_search(c, sc, nstart=n_vac, rng=np.random.default_rng(seed))
    rec['vacuum'] = {k: g[k] for k in ('status', 'n', 'n_conv', 'n_fail', 'deepest', 'V_EW', 'ew_grad')}
    cuts['vacuum'] = g['status'] == 'ok'
    return fin()
