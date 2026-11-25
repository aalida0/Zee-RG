"""Campaign driver (one Condor job = one chain).  Uses the final gate (gate2/accept.py) in a STAGED, fail-fast order so that most
proposals are rejected after one 30 ms model evaluation:
  1 chi2 <= chi2_min + 3.84, penalty, hard limits      (hgmodel.evaluate)
  2 B -> X_s gamma, Drell-Yan screen                     (flavour, chargedbr + dy; no HiggsTools needed)
  3 H+-W-gamma variants (0, -1)                          (2 more evaluations)
  4 HiggsBounds / HiggsSignals / h -> tau mu             (only if the Higgs module is importable; otherwise points are marked
                                                          collider='not run' and are UNRESOLVED until re-checked locally)
  5 RG reach Lf >= LOGL for the chain's own quartics     (lam_valid)
The vacuum test (80 starts) and the full record are done only for the points that are STORED.
Modes:  sample   Metropolis random walk, symmetric Gaussian proposals, proposals outside the parameter box are REJECTED (no clipping).
                 Stationary measure: uniform in the sampling coordinates z (Yukawas in arcsinh(x/1e-6), the other 14 parameters linear)
                 restricted to the region R_L = {cheap gate, coarse default-input cutoff >= LOGL}.  The step adapts only during the first
                 25% of the wall time (burn-in, discarded); afterwards it is fixed, so the chain is a valid MCMC.  Points are stored every
                 STORE_EVERY accepted steps and each stored point is certified (gate2/cert.py: minimum over matching variants).
        max:OBS  hill climb on OBS in {mu_e_g, tau_mu_g, tau_e_g, h_tau_mu} with a 5% annealing tolerance; the walk uses cutoff LOGL+1 as a
                 prefilter and a new best point is recorded only if its certified scale (5 matching variants) is >= LOGL
Usage:  ZEE_INPUTS=modern python3 -W ignore gate2/campaign.py MODE LOGL SEED WALL OUTDIR [START_INDEX]
Output: OUTDIR/camp_<mode>_<LOGL>_<SEED>.json  (points with status, stats, timings, code hash)."""
import sys, os, json, time, hashlib, glob
import numpy as np
os.environ.setdefault('ZEE_INPUTS', 'modern')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
from zee import hgmodel as H, flavour as FL, chargedbr as CB, dy as DY, dipfull as DF, vac
import accept as AC
import lmax2 as L
import cert as CE

HAVE_HIGGS = False
try:
    import hcoll; HAVE_HIGGS = True
except Exception:
    hcoll = None

SQ = np.sqrt(4*np.pi); S0 = 1e-6; ZM = np.arcsinh(AC.FMAX/S0)
LO = np.array([-ZM]*11 + [np.log(0.3), 2.0, 2.0, 0.0] + [-SQ]*4 + [-AC.ETA_MAX]*6)
HI = np.array([ZM]*11 + [np.log(50.), np.log10(2e4), np.log10(2e4), 0.999] + [SQ]*4 + [AC.ETA_MAX]*6)
SPAN = HI - LO; SPAN[:11] = 3.0
tx = lambda z: np.concatenate([S0*np.sinh(z[:11]), z[11:19]])
tz = lambda x: np.concatenate([np.arcsinh(np.clip(np.asarray(x[:11], float), -AC.FMAX, AC.FMAX)/S0), np.asarray(x[11:19], float)])
CODE = hashlib.sha256(b''.join(open(f, 'rb').read() for f in sorted(glob.glob(os.path.join(ROOT, 'zee', '*.py'))) +
                               [os.path.join(ROOT, 'gate2', 'accept.py')])).hexdigest()[:16]
KEEP = ('mu_e_g', 'tau_mu_g', 'tau_e_g', 'h_tau_mu', 'd_e', 'sum_m', 'mu_3e')
CMIN = float(json.load(open(os.path.join(ROOT, 'results', 'refit_hw.json')))['best']['chi2'])


def stage_gate(x, eta, logL, t=None):
    """fail-fast gate; returns (ok, info, reason).  info carries the evaluate() result and the collider dict when run."""
    e = H.evaluate(list(x), 'NO', eta=eta); sc = e['sc']; o = e['obs']; info = dict(e=e)
    if e['chi2'] > CMIN + AC.DCHI2 or e['pen'] != 0: return False, info, 'chi2'
    if any(o[k] > lim for k, (lim, z) in H.BOUNDS.items()): return False, info, 'limit'
    if abs(FL.bsg_pull(sc)) > FL.BSG_NSIGMA: return False, info, 'bsg'
    cbr = CB.widths_and_brs(x, eta); br = cbr['BR']
    dyr = DY.decide(cbr['m'], cbr['d'], [b['enu'] for b in br], [b['munu'] for b in br], [b['taunu'] for b in br])
    info['dy'] = dict(status=dyr['status'], S_fb=dyr['S_fb'], m=list(cbr['m']), tb=[b['tb'] for b in br])
    if dyr['status'] != 'PASS': return False, info, 'dy'
    for sgn in (0.0, -1.0):
        DF.HW_SCALE = sgn
        try:
            o2 = H.evaluate(list(x), 'NO', eta=eta)['obs']
            if any(o2[k] > lim for k, (lim, z) in H.BOUNDS.items()): return False, info, 'hw'
        finally: DF.HW_SCALE = 1.0
    if HAVE_HIGGS:
        col, _ = hcoll.run(x, eta); info['col'] = col
        if not (col['HB_allowed'] and col['HS_chi2'] - AC.HSMIN <= AC.DCHI2 and col['BR_h_tau_mu'] <= AC.H_TAU_MU and col['BR_h_tau_e'] <= AC.H_TAU_E):
            return False, info, 'higgs'
    c = AC.quartics(sc, eta); t0 = np.log(min(sc['m1'], sc['mA'], sc['mH']))
    Lf, why = L.lam_valid(c, t0, -sc['tb'], status=True); info['Lf'] = float(Lf); info['Lf_status'] = why
    if why in ('nonfinite', 'unitarity@start', 'BFB@start'): return False, info, 'rg_start'
    if Lf < logL: return False, info, 'cutoff'
    return True, info, 'ok'


def obs_of(info, key):
    if key == 'h_tau_mu' and 'col' in info: return info['col']['BR_h_tau_mu']
    return info['e']['obs'][key]


def full_record(x, eta, info, seed):
    e = info['e']; sc = e['sc']; c = AC.quartics(sc, eta)
    g = vac.global_search(c, sc, nstart=80, rng=np.random.default_rng(seed))
    rec = dict(x=[float(v) for v in x], eta=[float(v) for v in eta], chi2=float(e['chi2']), Lf=info['Lf'], Lf_status=info['Lf_status'],
               obs={k: float(e['obs'][k]) for k in KEEP}, masses=dict(m1=float(sc['m1']), m2=float(sc['m2']), mA=float(sc['mA']), mH=float(sc['mH']),
               tb=float(sc['tb']), cba=float(sc['cba']), phi=float(sc['phi'])), dy=info['dy'], vacuum=g['status'],
               collider=('run' if 'col' in info else 'not run'))
    if 'col' in info: rec['col'] = dict(HB=bool(info['col']['HB_allowed']), HS=float(info['col']['HS_chi2']), h_tau_mu=float(info['col']['BR_h_tau_mu']))
    rec['status'] = 'PASS' if (g['status'] == 'ok' and 'col' in info) else ('UNRESOLVED' if g['status'] == 'ok' else 'FAIL')
    return rec


def starts():
    """validated diverse starts (results/campaign_starts.json, built by gate2/make_starts.py + select_starts.py); BM1/BM2 as fallback"""
    f = os.path.join(ROOT, 'results', 'campaign_starts.json')
    src = json.load(open(f)) if os.path.exists(f) else list(json.load(open(os.path.join(ROOT, 'results', 'benchmarks_input.json'))).values())
    return [(np.array(v['x']), np.array(v['eta'])) for v in src]


STORE_EVERY = 40; NMAX = 400; BURN = 0.25


if __name__ == '__main__':
    mode, logL, seed, wall, out = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    si = int(sys.argv[6]) if len(sys.argv) > 6 else seed
    os.makedirs(out, exist_ok=True); rng = np.random.default_rng(1000 + seed)
    key = mode.split(':')[1] if mode.startswith('max:') else None
    S = starts(); si = si % len(S); x0, e0 = S[si]; zc = np.concatenate([tz(x0), e0])
    ok, info, why = stage_gate(x0, e0, -1.0)
    if not ok: sys.exit('start fails the gate: %s' % why)
    walk_cut = logL + (1.0 if key else 0.0); climb = info['Lf'] < walk_cut; Lcur = info['Lf']     # a start below the cutoff first climbs (counts as burn-in)
    cur = obs_of(info, key) if key else 0.0; step = 0.01; n = acc = 0; nacc_win = nwin = 0; acc_post = n_post = 0
    Lc = None; m_n = 0; m_mu = np.zeros(25); m_M2 = np.zeros((25, 25)); best = None; stored = []; rej = {}; t0 = time.time(); tlast = t0; ncert = 0; tcert = 0.0
    def dump():
        json.dump(dict(mode=mode, logL=logL, seed=seed, start=si, code=CODE, cmin=CMIN, higgs=HAVE_HIGGS, n_prop=n, n_acc=acc, step=step, burn=BURN,
                       acc_post=acc_post, n_post=n_post, n_cert=ncert, t_cert=tcert, wall=time.time()-t0, rejects=rej, points=stored, best=best,
                       measure='uniform in z on the region R_L; Yukawas in arcsinh(x/1e-6)' if not key else 'optimiser'),
                  open(os.path.join(out, 'camp_%s_%g_%d.json' % (mode.replace(':', '-'), logL, seed)), 'w'))
    while time.time() - t0 < wall:
        burn = climb or (time.time() - t0) < BURN*wall
        if Lc is None:                                   # isotropic start-up proposal until a covariance is available
            mask = rng.random(25) < 0.3
            if not mask.any(): mask[rng.integers(25)] = True
            zn = zc + rng.normal(size=25)*SPAN*step*mask
        else:                                            # adaptive-Metropolis proposal: learned covariance (frozen after burn-in), scale `step`
            zn = zc + step*(Lc @ rng.normal(size=25))
        n += 1
        if np.any(zn < LO) or np.any(zn > HI):
            rej['box'] = rej.get('box', 0) + 1; ok = False
        else:
            ok, inf, why = stage_gate(tx(zn), zn[19:], -1.0 if climb else walk_cut)
            if not ok: rej[why] = rej.get(why, 0) + 1
        take = ok
        if ok and climb: take = inf['Lf'] >= Lcur
        if ok and key:
            v = obs_of(inf, key); take = v >= cur*(1 - 0.05*rng.random())
        if not burn: n_post += 1
        nwin += 1
        if take:
            zc = zn; acc += 1; nacc_win += 1
            if climb:
                Lcur = inf['Lf']; climb = Lcur < walk_cut
            if not burn: acc_post += 1
            if key:
                cur = obs_of(inf, key)
                if cur > (best['obs'][key] if best else 0.0)*1.02:
                    tt = time.time(); cs, cd = CE.cert(tx(zn), zn[19:]); tcert += time.time() - tt; ncert += 1
                    if cs >= logL:
                        r = full_record(tx(zn), zn[19:], inf, seed*100000 + acc); r['cert'] = dict(min=cs, variants=cd); best = r
            elif not burn and acc_post % STORE_EVERY == 0 and len(stored) < NMAX:
                tt = time.time(); cs, cd = CE.cert(tx(zn), zn[19:]); tcert += time.time() - tt; ncert += 1
                r = full_record(tx(zn), zn[19:], inf, seed*100000 + acc); r['cert'] = dict(min=cs, variants=cd); stored.append(r)
        if burn:                                         # running mean/covariance of the chain states (burn-in only)
            m_n += 1; dz = zc - m_mu; m_mu = m_mu + dz/m_n; m_M2 += np.outer(dz, zc - m_mu)
            if m_n >= 300 and m_n % 100 == 0:
                C_ = m_M2/(m_n - 1) + 1e-10*np.diag(SPAN**2)
                try:
                    L_new = np.linalg.cholesky(C_)
                    if Lc is None: step = 2.38/np.sqrt(25)
                    Lc = L_new
                except np.linalg.LinAlgError: pass
        if burn and nwin == 50:
            r_ = nacc_win/50
            if Lc is None: step = float(np.clip(step*(1.3 if r_ > 0.3 else 0.8 if r_ < 0.1 else 1.0), 2e-4, 0.1))
            else: step = float(np.clip(step*(1.2 if r_ > 0.3 else 0.8 if r_ < 0.15 else 1.0), 1e-3, 2.4/np.sqrt(25)*2))
            nwin = nacc_win = 0
        if time.time() - tlast > 120: dump(); tlast = time.time()
    dump()
    print('done mode %s logL %g seed %d start %d: %d proposals, %d accepted (post-burn %d/%d), %d stored, %d certs (%.0fs), rejects %s, higgs=%s' %
          (mode, logL, seed, si, n, acc, acc_post, n_post, len(stored), ncert, tcert, rej, HAVE_HIGGS))
