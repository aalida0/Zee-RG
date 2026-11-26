"""Campaign driver (one Condor job = one chain).  Uses the final gate (gate2/accept.py) in a STAGED, fail-fast order so that most
proposals are rejected after one 30 ms model evaluation:
  1 chi2 <= chi2_min + 3.84, penalty, hard limits      (hgmodel.evaluate)
  2 B -> X_s gamma, Drell-Yan screen                     (flavour, chargedbr + dy; no HiggsTools needed)
  3 H+-W-gamma variants (0, -1)                          (2 more evaluations)
  4 HiggsBounds / HiggsSignals / h -> tau mu             (only if the Higgs module is importable; otherwise points are marked
                                                          collider='not run' and are UNRESOLVED until re-checked locally)
  5 RG reach >= LOGL for the chain's own quartics and for quartics x1.1, x0.9 (lam_valid, coarse)
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
import sys, os, json, time, hashlib, glob, subprocess
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ.setdefault(_v, '1')   # one-core jobs
import numpy as np
os.environ.setdefault('ZEE_INPUTS', 'modern')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
from zee import hgmodel as H, flavour as FL, chargedbr as CB, dy as DY, dipfull as DF, vac
import accept as AC
import lmax2 as L
import cert as CE

HAVE_HIGGS = False; HIGGS_ERR = None
try:
    import hcoll; HAVE_HIGGS = True
except Exception as _ex:
    hcoll = None; HIGGS_ERR = '%s: %s' % (type(_ex).__name__, _ex)
if os.environ.get('ZEE_REQUIRE_HIGGS') and not HAVE_HIGGS:      # production jobs: no silent fallback to 'collider not run'
    sys.exit('HiggsTools/datasets failed to initialise (%s); refusing to run a production chain without the collider stage' % HIGGS_ERR)

SQ = np.sqrt(4*np.pi); S0 = 1e-6; ZM = np.arcsinh(AC.FMAX/S0)
LO = np.array([-ZM]*11 + [np.log(0.3), 2.0, 2.0, 0.0] + [-SQ]*4 + [-AC.ETA_MAX]*6)
HI = np.array([ZM]*11 + [np.log(50.), np.log10(2e4), np.log10(2e4), 0.999] + [SQ]*4 + [AC.ETA_MAX]*6)
SPAN = HI - LO; SPAN[:11] = 3.0
tx = lambda z: np.concatenate([S0*np.sinh(z[:11]), z[11:19]])
tz = lambda x: np.concatenate([np.arcsinh(np.clip(np.asarray(x[:11], float), -AC.FMAX, AC.FMAX)/S0), np.asarray(x[11:19], float)])
def _hash_files(files):
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(os.path.relpath(f, ROOT).encode()); h.update(open(f, 'rb').read())
    return h.hexdigest()[:16]
_CODE_FILES = [f for d in ('zee', 'gate1', 'gate2') for f in glob.glob(os.path.join(ROOT, d, '*.py'))]
_DATA_FILES = (glob.glob(os.path.join(ROOT, 'zee', '*.npz')) + glob.glob(os.path.join(ROOT, 'zee', '_nufit_cache', '*.npz')) +
               glob.glob(os.path.join(os.environ.get('ZEE_DATA', ROOT), '_nufit_cache', '*.npz')) +
               [os.path.join(ROOT, 'results', n) for n in ('refit_hw.json', 'campaign_starts.json', 'benchmarks_input.json') if os.path.exists(os.path.join(ROOT, 'results', n))])
CODE = _hash_files(_CODE_FILES); DATA = _hash_files(_DATA_FILES)      # code and input-data hashes: the collector rejects mixed values


def _has(n):
    from importlib import metadata as _md
    try: _md.version(n); return True
    except Exception: return False


def _tree_hash(path):
    """identity of a data tree: git HEAD if it is a checkout, plus a hash of all relative file names and sizes"""
    h = hashlib.sha256(); head = None
    if os.path.isdir(path):
        try: head = subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception: pass
        for r, ds, fs in sorted(os.walk(path)):
            ds[:] = sorted(d for d in ds if d != '.git')
            for f in sorted(fs): h.update(os.path.relpath(os.path.join(r, f), path).encode()); h.update(str(os.path.getsize(os.path.join(r, f))).encode())
    return dict(git=head, files=h.hexdigest()[:16], exists=os.path.isdir(path))


def manifest():
    m = dict(code=CODE, data=DATA, run_id=os.environ.get('RUN_ID', 'dev'), python=sys.version.split()[0], numpy=np.__version__)
    m['env'] = {k: os.environ.get(k) for k in ('ZEE_INPUTS', 'ZEE_SUMM_LIMIT', 'CAMP_NPROP', 'CAMP_STORE', 'ZEE_REQUIRE_HIGGS')}
    m['hw_scale'] = DF.HW_SCALE; m['higgs_init'] = HAVE_HIGGS; m['higgs_error'] = HIGGS_ERR
    m['collider_data'] = {d: _tree_hash(os.path.join(os.environ.get('ZEE_DATA', ''), d)) for d in ('hbdataset', 'hsdataset')} if os.environ.get('ZEE_DATA') else {}
    try:
        from importlib import metadata as _md
        m['packages'] = {n: _md.version(n) for n in ('numpy', 'scipy', 'pandas', 'matplotlib', 'higgstools', 'HiggsTools') if _has(n)}
    except Exception: pass
    try:
        import scipy; m['scipy'] = scipy.__version__
    except Exception: pass
    try: m['commit'] = subprocess.check_output(['git', '-C', ROOT, 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: m['commit'] = None
    try: m['dirty'] = bool(subprocess.check_output(['git', '-C', ROOT, 'status', '--porcelain'], text=True, stderr=subprocess.DEVNULL).strip())
    except Exception: m['dirty'] = None
    m['host'] = os.uname().nodename
    return m


def clean(o):
    """strict JSON: non-finite floats become null"""
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)): return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o


KEEP = ('mu_e_g', 'tau_mu_g', 'tau_e_g', 'h_tau_mu', 'd_e', 'sum_m', 'mu_3e')
CMIN = float(json.load(open(os.path.join(ROOT, 'results', 'refit_hw.json')))['best']['chi2'])


def coarse_reach(info):
    """single coarse score of a state: the minimum over the default boundary and the quartic-rescaled boundaries (x1.1, x0.9) that the gate evaluated;
    this -- not the default reach alone -- decides whether a state is inside the target of a chain"""
    return min([info['Lf']] + [info[k] for k in ('Lf_q1.1', 'Lf_q0.9') if k in info])


def in_target(info, cut):
    return coarse_reach(info) >= cut


def stage_gate(x, eta, logL, t=None, rescaled=True):
    """fail-fast gate; returns (ok, info, reason).  info carries the evaluate() result and the collider dict when run.
    rescaled=True also evaluates (and requires valid starts of) the quartic-rescaled boundaries x1.1 and x0.9; info['Lmin'] is then the coarse score
    of coarse_reach().  rescaled=False (low-energy baseline) tests only the nominal tree-level boundary."""
    if AC.in_domain(x, eta): return False, dict(), 'domain'
    e = H.evaluate(list(x), 'NO', eta=eta); sc = e['sc']; o = e['obs']; info = dict(e=e)
    # every test is written so that NaN fails it (a comparison with NaN is False)
    if not (np.isfinite(e['chi2']) and all(np.isfinite(v) for v in o.values())): return False, info, 'nonfinite'
    if not (e['chi2'] <= CMIN + AC.DCHI2) or e['pen'] != 0: return False, info, 'chi2'
    if not all(o[k] <= lim for k, (lim, z) in H.BOUNDS.items()): return False, info, 'limit'
    if not (H.yukawa_max(x, sc) < H.YUK_MAX): return False, info, 'yukawa'
    if not (abs(FL.bsg_pull(sc)) <= FL.BSG_NSIGMA): return False, info, 'bsg'
    cbr = CB.widths_and_brs(x, eta); br = cbr['BR']
    dyr = DY.decide(cbr['m'], cbr['d'], [b['enu'] for b in br], [b['munu'] for b in br], [b['taunu'] for b in br])
    info['dy'] = dict(status=dyr['status'], S_fb=dyr['S_fb'], m=list(cbr['m']), tb=[b['tb'] for b in br])
    if dyr['status'] != 'PASS': return False, info, 'dy'
    for sgn in (0.0, -1.0):
        DF.HW_SCALE = sgn
        try:
            o2 = H.evaluate(list(x), 'NO', eta=eta)['obs']
            if not all(o2[k] <= lim for k, (lim, z) in H.BOUNDS.items()): return False, info, 'hw'
        finally: DF.HW_SCALE = 1.0
    if HAVE_HIGGS:
        try: col, _ = hcoll.run(x, eta)
        except Exception as ex:                          # a numerical corner rejects the proposal and is counted, it does not abort the chain
            info['hcoll_exc'] = type(ex).__name__; return False, info, 'hcoll_error:' + type(ex).__name__
        info['col'] = col
        if not (col['HB_allowed'] and col['HS_chi2'] - AC.HSREF <= AC.DCHI2 and col['BR_h_tau_mu'] <= AC.H_TAU_MU and col['BR_h_tau_e'] <= AC.H_TAU_E):
            return False, info, 'higgs'
    c = AC.quartics(sc, eta); t0 = np.log(min(sc['m1'], sc['mA'], sc['mH']))
    Lf, why = L.lam_valid(c, t0, -sc['tb'], status=True); info['Lf'] = float(Lf); info['Lf_status'] = why
    if why in ('nonfinite', 'unitarity@start', 'BFB@start', 'yukawa@start'): return False, info, 'rg_start'
    if not (Lf >= logL): return False, info, 'cutoff'
    # prefilter against the certification variants that move the reach most: boundary quartics x1.1 and x0.9 (coarse grid, ~0.05 s each)
    for kq in ((1.1, 0.9) if rescaled else ()):
        Lk, whyk = L.lam_valid(c * kq, t0, -sc['tb'], status=True)
        info['Lf_q%.1f' % kq] = float(Lk)
        if whyk in ('nonfinite', 'unitarity@start', 'BFB@start', 'yukawa@start') or not (Lk >= logL): return False, info, 'cutoff'
    info['Lmin'] = float(coarse_reach(info))
    return True, info, 'ok'


def obs_of(info, key):
    if key == 'h_tau_mu' and 'col' in info: return info['col']['BR_h_tau_mu']
    return info['e']['obs'][key]


def full_record(x, eta, info, seed, logL=None, cs=None, cd=None):
    """complete record of a state: separate fields for each stage; final_status is PASS only if the vacuum test passed, the collider stage ran
    and the certified scale meets the requested cutoff (None when no cutoff was requested)."""
    e = info['e']; sc = e['sc']; c = AC.quartics(sc, eta)
    g = vac.global_search(c, sc, nstart=80, rng=np.random.default_rng(seed))
    obs = {k: float(e['obs'][k]) for k in KEEP}
    rec = dict(x=[float(v) for v in x], eta=[float(v) for v in eta], chi2=float(e['chi2']), Lf=info['Lf'], Lf_status=info['Lf_status'], Lmin=info.get('Lmin'), obs=obs,
               masses=dict(m1=float(sc['m1']), m2=float(sc['m2']), mA=float(sc['mA']), mH=float(sc['mH']),
               tb=float(sc['tb']), cba=float(sc['cba']), phi=float(sc['phi'])), dy=info['dy'], vacuum=g['status'], vacuum_status=g['status'],
               low_energy_status='PASS', collider=('run' if 'col' in info else 'not run'))
    obs['h_tau_mu_fit'] = obs['h_tau_mu']                                # approximate fit-model value, kept for reference
    if 'col' in info:
        rec['col'] = dict(HB=bool(info['col']['HB_allowed']), HS=float(info['col']['HS_chi2']), h_tau_mu=float(info['col']['BR_h_tau_mu']))
        obs['h_tau_mu'] = float(info['col']['BR_h_tau_mu'])               # canonical value (model total width) used for objective, ranking, claims
    rc = CE.reach_class(cs, cd) if cs is not None else None            # the shared classifier (gate2/cert.py): a numerical stop, a borderline BFB margin or a failed matching never counts as a valid reach
    if cs is not None: rec['cert'] = dict(min=float(cs), variants=cd, reach_class=rc)
    base = 'PASS' if (g['status'] == 'ok' and 'col' in info) else ('UNRESOLVED' if g['status'] == 'ok' else 'FAIL')
    rec['rg_status'], rec['baseline'], rec['meets_requested_cutoff'], rec['final_status'] = CE.status_fields(cs, rc, logL, base)
    rec['cutoff_requested'] = logL; rec['status'] = base
    return rec


def starts():
    """validated diverse starts (results/campaign_starts.json, built by gate2/make_starts.py + select_starts.py); BM1/BM2 as fallback"""
    f = os.path.join(ROOT, 'results', 'campaign_starts.json')
    src = json.load(open(f)) if os.path.exists(f) else list(json.load(open(os.path.join(ROOT, 'results', 'benchmarks_input.json'))).values())
    return [(np.array(v['x']), np.array(v['eta'])) for v in src]


NPROP = int(os.environ.get('CAMP_NPROP', 8000))         # proposals per chain (fixed length: the stopping rule does not depend on the state); local pilot: ~0.1 s per proposal incl. gate, so ~15 min inside the 23-min wall
STORE_PROP = int(os.environ.get('CAMP_STORE', 150))     # a store event every STORE_PROP post-burn-in PROPOSALS (accepted or not)
NMAX = 400; BURN = 0.25                                 # burn-in = first BURN*NPROP proposals


if __name__ == '__main__':
    mode, logL, seed, wall, out = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    si = int(sys.argv[6]) if len(sys.argv) > 6 else seed
    os.makedirs(out, exist_ok=True); rng = np.random.default_rng(1000 + seed)
    MAN = manifest()
    fn = os.path.join(out, 'camp_%s_%s_%g_%d.json' % (MAN['run_id'], mode.replace(':', '-'), logL, seed))
    if os.path.exists(fn):
        try:
            if json.load(open(fn)).get('complete'): sys.exit('output exists and is complete: ' + fn)
        except Exception: pass
    key = mode.split(':')[1] if mode.startswith('max:') else None
    S = starts(); si = si % len(S); x0, e0 = S[si]; zc = np.concatenate([tz(x0), e0])
    MARGIN = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0       # maximiser: the walk uses logL + MARGIN as a prefilter (0 = the requested cutoff itself)
    RESC = logL >= 0                                                 # logL < 0: low-energy baseline, nominal boundary only (no rescaled-boundary requirement)
    ok, info, why = stage_gate(x0, e0, -1.0, rescaled=RESC)
    if not ok: sys.exit('start fails the gate: %s' % why)
    walk_cut = logL + (MARGIN if key else 0.0)
    Lcur = coarse_reach(info); climb = Lcur < walk_cut              # a start below the cutoff (in the coarse score of ALL boundary variants) first climbs (burn-in)
    cur = obs_of(info, key) if key else 0.0; step = 0.01; n = acc = 0; nacc_win = nwin = 0; acc_post = n_post = 0
    Lc = None; m_n = 0; m_mu = np.zeros(25); m_M2 = np.zeros((25, 25)); best = None; best_unres = None; rej = {}; t0 = time.time(); tlast = t0; ncert = 0; tcert = 0.0
    cur_inf = info; cur_id = 0; recs = {}; trace = []; complete = False; bests = []; n_enter = None

    def make_rec(z_, inf_, rid):
        global ncert, tcert
        x_, e_ = tx(z_), z_[19:]; tt = time.time(); cs_, cd_ = CE.cert(x_, e_); tcert += time.time() - tt; ncert += 1
        return full_record(x_, e_, inf_, seed*100000 + rid, logL, cs_, cd_)

    def dump():
        pts = []
        for k_ in sorted(recs):
            r_ = dict(recs[k_]); r_['id'] = k_; r_['dwell'] = trace.count(k_); pts.append(r_)
        tmp = fn + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(clean(dict(mode=mode, logL=logL, seed=seed, start=si, code=CODE, manifest=MAN, cmin=CMIN, higgs=HAVE_HIGGS, complete=complete,
                       n_prop=n, n_acc=acc, nprop_target=NPROP, store_every_proposals=STORE_PROP, step=step, burn=BURN, acc_post=acc_post, n_post=n_post, n_enter=n_enter, climb_at_end=bool(climb),
                       n_cert=ncert, t_cert=tcert, wall=time.time()-t0, rejects=rej, points=pts, trace=trace, best=best, best_unresolved=best_unres, best_archive=bests, margin=MARGIN, rescaled=RESC,
                       measure=('uniform in z on the region R_L (cheap gate + coarse cutoff); Yukawas in arcsinh(x/1e-6); stored = the current state '
                                'every %d proposals, repeated states carry dwell counts' % STORE_PROP) if not key else 'optimiser (no stationary measure)')),
                      fh, allow_nan=False)
        os.replace(tmp, fn)

    def consider_best(zz, inf_, rid):
        """certify a candidate best and keep it (PASS -> best, UNRESOLVED -> best_unres); the last 8 improvements are archived in `bests`"""
        global ncert, tcert, best, best_unres
        tt = time.time(); cs_, cd_ = CE.cert(tx(zz), zz[19:]); tcert += time.time() - tt; ncert += 1
        if cs_ < logL: return
        r_ = full_record(tx(zz), zz[19:], inf_, seed*100000 + rid, logL, cs_, cd_)
        if r_['final_status'] == 'PASS': best = r_; bests.append(r_); del bests[:-8]
        elif r_['final_status'] == 'UNRESOLVED': best_unres = r_
        else: rej['best_failed_' + r_['vacuum']] = rej.get('best_failed_' + r_['vacuum'], 0) + 1

    tried_obs = 0.0; tried_acc = -10**9                    # observable value / accepted-move index of the last certification attempt of ANY status (throttle, see below)
    if key and (not climb or coarse_reach(info) >= logL): consider_best(zc, info, 0); tried_obs = obs_of(info, key); tried_acc = 0   # a valid start is a candidate best before any proposal is accepted

    n_enter = None if climb else 0                       # proposal index at which the chain first stands inside the walk target (0: valid from the start, None: never)
    while n < NPROP and time.time() - t0 < wall:
        burn = climb or n < BURN*NPROP
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
            ok, inf, why = stage_gate(tx(zn), zn[19:], -1.0 if climb else walk_cut, rescaled=RESC)
            if not ok: rej[why] = rej.get(why, 0) + 1
        take = ok
        if ok and climb: take = coarse_reach(inf) >= Lcur                          # until the cutoff is reached only the cutoff is optimised
        elif ok and key:
            v = obs_of(inf, key); take = v >= cur*(1 - 0.05*rng.random())
        nwin += 1
        if take:
            zc = zn; acc += 1; nacc_win += 1; cur_id += 1; cur_inf = inf
            if climb:
                Lcur = coarse_reach(inf); climb = Lcur < walk_cut
                if not climb and n_enter is None: n_enter = n
            if not burn: acc_post += 1
            if key and (not climb or coarse_reach(inf) >= logL):        # a state that already meets the REQUESTED cutoff is a candidate even if the optional extra-decade walk margin is not yet reached
                if not climb: cur = obs_of(inf, key)
                v_ = obs_of(inf, key)
                # valid and unresolved tracks are independent: an unresolved high-rate point must not hide a lower, fully valid improvement.
                # Throttle (pilot 1: a chain without a PASS best certified every accepted move, 151 certs / 537 s, 137 failed): after an attempt of any status, certify again
                # only if the observable rose by >30% or 25 further moves were accepted; certified bests are unchanged, only the attempt cadence is.
                if v_ > (best['obs'][key] if best else 0.0)*1.02 and (v_ > 1.3*tried_obs or acc - tried_acc >= 25):
                    consider_best(zn, inf, acc); tried_obs = v_; tried_acc = acc
        if not key and not burn:                         # proposal-indexed storage: rejected proposals repeat the current state
            n_post += 1
            if n_post % STORE_PROP == 0 and len(trace) < NMAX:
                if cur_id not in recs: recs[cur_id] = make_rec(zc, cur_inf, cur_id)
                trace.append(cur_id)
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
    complete = n >= NPROP
    dump()
    print('done mode %s logL %g seed %d start %d: %d proposals (%s), %d accepted (post-burn %d/%d), %d store events, %d unique records, %d certs (%.0fs), rejects %s, higgs=%s' %
          (mode, logL, seed, si, n, 'complete' if complete else 'INCOMPLETE: wall limit', acc, acc_post, n_post, len(trace), len(recs), ncert, tcert, rej, HAVE_HIGGS))
