import os, json, subprocess, sys, glob
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
import campaign as C


def test_starts_pass_staged_gate_and_cutoff_matters():
    for x, eta in C.starts():
        ok, info, why = C.stage_gate(x, eta, -1.0); assert ok, why
        ok2, _, why2 = C.stage_gate(x, eta, 50.0); assert not ok2 and why2 == 'cutoff'


def test_staged_gate_rejects_bad_chi2_first():
    x, eta = C.starts()[0]; x = x.copy(); x[11] = np.log(40.0)      # large tan(beta): far outside the fit region
    ok, info, why = C.stage_gate(x, eta, -1.0); assert not ok and why in ('chi2', 'limit', 'bsg', 'dy')


def test_short_chain_writes_valid_json(tmp_path):
    p = subprocess.run([sys.executable, '-W', 'ignore', os.path.join(ROOT, 'gate2', 'campaign.py'), 'sample', '6', '1', '6', str(tmp_path), '0'],
                       capture_output=True, text=True, env=dict(os.environ, ZEE_INPUTS='modern'))
    assert p.returncode == 0, p.stderr[-500:]
    d = json.load(open(glob.glob(str(tmp_path / 'camp_*.json'))[0]))
    assert d['n_prop'] > 20 and d['cmin'] > 9 and all(q['status'] in ('PASS', 'UNRESOLVED') for q in d['points'])


def test_rhat_and_cert_helpers():
    import collect_campaign as CC
    rng = np.random.default_rng(0)
    same = [rng.normal(size=200) for _ in range(4)]; diff = [rng.normal(size=200) + 3*i for i in range(4)]
    assert CC.rhat(same) < 1.05 and CC.rhat(diff) > 1.5
    assert CC.act(rng.normal(size=400)) < 2.0
    import cert as CE
    for (x, eta), ref in zip(C.starts(), (11.6, 12.07)):
        assert abs(CE.cert(x, eta)[0] - ref) < 0.1           # reproduces the certified scales of benchmarks.py (BM1, BM2)


def _strict(txt):
    return json.loads(txt, parse_constant=lambda c: (_ for _ in ()).throw(ValueError('non-finite constant ' + c)))


def test_proposal_indexed_storage_and_strict_json(tmp_path):
    """The chain stores its current state every CAMP_STORE PROPOSALS (rejections repeat the state, dwell counts add up) in strict JSON with a manifest."""
    env = dict(os.environ, ZEE_INPUTS='modern', CAMP_NPROP='400', CAMP_STORE='20', RUN_ID='unit')
    p = subprocess.run([sys.executable, '-W', 'ignore', os.path.join(ROOT, 'gate2', 'campaign.py'), 'sample', '6', '2', '300', str(tmp_path), '0'],
                       capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr[-500:]
    fs = glob.glob(str(tmp_path / 'camp_unit_*.json')); assert len(fs) == 1
    d = _strict(open(fs[0]).read())
    assert d['complete'] and d['n_prop'] == 400 and len(d['trace']) == (400 - 100)//20
    assert sum(q['dwell'] for q in d['points']) == len(d['trace']) and {q['id'] for q in d['points']} == set(d['trace'])
    assert len(d['points']) < len(d['trace'])           # at ~20% acceptance and 20-proposal spacing some stored states are repeats
    assert d['manifest']['code'] == d['code'] and d['manifest']['run_id'] == 'unit'
    q = subprocess.run([sys.executable, '-W', 'ignore', os.path.join(ROOT, 'gate2', 'campaign.py'), 'sample', '6', '2', '300', str(tmp_path), '0'],
                       capture_output=True, text=True, env=env)
    assert 'complete' in (q.stdout + q.stderr)          # a completed output is not overwritten


def test_nonfinite_state_is_rejected_not_accepted():
    x, eta = C.starts()[0]; x = x.copy(); x[12] = np.nan
    ok, info, why = C.stage_gate(x, eta, -1.0); assert not ok


def test_cert_reports_start_failure():
    import cert as CE
    x, eta = C.starts()[0]; sc, c = CE.bench.coup(x, eta)
    bad = c.copy(); bad[0] = 50.0                                   # non-perturbative at the boundary
    v = CE.run(bad, np.log(min(sc['m1'], sc['mA'], sc['mH'])), -sc['tb'], CE.G0)
    assert v[0] < 0 and v[1].startswith('start:')


def test_coarse_reach_is_minimum_over_boundary_variants():
    """report v8 P1: a state with default reach 13 but quartic-rescaled reach 10 is NOT inside a target of 12"""
    info = {'Lf': 13.0, 'Lf_q1.1': 10.0, 'Lf_q0.9': 14.0}
    assert C.coarse_reach(info) == 10.0 and not C.in_target(info, 12.0) and C.in_target(info, 10.0)
    assert C.coarse_reach({'Lf': 9.0}) == 9.0


def test_stage_gate_enforces_all_boundary_variants_for_target():
    """BM1: default coarse reach 11.75 but x1.1 reach 10.75: target 11 must reject (previously accepted at the start), target 10.5 accepts;
    the low-energy baseline (rescaled=False) evaluates the nominal boundary only"""
    x, eta = C.starts()[0]
    ok, info, why = C.stage_gate(x, eta, 11.0); assert not ok and why == 'cutoff'
    ok, info, why = C.stage_gate(x, eta, 10.5); assert ok and abs(info['Lmin'] - C.coarse_reach(info)) < 1e-12 and info['Lmin'] < info['Lf']
    ok, info, why = C.stage_gate(x, eta, -1.0, rescaled=False); assert ok and 'Lf_q1.1' not in info and info['Lmin'] == info['Lf']
