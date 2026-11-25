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
