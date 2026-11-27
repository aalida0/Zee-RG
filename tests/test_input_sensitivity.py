"""Input-sensitivity records (results/input_sensitivity.json, corrected oblique-T statistic, primary = ZEE_INPUTS=modern2026): the primary column must reproduce the production
statistic evaluated in a clean environment, and the margins must be limit minus chi^2.  The pytest process itself runs in the historical legacy-T mode (conftest.py), so every evaluation here is a subprocess."""
import os, sys, json, subprocess
import numpy as np
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
from zee import stu as S
REC = json.load(open(os.path.join(ROOT, 'results', 'input_sensitivity.json')))


def _bm1(env):
    code = ("import sys, json; sys.path[:0] = [%r, %r, %r]\nfrom zee import hgmodel as H\nd = json.load(open(%r))['BM1']\n"
            "e = H.evaluate(list(d['x']), 'NO', eta=d['eta'])\nprint(json.dumps([e['chi2'], e['chi_stu'], list(map(float, e['stu']))]))"
            % (ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2'), os.path.join(ROOT, 'results', 'benchmarks_input.json')))
    e = {k: v for k, v in os.environ.items() if not k.startswith(('ZEE_STU', 'ZEE_Z95', 'ZEE_SUMM', 'ZEE_INPUTS'))}; e.update(env)
    r = subprocess.run([sys.executable, '-W', 'ignore', '-c', code], capture_output=True, text=True, env=e)
    assert r.returncode == 0, r.stderr[-300:]
    return json.loads(r.stdout.strip().split('\n')[-1])


def test_primary_matches_record_in_clean_environment():
    chi2, chi_stu, stu = _bm1({'ZEE_INPUTS': 'modern2026'})
    assert chi2 == pytest.approx(REC['primary']['chi2']['BM1'], abs=1e-9) and chi_stu == pytest.approx(S.chi2_stu(*stu), abs=1e-9)
    eff = REC['primary']['effective_inputs']
    assert eff['sum_m_limit'] == 0.0642 and eff['z_factor'] == 1.95996 and eff['stu_center'] == [0.018, 0.04, -0.013] and not eff['T_charged_pair_legacy']


def test_frozen_inputs_column_matches_record():
    chi2, _, _ = _bm1({'ZEE_INPUTS': 'modern'})
    assert chi2 == pytest.approx(REC['frozen']['chi2']['BM1'], abs=1e-9)


def test_margins_are_limit_minus_chi2():
    for v in REC.values():
        for k, c in v['chi2'].items(): assert abs(v['margin'][k] - (v['limit'] - c)) < 1e-9
