"""Input-sensitivity records (results/input_sensitivity.json): the default column must reproduce the production statistic and the RPP-2026 hook
(ZEE_STU=rpp2026) must give the oblique chi^2 of zee.stu.chi2_stu for the same S, T, U."""
import os, sys, json, subprocess
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
os.environ.setdefault('ZEE_INPUTS', 'modern')
from zee import hgmodel as H, stu as S
REC = json.load(open(os.path.join(ROOT, 'results', 'input_sensitivity.json')))
BM1 = json.load(open(os.path.join(ROOT, 'results', 'benchmarks_input.json')))['BM1']


def test_default_matches_record():
    e = H.evaluate(list(BM1['x']), 'NO', eta=BM1['eta'])
    assert abs(e['chi2'] - REC['default']['chi2']['BM1']) < 1e-6


def test_margins_are_limit_minus_chi2():
    for v in REC.values():
        for k, c in v['chi2'].items(): assert abs(v['margin'][k] - (v['limit'] - c)) < 1e-9


def test_rpp2026_hook():
    code = ("import sys, json; sys.path[:0] = [%r, %r, %r]\nfrom zee import hgmodel as H\n"
            "d = json.load(open(%r))['BM1']\ne = H.evaluate(list(d['x']), 'NO', eta=d['eta'])\nprint(json.dumps([e['chi_stu'], list(map(float, e['stu']))]))"
            % (ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2'), os.path.join(ROOT, 'results', 'benchmarks_input.json')))
    r = subprocess.run([sys.executable, '-W', 'ignore', '-c', code], capture_output=True, text=True, env=dict(os.environ, ZEE_STU='rpp2026'))
    assert r.returncode == 0, r.stderr[-300:]
    chi, stu = json.loads(r.stdout.strip().split('\n')[-1])
    assert abs(chi - S.chi2_stu(*stu)) < 1e-9 and abs(chi - REC['rpp2026']['stu']['BM1']) < 1e-9
