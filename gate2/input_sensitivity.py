"""Input-sensitivity of the benchmark chi^2 (referee plan): one subprocess per variation, because the limits are fixed at import time of zee.hgmodel.
Variations (environment): default; z95 (exact half-normal factor 1.95996, ZEE_Z95); dr2 (sum m_nu < 0.0642 eV, ZEE_SUMM_LIMIT); rpp2026 (ZEE_STU=rpp2026); nocosmo (sum m_nu term removed, ZEE_SUMM_LIMIT=1e3); all (z95 + dr2 + rpp2026 together).
Prints chi^2 of the points BM1 BM2 (results/benchmarks_input.json) and BP1 BP3 (results/prod1/benchmarks_input_prod1.json) and writes results/input_sensitivity_chi2.json.
The reference minima chi^2_min of each variation come from gate2/refit_hw.py run with the same environment (TAG), not from this script.
Usage: ZEE_INPUTS=modern python3 -W ignore gate2/input_sensitivity.py"""
import os, sys, json, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VARS = {'default': {}, 'z95': {'ZEE_Z95': '1.95996'}, 'dr2': {'ZEE_SUMM_LIMIT': '0.0642'}, 'rpp2026': {'ZEE_STU': 'rpp2026'}, 'nocosmo': {'ZEE_SUMM_LIMIT': '1000'}, 'all': {'ZEE_Z95': '1.95996', 'ZEE_SUMM_LIMIT': '0.0642', 'ZEE_STU': 'rpp2026'}}
CODE = r'''
import sys, json, os
sys.path[:0] = [%r, os.path.join(%r, 'gate1'), os.path.join(%r, 'gate2')]
from zee import hgmodel as H
out = {}
for f in ('results/benchmarks_input.json', 'results/prod1/benchmarks_input_prod1.json'):
    d = json.load(open(os.path.join(%r, f)))
    for k in ('BM1', 'BM2', 'BP1', 'BP3'):
        if k in d:
            e = H.evaluate(list(d[k]['x']), 'NO', eta=d[k]['eta']); out[k] = dict(chi2=e['chi2'], stu=e['chi_stu'], univ=e['chi_univ'], bounds_sum_m=e['chi_bounds'].get('sum_m'))
print(json.dumps(out))
''' % (ROOT, ROOT, ROOT, ROOT)
if __name__ == '__main__':
    res = {}
    for tag, env in VARS.items():
        e = dict(os.environ, ZEE_INPUTS='modern', **env)
        r = subprocess.run([sys.executable, '-W', 'ignore', '-c', CODE], capture_output=True, text=True, env=e, cwd=ROOT)
        if r.returncode: print(tag, 'FAILED', r.stderr[-400:]); continue
        res[tag] = json.loads(r.stdout.strip().split('\n')[-1])
    # margins chi2_min + 3.84 - chi2, with the refit minimum of the same variation (gate2/refit_hw.py, TAG as below)
    REF = {'default': 'refit_hw.json', 'z95': 'refit_hw_z95.json', 'dr2': 'refit_hw_desi_dr2.json', 'rpp2026': 'refit_hw_rpp2026.json', 'nocosmo': 'refit_hw_nocosmo.json', 'all': 'refit_hw_allnew.json'}
    out = {}
    for tag, d in res.items():
        m = json.load(open(os.path.join(ROOT, 'results', REF[tag])))['best']['chi2']
        out[tag] = dict(chi2_min=m, limit=m + 3.84, chi2={k: v['chi2'] for k, v in d.items()}, margin={k: m + 3.84 - v['chi2'] for k, v in d.items()}, stu={k: v['stu'] for k, v in d.items()})
        print('%-8s min %.3f  ' % (tag, m) + '  '.join('%s %.3f (margin %+.3f)' % (k, v['chi2'], m + 3.84 - v['chi2']) for k, v in d.items()))
    json.dump(out, open(os.path.join(ROOT, 'results', 'input_sensitivity.json'), 'w'), indent=1)
