"""Input-sensitivity of the benchmark chi^2 under the corrected oblique-T statistic (V13 referee).  One subprocess per variation (the limits are fixed at import time of zee.hgmodel).
Primary = ZEE_INPUTS=modern2026 (RPP 2026 free-U oblique fit, exact z_0.975 = 1.95996, DESI-DR2 sum m_nu < 0.0642 eV; zee/hgmodel.py).  Variations override one ingredient at a time:
  oblique2016 (ZEE_STU=fit2016, the frozen 2016 fit values), z95_1.949 (the frozen half-normal factor), desi2024 (sum m_nu < 0.072 eV), nocosmo (cosmological term removed), frozen (all three frozen inputs = ZEE_INPUTS=modern).
Margins are chi^2_min + 3.84 - chi^2 with the refit minimum of the SAME variation (results/refit_hw_cT_<tag>.json from gate2/refit_hw.py with the same environment).  Ambient ZEE_* variables are
unset before each configuration and the environment actually used is stored.  The earlier legacy-T table is kept as results/input_sensitivity_legacyT.json (historical, printed-HG T factor).
Usage: python3 -W ignore gate2/input_sensitivity.py"""
import os, sys, json, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AMBIENT = ('ZEE_STU', 'ZEE_Z95', 'ZEE_SUMM_LIMIT', 'ZEE_STU_LEGACY_HG', 'ZEE_ING', 'ZEE_INPUTS')
VARS = {'primary': {}, 'oblique2016': {'ZEE_STU': 'fit2016'}, 'z95_1.949': {'ZEE_Z95': '1.949'}, 'desi2024': {'ZEE_SUMM_LIMIT': '0.072'}, 'nocosmo': {'ZEE_SUMM_LIMIT': '1000'},
        'frozen': {'ZEE_INPUTS': 'modern'}}
REF = {'primary': 'refit_hw_cT_all.json', 'oblique2016': 'refit_hw_cT_oblique2016.json', 'z95_1.949': 'refit_hw_cT_z95_1.949.json', 'desi2024': 'refit_hw_cT_desi2024.json',
       'nocosmo': 'refit_hw_cT_nocosmo2026.json', 'frozen': 'refit_hw_cT_default.json'}
CODE = r'''
import sys, json, os
sys.path[:0] = [%r, os.path.join(%r, 'gate1'), os.path.join(%r, 'gate2')]
from zee import hgmodel as H
out = {}
for f in ('results/benchmarks_input.json', 'results/prod1/benchmarks_input_prod1.json', 'results/corrected/BP3c.json'):
    p = os.path.join(%r, f)
    if not os.path.exists(p): continue
    d = json.load(open(p)); d = d if 'x' not in d else {'BP3c': d}
    for k in ('BM1', 'BM2', 'BP1', 'BP3', 'BP3c'):
        if k in d:
            e = H.evaluate(list(d[k]['x']), 'NO', eta=d[k]['eta']); out[k] = dict(chi2=e['chi2'], stu=e['chi_stu'], stu_values=list(map(float, e['stu'])), univ=e['chi_univ'], osc=e['chi_osc'], bounds_sum_m=e['chi_bounds'].get('sum_m'))
out['_effective'] = dict(sum_m_limit=H.BOUNDS['sum_m'][0], z_factor=H.BOUNDS['sum_m'][1], stu_center=list(map(float, H.STU_C)), stu_sigma=list(map(float, H.STU_S)), stu_cov=[list(map(float, r)) for r in H.STU_COV], T_charged_pair_legacy=bool(H.STU_HG_LEGACY))
print(json.dumps(out))
''' % (ROOT, ROOT, ROOT, ROOT)
if __name__ == '__main__':
    res = {}; envs = {}; eff = {}
    for tag, env in VARS.items():
        e = {k: v for k, v in os.environ.items() if k not in AMBIENT}; e['ZEE_INPUTS'] = 'modern2026'; e.update(env); envs[tag] = {k: v for k, v in e.items() if k.startswith('ZEE_')}
        r = subprocess.run([sys.executable, '-W', 'ignore', '-c', CODE], capture_output=True, text=True, env=e, cwd=ROOT)
        if r.returncode: print(tag, 'FAILED', r.stderr[-400:]); continue
        res[tag] = json.loads(r.stdout.strip().split('\n')[-1]); eff[tag] = res[tag].pop('_effective')
    out = {}
    for tag, d in res.items():
        fn = os.path.join(ROOT, 'results', REF[tag]); m = json.load(open(fn))['best']['chi2'] if os.path.exists(fn) else None
        out[tag] = dict(chi2_min=m, limit=None if m is None else m + 3.84, chi2={k: v['chi2'] for k, v in d.items()}, margin={k: (None if m is None else m + 3.84 - v['chi2']) for k, v in d.items()},
                        stu={k: v['stu'] for k, v in d.items()}, stu_values={k: v['stu_values'] for k, v in d.items()}, T_factor='general Zee (-2 w1 w2 F)', env=envs[tag], effective_inputs=eff[tag], reference_file=REF[tag])
        print('%-12s min %s  ' % (tag, 'n/a' if m is None else '%.3f' % m) + '  '.join('%s %.3f%s' % (k, v['chi2'], '' if m is None else ' (%+.3f)' % (m + 3.84 - v['chi2'])) for k, v in d.items()))
    json.dump(out, open(os.path.join(ROOT, 'results', 'input_sensitivity.json'), 'w'), indent=1)
