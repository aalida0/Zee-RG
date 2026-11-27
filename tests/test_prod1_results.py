import os, sys, json, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, 'results', 'prod1')


def test_prod1_table_and_validation_records_are_consistent():
    out = subprocess.run([sys.executable, os.path.join(ROOT, 'gate2', 'make_table_prod1.py')], capture_output=True, text=True, check=True).stdout
    assert out.startswith('% columns: BM1 BM2 BP1 BP3') and 'n/a' not in out and 'complete prescriptions' in out and 'including the stress variants' in out
    sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate2'), os.path.join(ROOT, 'gate1')]; os.environ.setdefault('ZEE_INPUTS', 'modern')
    import validate_finalist as VF
    ass = lambda p: VF.assess(json.load(open(p)))                                      # stored records re-assessed with the current exact-coverage logic
    V0 = os.path.join(ROOT, 'results', 'validation')
    a1, a2 = ass(os.path.join(V0, 'validate_BM1.json')), ass(os.path.join(V0, 'validate_BM2.json'))
    assert a1['all_prescriptions_complete'] and a1['classification'].startswith('robust') and abs(a1['robust_scale_matching_only'] - 11.3717) < 1e-3 and abs(a1['robust_scale'] - 11.3692) < 1e-3 and a1['all_combined_yukawa_complete']          # BM1 is a fully prescription-robust reference witness (report v12 P0)
    assert not a2['all_prescriptions_complete'] and a2['n_prescriptions_complete'] == 5 and a2['classification'].startswith('EP-clean')
    cls = {c: ass(os.path.join(P, 'validate_%s.json' % c))['classification'] for c in ('C42', 'C57', 'C35')}
    assert cls['C35'].startswith('robust') and cls['C42'].startswith('EP-clean') and cls['C57'].startswith('EP-clean')                               # BP3 robust; BP1 and BP2 EP-clean only
    B = json.load(open(os.path.join(P, 'benchmarks_prod1.json')))
    assert all(B[k]['accepted'] for k in ('BP1', 'BP2', 'BP3', 'BP4')) and all(B[k]['robustness']['vacuum']['global_ok'] for k in ('BP1', 'BP2', 'BP3'))
    S = json.load(open(os.path.join(P, 'campaign_summary.json')))
    assert S['n_files'] == 2089 and not S['missing_jobs'] and not S['duplicate_jobs'] and not S['unexpected_jobs'] and S['n_dirty'] == 0
    v = json.load(open(os.path.join(P, 'validate_C35.json')))                       # the one fully robust campaign benchmark (BP3)
    assert v['assessment']['all_prescriptions_complete'] and v['assessment']['robust_scale'] > 11.0 and v['assessment']['n_start_failures_total'] == 0
