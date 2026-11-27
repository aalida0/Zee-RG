import os, sys, json, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, 'results', 'prod1')


def test_prod1_table_and_validation_records_are_consistent():
    out = subprocess.run([sys.executable, os.path.join(ROOT, 'gate2', 'make_table_prod1.py')], capture_output=True, text=True, check=True).stdout
    assert out.startswith('% columns: BM1 BM2 BP1 BP2 BP3') and 'n/a' not in out
    B = json.load(open(os.path.join(P, 'benchmarks_prod1.json')))
    assert all(B[k]['accepted'] for k in ('BP1', 'BP2', 'BP3', 'BP4')) and all(B[k]['robustness']['vacuum']['global_ok'] for k in ('BP1', 'BP2', 'BP3'))
    S = json.load(open(os.path.join(P, 'campaign_summary.json')))
    assert S['n_files'] == 2089 and not S['missing_jobs'] and not S['duplicate_jobs'] and not S['unexpected_jobs'] and S['n_dirty'] == 0
    v = json.load(open(os.path.join(P, 'validate_C35.json')))                       # the one fully robust campaign benchmark (BP3)
    assert v['assessment']['all_prescriptions_complete'] and v['assessment']['robust_scale'] > 11.0 and v['assessment']['n_start_failures_total'] == 0
