"""Corrected-statistic (V13) records: the four benchmarks pass the corrected gate, the reference minimum is stored, and the archive re-evaluation is consistent with it."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = lambda *p: json.load(open(os.path.join(ROOT, 'results', *p)))


def test_benchmarks_inside_corrected_limit():
    B = R('corrected', 'benchmarks_corrected.json'); c = R('refit_hw_cT_all.json')['best']['chi2']
    for k in ('BM1', 'BM2', 'BP1', 'BP3c'):
        assert B[k]['accepted'] and abs(B[k]['chi2_min'] - c) < 1e-6 and B[k]['chi2'] <= c + 3.84


def test_archive_reeval_consistent():
    r = R('prod1', 'reeval_corrected_all.json'); c = R('refit_hw_cT_all.json')['best']['chi2']
    assert abs(r['cmin'] - c) < 1e-6 and r['n'] == len(r['records']) == 559
    assert all(x['chi2_new'] <= c + 3.84 for x in r['records'] if x['status_new'] == 'PASS')
    assert sum(x['status_new'] == 'PASS' for x in r['records'] if x.get('also_finalist')) == 13


def test_input_sensitivity_primary_margins():
    I = R('input_sensitivity.json'); m = I['primary']['margin']
    assert all(m[p] > 0 for p in ('BM1', 'BM2', 'BP1', 'BP3c')) and all(I['oblique2016']['margin'][p] < 0 for p in ('BM1', 'BM2', 'BP1', 'BP3c'))
