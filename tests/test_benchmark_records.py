import os, sys, json, numpy as np, pytest
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [R, R + '/gate1', R + '/gate2']; os.environ.setdefault('ZEE_INPUTS', 'modern')
B = json.load(open(R + '/results/benchmarks.json')); INP = json.load(open(R + '/results/benchmarks_input.json'))
CMIN = json.load(open(R + '/results/refit_hw.json'))['best']['chi2']

def test_records_are_current_and_complete():
    assert set(B) == set(INP)
    for n, r in B.items():
        assert r['status'] == 'PASS' and r['accepted'] is True and not r['a_unresolved'] and r['a_failed_cuts'] == []
        for k in ('domain', 'chi2', 'penalty', 'DY', 'bsgamma', 'hw_variants', 'HB', 'HS', 'h_tau_mu', 'h_tau_e', 'rg_start', 'vacuum'):
            assert r['cuts'].get(k) is True, (n, k)
        assert r['dy']['status'] == 'PASS' and r['chi2'] <= CMIN + 3.84 and r['chi2_min'] == pytest.approx(CMIN)
        assert r['certified_log10_scale'] >= 11.0 and r['robustness']['vacuum']['global_ok']

def test_records_reproduce_with_current_code():
    pytest.importorskip('Higgs')
    import accept as AC
    for n in B:
        a = AC.accept(np.array(INP[n]['x']), np.array(INP[n]['eta']), CMIN, n_vac=40)
        assert a['status'] == 'PASS' and a['chi2'] == pytest.approx(B[n]['chi2'], rel=1e-9)

def test_old_benchmarks_stay_rejected():
    pytest.importorskip('Higgs')
    import accept as AC
    old = json.load(open(R + '/results/benchmarks_input_superseded.json'))
    for n, v in old.items():
        a = AC.accept(np.array(v['x']), np.array(v['eta']), CMIN, n_vac=20)
        assert a['status'] == 'FAIL' and a['cuts']['DY'] is False
