import os, sys, numpy as np, json, pytest
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R); sys.path.insert(0, R + '/gate2'); sys.path.insert(0, R + '/gate1')
os.environ.setdefault('ZEE_INPUTS', 'modern')
import accept as AC
from zee import flavour as FL

B = json.load(open(R + '/results/benchmarks_input.json'))

def test_domain_rejects_large_yukawa():
    x = np.array(B['BM1']['x']); x[3] = 0.5
    assert AC.in_domain(x, B['BM1']['eta']) == ['yukawa>0.1']
    r = AC.accept(x, B['BM1']['eta'], 14.0, collider=False)
    assert r['status'] == 'FAIL' and not r['accepted']

def test_no_collider_is_unresolved_not_accepted():
    r = AC.accept(np.array(B['BM1']['x']), B['BM1']['eta'], 14.0, collider=False, n_vac=10)
    assert r['status'] in ('UNRESOLVED', 'FAIL') and r['accepted'] is False
    assert any('collider' in u for u in r['unresolved'])

def test_bsgamma_decoupling_and_sign():
    sc = dict(tb=0.3, phi=0.3, m1=2000., m2=3000.)
    assert abs(FL.bsg_e4(sc) - 3.36) < 0.05                       # heavy scalars: only the SM remains
    assert FL.bsg_e4(dict(tb=3., phi=0.0, m1=150., m2=900.)) < 3.36 - 1   # light type-I charged Higgs with large tan(beta) lowers the rate

def test_old_benchmarks_not_accepted():
    pytest.importorskip('Higgs')
    for k in ('BM1', 'BM2'):
        r = AC.accept(np.array(B[k]['x']), B[k]['eta'], 13.85, n_vac=10)
        assert r['accepted'] is False and not r['cuts'].get('DY', True)
