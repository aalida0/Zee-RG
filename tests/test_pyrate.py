import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'validation'))
import compare_pyrate as C


def test_all_beta_components_match_pyrate():
    w = C.compare(n=100)
    assert set(w) >= set(C.NAMES + ['yt', 'rho', 'g1', 'g2', 'g3'])
    assert all(v < C.RTOL for v in w.values()), w
