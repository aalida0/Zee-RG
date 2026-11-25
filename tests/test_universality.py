"""HFLAV-Tau 2023 universality likelihood (pure leptonic ratios with correlations)."""
import numpy as np
from zee import hgmodel as H


def test_sm_point_chi2():
    v = np.array([1.0016, 1.0018, 1.0002]); s = np.array([.0014, .0014, .0011]); R = H.UNIV_CORR
    C = R*np.outer(s, s); r = 1 - v
    chi = float(r @ np.linalg.solve(C, r))
    got = float(np.sum(H.univ_resid(dict(tm=1.0, te=1.0, me=1.0))**2))
    assert abs(chi - got) < 1e-6 and abs(got - 1.829) < 2e-3


def test_central_values_and_consistency():
    assert H.UNIV["tm"][0] == 1.0016 and H.UNIV["te"][0] == 1.0018 and H.UNIV["me"][0] == 1.0002
    assert abs(1.0018/1.0016 - 1.0002) < 3e-5                      # g_mu/g_e = (g_tau/g_e)/(g_tau/g_mu)
    assert np.all(np.linalg.eigvalsh(H.UNIV_CORR) > 0)


def test_minimum_at_central_values():
    d = dict(tm=1.0016, te=1.0018, me=1.0002)
    assert np.sum(H.univ_resid(d)**2) < 1e-9
