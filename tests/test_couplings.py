"""Referee v5 C01, C04, C07 regression tests on the production helpers."""
import numpy as np
from zee import hgmodel as H

SC = dict(tb=2.0, sba=0.999, cba=-0.0447, phi=0.4, m1=400.0, m2=900.0)


def test_universality_decoupling_and_hg_formula():
    f = np.zeros((3, 3)); f[1, 2] = 0.1; f -= f.T
    u1, _ = H.universality(f, SC)
    sc2 = dict(SC, m1=SC['m1']*10, m2=SC['m2']*10)
    u2, _ = H.universality(f, sc2)
    d1, d2 = u1['te']**2 - 1, u2['te']**2 - 1
    assert abs(d1/d2 - 100) < 1e-6                      # inverse-square decoupling
    Mt2 = 1/(np.sin(SC['phi'])**2/SC['m1']**2 + np.cos(SC['phi'])**2/SC['m2']**2)
    assert abs(d1 - np.sqrt(2)*0.01/(H.GF*Mt2)) < 1e-12  # HG eq. (3.32)


def test_charged_width_single_yukawa():
    """Only Y2^{mu tau} != 0: Dd = Y2^dag/cb has its entry at [tau, mu], i.e. the charged scalar decays to mu nu."""
    Y = np.zeros((3, 3), complex); Y[1, 2] = 0.03
    sc = dict(SC, tb=1e-9)    # switch off the mass term
    Lam, Dd = H.lepton_couplings(Y, sc)
    G = H.charged_lepton_widths(np.zeros((3, 3)), Dd, sc)
    assert G[0, 1] > 0 and G[0, 2] < 1e-12*G[0, 1] and G[0, 0] < 1e-12*G[0, 1]


def test_pseudoscalar_real_yukawa():
    """real Dd gives a purely CP-odd (imaginary) A coupling, nonzero"""
    Y = np.diag([0.0, 0.0, 0.02]).astype(complex)
    Lam, Dd = H.lepton_couplings(Y, SC)
    a = Lam['A'][2, 2]
    assert abs(a.real) < 1e-15 and abs(a.imag) > 1e-3
