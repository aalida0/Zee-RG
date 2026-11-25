import numpy as np

from zee import observables as O

M_MU = O.M_LEP[1]


def f_with(**kw):
    idx = {"e": 0, "m": 1, "t": 2}
    f = np.zeros((3, 3), complex)
    for k, val in kw.items():
        a, b = idx[k[0]], idx[k[1]]
        f[a, b], f[b, a] = val, -val
    return f


def test_mu_egamma_matches_width_over_total():
    f = f_with(et=0.02, mt=0.03 * np.exp(0.7j))
    m1, m2, phi = 600.0, 2500.0, 0.4
    gamma_mu = O.GF**2 * M_MU**5 / (192.0 * np.pi**3)
    br_w = O.width_lgamma(1, 0, f, m1, m2, phi) / gamma_mu
    assert np.isclose(br_w, O.br_mu_egamma(f, m1, m2, phi), rtol=1e-12)


def test_only_f_et_f_mt_contribute_to_mu_e_gamma():
    base = O.br_mu_egamma(f_with(et=0.02, mt=0.03), 600.0, 2500.0, 0.4)
    assert base > 0
    # f_em alone cannot give mu->e gamma: sum_a conj(f_ae) f_am = 0 for a in {e,m,t}
    assert O.br_mu_egamma(f_with(em=0.1), 600.0, 2500.0, 0.4) == 0.0
    # explicit closed form S = 4 conj(f_te) f_tm
    S = 4 * np.conj(-0.02) * (-0.03)
    K = np.sin(0.4) ** 2 / 600.0**2 + np.cos(0.4) ** 2 / 2500.0**2
    ref = O.ALPHA / (768 * np.pi * O.GF**2) * abs(S * K) ** 2
    assert np.isclose(base, ref, rtol=1e-12)


def test_mixing_limits():
    f = f_with(et=0.02, mt=0.03)
    a0 = O.dipole_amplitude(0, 1, f, 600.0, 2500.0, 0.0)
    a1 = O.dipole_amplitude(0, 1, f, 600.0, 2500.0, np.pi / 2)
    assert np.isclose(a0 / a1, 600.0**2 / 2500.0**2)


def test_magnitude_sanity():
    f = f_with(et=1e-2, mt=1e-2)
    br = O.br_mu_egamma(f, 1000.0, 1000.0, 0.3)
    assert 1e-15 < br < 1e-14  # ~3.6e-15 by hand


def test_width_equals_g2_dipole_relation():
    # Gamma = alpha m |a|^2 / 16 with a = m^2 |S| K / (96 pi^2)
    f = f_with(et=0.02, mt=0.03)
    m1, m2, phi = 600.0, 2500.0, 0.4
    S = abs(O.dipole_amplitude(0, 1, f, m1, m2, phi))
    a = M_MU**2 * S / (96 * np.pi**2)
    assert np.isclose(O.width_lgamma(1, 0, f, m1, m2, phi), O.ALPHA * M_MU * a**2 / 16.0, rtol=1e-12)


def test_delta_a_mu_negative_and_scales():
    f = f_with(em=0.05, mt=0.05)
    a = O.delta_a(1, f, 600.0, 2500.0, 0.4)
    assert a < 0
    assert np.isclose(O.delta_a(1, 2 * f, 600.0, 2500.0, 0.4), 4 * a)


def test_higgs_lfv_width_and_alignment_scaling():
    D = np.zeros((3, 3), complex)
    D[1, 2] = 0.01
    c = 0.05
    g = O.higgs_lfv_width(1, 2, D, c)
    assert np.isclose(g, O.M_H / (8 * np.pi) * c**2 * 0.01**2 / 2, rtol=1e-14)
    assert np.isclose(O.higgs_lfv_width(1, 2, D, 2 * c), 4 * g)
    assert np.isclose(O.higgs_lfv_width(2, 1, D, c), g)  # sum over both charge states


def test_higgs_lfv_from_y2_only_offdiagonal():
    Y2 = np.zeros((3, 3), complex)
    Y2[1, 2] = 0.01
    tb = 5.0
    cb = 1 / np.sqrt(1 + tb**2)
    br = O.higgs_lfv_br(1, 2, Y2, tb, 0.1)
    g = O.M_H / (8 * np.pi) * (0.1 * 0.01 / cb) ** 2 / 2
    assert np.isclose(br, g / (O.GAMMA_H_SM + g), rtol=1e-12)


def test_dipole_conversion_ratio_from_overlap_integrals():
    """CR(mu->e, N) / BR(mu->e gamma) = pi D^2 Gamma_mu / Gamma_capt for dipole dominance (Kitano-Koike-Okada, hep-ph/0203110 normalisation
    BR = 384 pi^2 (|A_L|^2 + |A_R|^2), Gamma_conv = 2 G_F^2 |A D|^2): D(Al) = 0.0362, D(Au) = 0.189 (units m_mu^{5/2}),
    Gamma_capt(Al) = 0.7054e6/s, Gamma_capt(Au) = 13.07e6/s; the model uses 2.67e-3 and 3.9e-3."""
    import numpy as np
    gmu = 1/2.1969811e-6
    assert abs(np.pi*0.0362**2*gmu/0.7054e6/2.67e-3 - 1) < 0.02
    assert abs(np.pi*0.189**2*gmu/13.07e6/3.9e-3 - 1) < 0.02
