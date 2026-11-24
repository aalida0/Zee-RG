import numpy as np

from zee import dipoles as P
from zee import observables as O


def f_with(**kw):
    idx = {"e": 0, "m": 1, "t": 2}
    f = np.zeros((3, 3), complex)
    for k, v in kw.items():
        a, b = idx[k[0]], idx[k[1]]
        f[a, b], f[b, a] = v, -v
    return f


def test_f_only_matches_observables_layer():
    f = f_with(et=0.02, mt=0.03 * np.exp(0.5j))
    Y2 = np.zeros((3, 3), complex)
    m1, m2, phi = 600.0, 2500.0, 0.4
    # remove H+ term: with Y2=0, D = -tb sqrt2 m/v is diagonal and gives (DD^dag)_{ij}=0 for i!=j
    b = P.br(1, 0, f, Y2, 5.0, 0.0, 900.0, 900.0, m1, m2, phi)
    assert np.isclose(b, O.br_mu_egamma(f, m1, m2, phi), rtol=1e-10)
    b2 = P.br(2, 1, f, Y2, 5.0, 0.0, 900.0, 900.0, m1, m2, phi)
    assert np.isclose(b2, O.br_tau_lgamma(1, f, m1, m2, phi), rtol=1e-10)


def test_alignment_couplings():
    g1, g2 = P.neutral_couplings(7.0, 0.0)
    v = 246.21965
    assert np.isclose(g1[0], 1.0)           # h couples as m/v in the alignment limit
    assert abs(g2[0]) < 1e-15               # no off-diagonal h coupling when aligned
    cb = 1 / np.sqrt(1 + 49)
    assert np.isclose(abs(g2[1]), 1 / (np.sqrt(2) * cb))
    assert np.isclose(g1[2], -7j)


def test_neutral_mu_e_scaling_and_signs():
    Y2 = np.zeros((3, 3), complex)
    Y2[2, 1], Y2[0, 2] = 1e-2, 1e-2
    a = P.mu_to_e(Y2, 3.0, 0.0, (125.25, 800.0, 800.0))
    # H and A (equal mass) have (g2)^2 of opposite sign at c_ba=0 -> exact cancellation
    # of the log-enhanced term: (1/(2cb^2))(-s^2)... + (i)^2/(2cb^2)... -> -1 + -1? check
    g1, g2 = P.neutral_couplings(3.0, 0.0)
    assert np.isclose(g2[1] ** 2 + g2[2] ** 2, 0.0, atol=1e-15)
    assert abs(a[0]) < 1e-3 * abs(P.mu_to_e(Y2, 3.0, 0.0, (125.25, 800.0, 3000.0))[0])


def test_charged_H_term_structure_tau_mu():
    # tau -> mu: (D D^dag)_{mu tau} includes Y2^{mu tau} * conj(D_tautau)
    tb = 4.0
    Y2 = np.zeros((3, 3), complex)
    Y2[1, 2] = 0.02
    Y2[2, 2] = 0.01
    D = O.d_matrix(O.M_LEP, Y2, tb)
    ref = sum(D[1, a] * np.conj(D[2, a]) for a in range(3))
    cl, cr = P.charged(2, 1, np.zeros((3, 3)), Y2, tb, 500.0, 2000.0, 0.3)
    K_H = np.cos(0.3) ** 2 / 500.0**2 + np.sin(0.3) ** 2 / 2000.0**2
    assert np.isclose(cl, -K_H * ref / (192 * np.pi**2), rtol=1e-12) and cr == 0


def test_cr_cl_conjugation_rule_diag_only_vanishes():
    Y2 = np.diag([0, 0.01, 0.02]).astype(complex)
    cl, cr = P.tau_to_l(1, Y2, 3.0, 0.05, (125.25, 800.0, 800.0))
    assert cl == 0 and cr == 0  # needs Y2^{mu tau} or Y2^{tau mu}
