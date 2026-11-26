"""One-loop dipoles of zee/dipfull.py against the independent Feynman-parameter / explicit-Dirac-matrix calculation (gate2/dipole_xcheck.py),
for random complex flavour-violating couplings.  dipfull expands in m_l^2/m_phi^2 (leading logs + constants), so the agreement is
limited by that expansion: < 1e-3 at m_phi = 125.5 GeV, improving as (m_tau/m_phi)^2; charged-scalar loops (massless nu) agree to 1e-5."""
import numpy as np
from gate2 import dipole_xcheck as X


def test_independent_dipoles_agree():
    r = X.compare(masses=(125.5, 1000.), ntrial=2)
    assert all(x["fit_resid"] < 1e-12 for x in r["neutral"] + r["charged"])
    for x in r["neutral"]:
        assert x["worst"] < (1.5e-3 if x["mphi"] < 200 else 2e-4), x
    for x in r["charged"]:
        assert x["worst"] < 1e-4, x


def test_independent_code_reproduces_hkz_internal_flip():
    """known limit: tau in the loop, A_R[mu,e] = L[mu,tau] L[tau,e] m_tau/(16 pi^2 m_e m_phi^2) (ln(m_phi^2/m_tau^2) - 3/2)."""
    from zee.observables import M_LEP
    M = np.asarray(M_LEP); L = np.zeros((3, 3), complex); L[1, 2] = 0.7; L[2, 0] = 0.4j
    AL, AR, _ = X.neutral_loop(L, 800., M)
    ref = L[1, 2] * L[2, 0] * M[2] / (16 * np.pi**2 * M[0] * 800.**2) * (np.log(800.**2 / M[2]**2) - 1.5)
    assert abs(AR[1, 0] / ref - 1) < 2e-3
