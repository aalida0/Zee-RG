"""All-Yukawa RGEs (zee/rgfull.py): reduction to the production top-only RGEs and the analytic top Yukawa beta function."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('ZEE_INPUTS', 'modern')
from zee import rgfull as F, rgfast2 as R


def _top_only(c, r=-0.43, yt=0.88):
    g = np.array([.36, .64, 1.05]); Z = np.zeros((3, 3), complex)
    return g, F.pack(g, yt, r*yt, 0, 0, Z, Z, Z, c)


def test_top_beta_function_matches_analytic_formula():
    g, z = _top_only(np.zeros(11)); d = F.rhs(0, z); k = 1/(16*np.pi**2); r, yt = -0.43, 0.88
    assert abs(d[3] - k*yt*(4.5*yt**2*(1 + r**2) - 8*g[2]**2 - 2.25*g[1]**2 - 17/12*g[0]**2)) < 1e-12
    assert abs(d[4]/d[3] - r) < 1e-12                      # r = rho_t/y_t is invariant for a single top


def test_quartic_beta_reduces_to_production_rgeqs():
    c = np.random.default_rng(3).normal(size=11)*0.4; g, z = _top_only(c)
    d = F.rhs(0, z); d2 = R.rhs_r(-0.43)(0, np.concatenate([g, [0.88], c]))
    assert np.abs(d[61:] - d2[4:]).max() < 1e-12
    assert F.rhs(0, z, return_resid=True)[0] < 1e-12           # the quartic beta function stays inside the 11-parameter family


def test_tau_yukawa_beta_function_matches_the_analytic_leading_terms():
    g, z = _top_only(np.zeros(11)); r, yt, ytau = -0.43, 0.88, 0.0102
    Y1 = np.diag([2.9e-6, 6e-4, ytau]).astype(complex)
    z = F.pack(g, yt, r*yt, 0.0155, r*0.0155, Y1, r*Y1, np.zeros((3, 3), complex), np.zeros(11)); dY1 = F.unpack(F.rhs(0, z))[5]; k = 1/(16*np.pi**2)
    lead = k*ytau*(3*yt**2*(1 + r**2) - 2.25*g[1]**2 - 3.75*g[0]**2)           # two doublets: H1 and H2 (coupling r y_t) both enter the anomalous dimension of the Higgs fields
    assert abs(dY1[2, 2].real - lead) < 0.02*abs(lead)                           # remainder: O(y_tau^3, y_b^2) terms
    assert abs(dY1[2, 2].imag) < 1e-15 and np.abs(dY1 - np.diag(np.diag(dY1))).max() < 1e-15      # stays real and diagonal


def test_singlet_yukawa_gauge_only_beta_and_roundtrip():
    """report v11: the tensor entry of nu_i e_j is f_ij - f_ji = 2 f_ij, so beta_f is half of the extracted entry; gauge-only analytic beta_f = -3 (C_nu + C_e) f / 16 pi^2 in both RG modules,
    and the build -> extract roundtrip returns an antisymmetric matrix"""
    rng = np.random.default_rng(4); A = rng.normal(size=(3, 3)) + 1j*rng.normal(size=(3, 3)); f = (A - A.T)*1e-8
    g = np.array([0.36, 0.64, 1.05]); Z3 = np.zeros((3, 3), complex); C = F.casimirs(g); ana = -3*(C[F.NU] + C[F.E])*f/(16*np.pi**2)
    d = F.unpack(F.rhs(0, F.pack(g, 0, 0, 0, 0, Z3, Z3, f, np.zeros(11))))[7]
    m = np.abs(f) > 0
    assert np.abs(d + d.T).max() < 1e-20 and np.abs(d[m]/ana[m] - 1).max() < 1e-6
    from zee import rge
    d2 = rge.unpack(rge.rhs(0, rge.pack(Z3, Z3, f, 0.0, g)))[2]
    assert np.abs(d2[m]/ana[m] - 1).max() < 1e-6
