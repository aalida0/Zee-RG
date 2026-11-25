"""Production dipole kernel (zee/dipfull.py): textbook limits and Barr-Zee signs."""
import numpy as np
from zee import dipfull as D
from zee.observables import M_LEP

M = 500.0; mmu = M_LEP[1]
SC = dict(phi=0.3, m1=400.0, m2=900.0, sba=1.0, cba=0.0, tb=1.0, lam3=0.0, mu=0.0)


def one(p, L, f=None, Dd=None, sc=SC):
    Z = np.zeros((3, 3), complex)
    Lam = {"h": Z.copy(), "H": Z.copy(), "A": Z.copy()}; Lam[p] = L
    mass = {"h": M, "H": M, "A": M}
    AL, AR, P = D.amplitudes(Lam, mass, Z if f is None else f, Z if Dd is None else Dd, sc, {"h": (0, 0), "H": (0, 0)}, parts=True)
    return AL, AR, P


def test_scalar_light_fermion_amu():
    y = 0.3; L = np.zeros((3, 3), complex); L[1, 1] = y
    AL, AR, P = one("H", L)
    a = mmu**2 * np.real(P["n1"][0][1, 1] + P["n1"][1][1, 1])
    ref = mmu**2 * y**2 / (8 * np.pi**2 * M**2) * (np.log(M**2 / mmu**2) - 7 / 6)
    assert abs(a / ref - 1) < 1e-12


def test_pseudoscalar_light_fermion_amu():
    y = 0.3; L = np.zeros((3, 3), complex); L[1, 1] = 1j * y
    AL, AR, P = one("A", L)
    a = mmu**2 * np.real(P["n1"][0][1, 1] + P["n1"][1][1, 1])
    ref = -mmu**2 * y**2 / (8 * np.pi**2 * M**2) * (np.log(M**2 / mmu**2) - 11 / 6)
    assert abs(a / ref - 1) < 1e-12


def test_charged_scalar_amu():
    Dd = np.zeros((3, 3), complex); Dd[2, 1] = 0.4            # nu_tau mu_R coupling
    sc = dict(SC, phi=0.0)
    AL, AR, P = one("h", np.zeros((3, 3), complex), Dd=Dd, sc=sc)
    a = mmu**2 * np.real(P["ch"][0][1, 1] + P["ch"][1][1, 1])
    ref = -mmu**2 * 0.4**2 / (96 * np.pi**2 * sc["m1"] ** 2)
    assert abs(a / ref - 1) < 1e-12


def test_hermitian_diagonal():
    rng = np.random.default_rng(1)
    L = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    AL, AR, P = one("H", 0.01 * L)
    assert np.allclose(np.diag(AL), np.conj(np.diag(AR)), rtol=1e-12, atol=0)


def test_single_lfv_entry_chirality():
    """Only Lam[mu, tau] != 0 (ebar_L mu ... tau_R): tau -> mu gamma goes to A_R, not A_L, at leading order."""
    L = np.zeros((3, 3), complex); L[1, 2] = 0.01; L[2, 2] = 0.01
    AL, AR, P = one("H", L)
    assert abs(P["n1"][1][1, 2]) > 1e3 * abs(P["n1"][0][1, 2])


def test_bz_signs():
    """mu g-2 from Barr-Zee with SM-like couplings: CP-even top loop < 0, CP-odd > 0, W loop opposite to top."""
    sc = dict(SC, tb=1.0, sba=1.0, cba=0.0)
    L = np.zeros((3, 3), complex); L[1, 1] = mmu / D.V
    Z = np.zeros((3, 3), complex)
    Lam = {"h": L, "H": Z, "A": Z}; mass = {"h": 125.0, "H": M, "A": M}
    K = D.bz_kernels(Lam, mass, sc, {"h": (0, 0), "H": (0, 0)})
    assert K["h"]["f"].real < 0 and K["h"]["W"] > 0
    LA = np.zeros((3, 3), complex); LA[1, 1] = 1j * mmu / D.V
    Lam = {"h": Z, "H": Z, "A": LA}
    K = D.bz_kernels(Lam, mass, sc, {"h": (0, 0), "H": (0, 0)})
    assert np.real(LA[1, 1] * K["A"]["f"]) > 0


def test_charged_bz_heavy_limit():
    """Charged-scalar Barr-Zee in the heavy limit = effective phi-gamma-gamma vertex times the top-loop log
    (same relative sign as in h -> gamma gamma): A = -(alpha/8 pi^3) (lambda v/24 m^2) ln(m^2/m_phi^2) / v^0."""
    from zee import loops as LF
    w = 1e4
    num = LF.F2(w)
    approx = -(np.log(w) + 5 / 3) / (12 * w)   # leading-log coefficient -1/12 = top-loop/hgg ratio
    assert abs(num / approx - 1) < 1e-3
