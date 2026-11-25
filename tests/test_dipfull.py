"""Production dipole kernel (zee/dipfull.py): textbook limits and Barr-Zee signs."""
import numpy as np
from zee import dipfull as D
DF = D
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


# ---------------------------------------------------------------------------------------------------------------------
# H+- W-+ gamma Barr-Zee sets (Ilisie 1502.04199 Eqs. 22-24): independent quad implementation of the three formulas in the
# unmixed limit (phi = 0: h1+ = H+, h2+ = h+ decoupled from the lepton vertex) against zee.dipfull.bz_HW.
def _G_ref(wa, wb, x):
    X = wa*x + wb*(1 - x); y = x*(1 - x)
    return np.log(X/y)/(y - X)


def _quad(f):
    from scipy.integrate import quad
    return quad(f, 0, 1, points=[1e-5, 1e-4, 1e-3, 1e-2, 0.5, 0.99], limit=800, epsabs=0, epsrel=1e-10)[0]


def _hw_setup(mH=600.0, mHc=700.0, tb=0.8, sba=0.999, mA=650.0):
    cba = np.sqrt(1 - sba**2)
    sc = dict(phi=0.0, m1=mHc, m2=1500.0, tb=tb, sba=sba, cba=cba)
    tri = dict(raw_h=(0.31, 0.0, 0.0), raw_H=(0.47, 0.0, 0.0))
    return sc, dict(h=125.5, H=mH, A=mA), tri


def _ilisie_diag(sc, mass, tri, Dmm):
    """Delta a_mu^{(4)+(5)+(6)} of Ilisie for the muon with varsigma_l = v D/(sqrt2 m_mu), pure 2HDM (phi = 0)."""
    V, MW, SW2, al = DF.V, DF.MW, DF.SW2, DF.ALPHA
    m = DF.M[1]; zl = V*Dmm/(np.sqrt(2)*m); zu = -sc['tb']; MH = sc['m1']; dm = MH**2 - MW**2
    pre = al*m**2/(64*np.pi**3*SW2)
    a4 = al*m**2*3/(32*np.pi**3*SW2*V**2*dm)*_quad(lambda x: (DF.QT*x + DF.QB*(1 - x))*(zu*zl*DF.MB**2*x*(1 - x) + zu*zl*DF.MT**2*x*(1 + x))
         *(_G_ref(DF.MT**2/MH**2, DF.MB**2/MH**2, x) - _G_ref(DF.MT**2/MW**2, DF.MB**2/MW**2, x)))
    a5 = a6 = 0.0
    for ph, R1, R2 in (("h", sc['sba'], sc['cba']), ("H", sc['cba'], -sc['sba'])):
        mp2 = mass[ph]**2
        a5 += pre/(V**2*dm)*zl*R1*R2*_quad(lambda x: x**2*((MH**2 + MW**2 - mp2)*(1 - x) - 4*MW**2)
              *(_G_ref(MW**2/MH**2, mp2/MH**2, x) - _G_ref(1.0, mp2/MW**2, x)))
        lam = tri['raw_' + ph][0]
        a6 += pre/dm*zl*R2*lam*_quad(lambda x: x**2*(x - 1)*(_G_ref(1.0, mp2/MH**2, x) - _G_ref(MH**2/MW**2, mp2/MW**2, x)))
    return a4 + a5 + a6


def test_hw_matches_ilisie_diagonal():
    sc, mass, tri = _hw_setup()
    Dd = np.zeros((3, 3), complex); Dd[1, 1] = 0.013
    AL, AR = DF.bz_HW(Dd, mass, sc, tri)
    got = DF.M[1]**2*np.real(AL[1, 1] + AR[1, 1])
    ref = _ilisie_diag(sc, mass, tri, 0.013)
    assert abs(got/ref - 1) < 1e-6


def test_hw_flavour_structure():
    """A_L[J,I] is proportional to conj(Dd[I,J])/m_I and A_R[J,I] to Dd[J,I]/m_I with the same real coefficient."""
    sc, mass, tri = _hw_setup()
    rng = np.random.default_rng(2); Dd = rng.normal(size=(3, 3)) + 1j*rng.normal(size=(3, 3))
    AL, AR = DF.bz_HW(Dd, mass, sc, tri)
    k = AL[1, 2]*DF.M[2]/np.conj(Dd[2, 1])
    assert abs(k.imag) < 1e-12*abs(k)
    for J in range(3):
        for I in range(3):
            assert abs(AL[J, I] - k*np.conj(Dd[I, J])/DF.M[I]) < 1e-9*abs(AL[J, I])
            assert abs(AR[J, I] - k*Dd[J, I]/DF.M[I]) < 1e-9*abs(AR[J, I])


def test_hw_alignment_and_decoupling():
    sc, mass, tri = _hw_setup(sba=1.0)
    Dd = np.zeros((3, 3), complex); Dd[1, 2] = 0.02
    AL, _ = DF.bz_HW(Dd, mass, sc, dict(tri, raw_h=(0.0, 0.0, 0.0)))
    # exact alignment: h and the (5) sets vanish; only the top-bottom set and the H+H- contact set through H (R2 = -1) remain
    assert np.isfinite(AL).all()
    sc2, mass2, tri2 = _hw_setup(); big = dict(mass2, H=3*mass2['H'], A=3*mass2['A'])
    sc3 = dict(sc2, m1=3*sc2['m1'])
    a1 = abs(DF.bz_HW(Dd, mass2, sc2, tri2)[0][2, 1]); a3 = abs(DF.bz_HW(Dd, big, sc3, tri2)[0][2, 1])
    assert a3 < 0.5*a1          # heavier charged scalar: suppressed (non-trivially, with logs)


def test_G_finite_at_degeneracy():
    x = np.array([0.2, 0.5, 0.8]); X = x*(1 - x)
    v = DF._G(X/x*0 + 0.0, X/(1 - x), x)          # wa x + wb (1-x) = x(1-x) -> t = 1
    assert np.isfinite(v).all() and np.allclose(v, -1/(x*(1 - x)), rtol=1e-6)


def test_hw_mass_basis_independence_when_degenerate():
    """Summing the two charged mass eigenstates must reproduce the unmixed H+ result when they are degenerate, for any phi."""
    sc, mass, tri = _hw_setup(); Dd = np.zeros((3, 3), complex); Dd[1, 2] = 0.02; Dd[2, 1] = 0.01j
    tri = dict(raw_h=(0.31, 0.12, -0.2), raw_H=(0.47, -0.3, 0.25))
    ref = DF.bz_HW(Dd, mass, dict(sc, phi=0.0, m1=700., m2=700.), tri)
    for phi in (0.3, 1.1, 1.4685, 2.5):
        got = DF.bz_HW(Dd, mass, dict(sc, phi=phi, m1=700., m2=700.), tri)
        assert np.allclose(got[0], ref[0], rtol=1e-9, atol=0) and np.allclose(got[1], ref[1], rtol=1e-9, atol=0)


def test_hw_rephasing_covariance():
    sc, mass, tri = _hw_setup(); rng = np.random.default_rng(5)
    Dd = rng.normal(size=(3, 3)) + 1j*rng.normal(size=(3, 3))
    P = np.diag(np.exp(1j*rng.uniform(0, 6, 3))); Q = np.diag(np.exp(1j*rng.uniform(0, 6, 3)))
    AL, AR = DF.bz_HW(Dd, mass, sc, tri); AL2, AR2 = DF.bz_HW(P @ Dd @ Q, mass, sc, tri)
    assert np.allclose(AR2, P @ AR @ Q, rtol=1e-10) and np.allclose(AL2, Q.conj() @ AL @ P.conj(), rtol=1e-10)


def test_hw_scale_variants():
    """HW_SCALE = 0 removes, -1 flips the sign of exactly the bzHW component."""
    sc, mass, tri = _hw_setup(); Dd = np.zeros((3, 3), complex); Dd[1, 2] = 0.02
    Lam = {p: np.zeros((3, 3), complex) for p in ('h', 'H', 'A')}; f = np.zeros((3, 3), complex)
    tri = dict(tri, h=(0.1, 0.1), H=(0.1, 0.1)); out = {}
    for s in (1.0, 0.0, -1.0):
        DF.HW_SCALE = s
        try: out[s] = DF.amplitudes(Lam, mass, f, Dd, sc, tri, parts=True)
        finally: DF.HW_SCALE = 1.0
    P1, P0, Pm = out[1.0][2]['bzHW'], out[0.0][2]['bzHW'], out[-1.0][2]['bzHW']
    assert np.abs(P0[0]).max() == 0 and np.allclose(Pm[0], -P1[0]) and np.abs(P1[0]).max() > 0


def test_divided_difference_continuous_at_degeneracy():
    from zee import loops as LP
    for x in (0.02, 0.3, 4.0):
        for f in (LP.FHt, LP.FAt):
            a, b, c = f(x, x*(1 + 1e-3)), f(x, x), f(x, x*(1 - 1e-3))
            assert np.isfinite(b) and abs(b/a - 1) < 2e-3 and abs(b/c - 1) < 2e-3
