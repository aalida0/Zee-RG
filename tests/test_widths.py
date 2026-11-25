import os, sys, numpy as np, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zee import widths as W, dipfull as DF, V_EW as V

SC = dict(lam1=0.26, lam3=0.7, lam4=-0.38, lam5=0.55, lam6=-0.018, mA=735.3, phi=1.4685, m1=137., m2=758., mu=324., sba=1.0, cba=0.002)

def test_cp_even_masses():
    mh, mH, uh, uH = W.cp_even_basis(SC)
    assert abs(mh - 125.5) < 0.2 and abs(mH - 757.6) < 1.5      # reproduces the scalar sector (BM1 values)
    assert abs(uh @ uH) < 1e-12 and np.isclose(np.linalg.norm(uh), 1)

def test_Hhh_alignment_limit():
    sc = dict(SC, lam6=0.02)
    assert abs(W.g_Hhh(sc)) == pytest.approx(3*V*0.02, rel=0.1)       # |g| -> 3 v |Z6| when aligned

def test_Hhh_Z7_irrelevant_at_alignment():
    sc = dict(SC, lam6=0.001)
    assert abs(W.g_Hhh(sc, Z7=2.0) - W.g_Hhh(sc, Z7=0.0)) < 0.02*V*abs(sc['lam6']) + 0.5   # Z7 enters only through cba^3

def test_Hhh_width_closed_below_threshold():
    assert W.gamma_Hhh(SC, 200.) == 0.0 and W.gamma_Hhh(SC, 758.) > 0

def test_charged_offdiag_is_rotation_of_basis_matrix():
    raw = (0.7, 0.1, -0.3); c, s = np.cos(SC['phi']), np.sin(SC['phi'])
    G = V*np.array([[raw[0], raw[2]], [raw[2], raw[1]]]); R = np.array([[c, s], [-s, c]])
    Gp = R @ G @ R.T
    assert W.g_h_h1_h2(SC, raw) == pytest.approx(Gp[0, 1], rel=1e-10)
    assert Gp[0, 0] + Gp[1, 1] == pytest.approx(G[0, 0] + G[1, 1])

def test_quark_widths_scale():
    a = W.gamma_hp_quarks(300., 0.3); b = W.gamma_hp_quarks(300., 0.6)
    assert b['cs'] == pytest.approx(4*a['cs']) and a['cb'] < a['cs'] < 1e-3
    assert W.gamma_hp_quarks(1.0, 0.3)['cs'] == 0.0
