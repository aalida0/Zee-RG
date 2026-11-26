"""Z2-basis inversion used for the GM2Calc comparison away from alignment reproduces GM2Calc's own lambda_1..5 (printed by the library for the same mass-basis input)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'gate2'))
import thdm_basis as T

V_GM2 = 242.1839719330102
GM2_LAMBDA = [8.91180661, 1.458494781, 1.936695144, -1.592264146, -3.283565034]       # GM2Calc 2.3.1, (mHp,mH,mA,sba,l6,l7,tb,m122) = (600,650,640,0.999,0.05,-0.03,1.5,1e5)


def test_inversion_reproduces_gm2calc_quartics():
    x = T.invert(600, 650, 640, 0.999, 0.05, -0.03, 1.5, 1e5, v=V_GM2)
    assert np.allclose(x, GM2_LAMBDA, rtol=1e-7)
    p = T.physical(list(x) + [0.05, -0.03], 1.5, 1e5, V_GM2)
    assert abs(p['mh'] - 125.09) < 1e-6 and abs(p['sba'] - 0.999) < 1e-9 and p['cba'] > 0


def test_higgs_basis_rotation_preserves_the_quartic_form():
    lam = list(GM2_LAMBDA) + [0.05, -0.03]; Z = T.higgs_basis(lam, 1.5)
    assert np.all(np.isfinite(Z))
    # Z2 = lambda_2-like combination must reproduce the SM-like quartic: Z1 v^2 equals the (h1,h1) CP-even entry
    hh, Ah, Hm = T.mass_matrices(lam, 1.5, 1e5, V_GM2); b = np.arctan(1.5); R = np.array([[np.cos(b), np.sin(b)], [-np.sin(b), np.cos(b)]])
    M_hb = R @ hh @ R.T
    assert abs(M_hb[0, 0] - Z[0]*V_GM2**2) < 1e-6*abs(M_hb[0, 0])
