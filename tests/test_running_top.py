"""Coarse-gate running-top Yukawa condition (report v11 4.2): rho_t = r y_t must stay perturbative."""
import numpy as np, sys, os
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _R); sys.path.insert(0, _R + '/gate1')
os.environ.setdefault('ZEE_INPUTS', 'modern')
from zee import rgfast2
import lmax2

C = np.array([.3, .3, .3, .3, .3, .0, .1, .0, .0, .0, .3])

def test_event_stops_large_rho():
    t0 = np.log(300.)
    s = rgfast2.run(C, t0, t1=np.log(1e16), cmax=1e6, r=3.5)      # quartic ceiling disabled: only the Yukawa event can stop it
    assert len(s.t_events[1]) == 1 and s.t[-1] < np.log(1e16)
    assert np.sqrt(4*np.pi) - abs(3.5*s.y[3, -1]) < 1e-3

def test_lam_valid_reports_yukawa():
    v, why = lmax2.lam_valid(C, np.log(300.), 3.0, status=True)
    assert why in ('yukawa', 'perturbativity', 'unitarity', 'BFB', 'unitarity@start', 'BFB@start')
    v2, why2 = lmax2.lam_valid(C*0, np.log(300.), 5.0, status=True)
    assert why2 == 'yukawa@start'
