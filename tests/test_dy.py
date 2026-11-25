import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zee import dy
import pytest

def test_endpoints_match_table():
    assert dy.sigma_dy(100., 0.) == pytest.approx(96.51); assert dy.sigma_dy(150., 1.) == pytest.approx(63.34)

def test_monotone_in_mass_and_doublet():
    ms = [100, 120, 175, 260, 333, 480]
    for a, b in zip(ms, ms[1:]): assert dy.sigma_dy(a, .3) > dy.sigma_dy(b, .3)
    assert dy.sigma_dy(200., 1.) > dy.sigma_dy(200., 0.)

def test_heavy_state_irrelevant():
    assert dy.sigma_dy(1000., 1.) == 0.0

def test_reference_exclusion():
    # right-handed e+mu at 100% BR: excluded below ~420 GeV, allowed above
    assert not dy.dy_allowed([300.], [0.], [1.], [0.])[0]      # single flavour at 300 GeV: 1.68 fb > 0.82
    assert dy.dy_allowed([420.], [0.], [.5], [.5])[0]          # 2 x 0.5^2 x sigma_R(420) < limit
    assert dy.dy_allowed([150.], [0.], [0.], [0.])[0]          # no e/mu: tau nu only evades
    assert dy.dy_signal([150.], [0.], [.1], [.05]) == pytest.approx(23.32*(0.01+0.0025))
    assert not dy.dy_allowed([150.], [0.], [.5], [.5])[0]
