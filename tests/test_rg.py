import numpy as np
from zee import rge, bfb, rgeq

Z3 = np.zeros((3, 3), complex)


def test_gauge_running_matches_one_loop_solution():
    g0 = np.array([0.36, 0.64, 1.05]); t1 = np.log(1e3 / 91.19)
    s = rge.run(Z3, Z3, Z3, 0.0, g0, 0.0, t1)
    g1 = s.y[-3:, -1]
    expected = 1 / g0**2 - 2 * rge.BG * t1 / (16 * np.pi**2)    # d(1/g^2)/dt = -2 b/(16 pi^2)
    assert np.allclose(1 / g1**2, expected, rtol=1e-6)


def test_yukawa_stays_small_for_small_inputs():
    Y2 = Z3.copy(); Y2[2, 1] = 1e-3
    s = rge.run(Z3, Y2, Z3, 0.9, [0.36, 0.64, 1.05], 0.0, np.log(1e4 / 91.19))
    assert s.status >= 0 and np.abs(s.y[:54, -1]).max() < 1.0


def test_bfb_agrees_with_tensor_minimiser_on_sign():
    good = np.array([0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0.5])
    bad = good.copy(); bad[2] = -5.0                                  # lambda3 very negative: unbounded
    assert bfb.min_v4(good, n=4000) >= -1e-6 and rgeq.min_quartic(rgeq.tensor_from(good)) >= -1e-6
    assert bfb.min_v4(bad, n=4000) < 0 and rgeq.min_quartic(rgeq.tensor_from(bad)) < 0


def test_barr_zee_functions_match_integrals():
    from scipy.integrate import quad
    from zee import bz
    for z in (0.3, 2.0, 25.0):
        f = z / 2 * quad(lambda x: (1 - 2 * x * (1 - x)) / (x * (1 - x) - z) * np.log(x * (1 - x) / z), 0, 1)[0]
        g = z / 2 * quad(lambda x: 1 / (x * (1 - x) - z) * np.log(x * (1 - x) / z), 0, 1)[0]
        assert np.isclose(bz.f_fn(z), f, rtol=1e-4) and np.isclose(bz.g_fn(z), g, rtol=1e-4)
