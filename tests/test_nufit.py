import os
import numpy as np
import pytest
from zee.nufit import NuFit, _CACHE

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(_CACHE, "TByes-NO.npz")), reason="no NuFIT cache")

def best(o):
    z = np.load(os.path.join(_CACHE, f"TByes-{o}.npz"))
    g = z["grid3d"]; i = np.unravel_index(g.argmin(), g.shape)
    b = lambda k: z[k][z[k][:, 1].argmin(), 0]
    return NuFit(o), (b("T12"), b("T13"), z["ax0"][i[0]], 10 ** b("DMS"), z["ax1"][i[1]] * 1e-3, z["ax2"][i[2]])

def test_no_minimum_zero():
    n, p = best("NO"); assert abs(n.chi2(*p)) < 1e-6

def test_io_offset_counted_once():
    n, p = best("IO"); assert abs(n.chi2(*p) - n.dmin) < 1e-6 and 5 < n.dmin < 7

def test_delta_wrap_and_bounds():
    n, p = best("NO")
    assert np.isclose(n.chi2(*p[:5], p[5] + 360.0), n.chi2(*p))
    assert n.chi2(p[0], p[1], p[2], p[3], 5e-3, p[5]) > 100
