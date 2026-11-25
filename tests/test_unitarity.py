"""55x55 tree-level unitarity matrix against closed forms: the 2HDM limit (Kanemura-Kubota-Takasugi / Akeroyd-Arhrib-Naimi eigenvalues
for the Higgs-basis potential with lambda_6 = lambda_7 = 0) and simple singlet-sector eigenvalues (complex singlet: 4 lambda_h;
h+h- <-> H_i^dagger H_i: sqrt(2) lambda_8; both: sqrt(2(l8^2+l9^2)))."""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from zee import vac
N = ['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9', 'l10', 'lh']
def c(**k):
    v = np.zeros(11)
    for a, b in k.items(): v[N.index(a)] = b
    return v
def kanemura(l1, l2, l3, l4, l5):
    a = [1.5*(l1 + l2) + s*np.sqrt(2.25*(l1 - l2)**2 + (2*l3 + l4)**2) for s in (1, -1)]
    b = [.5*((l1 + l2) + s*np.sqrt((l1 - l2)**2 + 4*l4**2)) for s in (1, -1)]
    cc = [.5*((l1 + l2) + s*np.sqrt((l1 - l2)**2 + 4*l5**2)) for s in (1, -1)]
    o = [l3 + 2*l4 - 3*l5, l3 - l5, l3 + 2*l4 + 3*l5, l3 + l4, l3 - l4]
    return max(abs(x) for x in a + b + cc + o)
def test_2hdm_limit():
    rng = np.random.default_rng(7)
    for _ in range(25):
        l = np.concatenate([rng.uniform(0.05, 3, 2), rng.uniform(-3, 3, 3)])
        assert abs(vac.unit_max(c(l1=l[0], l2=l[1], l3=l[2], l4=l[3], l5=l[4])) - kanemura(*l)) < 1e-10
def test_singlet_sector():
    assert abs(vac.unit_max(c(lh=1.3)) - 4*1.3) < 1e-10
    assert abs(vac.unit_max(c(l8=0.9)) - np.sqrt(2)*0.9) < 1e-10
    assert abs(vac.unit_max(c(l8=0.7, l9=-1.1)) - np.sqrt(2*(0.7**2 + 1.1**2))) < 1e-10
