"""B -> X_s gamma from the charged scalars (type-I-like quark sector, LO functions of Ciuchini et al. hep-ph/9710335)
with Misiak-Steinhauser 1702.04571: BR x 1e4 = 3.36 - 8.22 C7 - 1.99 C8 at mu0 = 160 GeV."""
import numpy as np
MT160 = 163.0
BSG_EXP, BSG_ERR_EXP, BSG_ERR_TH = 3.49, 0.19, 0.23     # HFLAV average x 1e4, theory error
BSG_NSIGMA = 2.0


def F7_1(y): return y*(7 - 5*y - 8*y**2)/(24*(y - 1)**3) + y**2*(3*y - 2)/(4*(y - 1)**4)*np.log(y)
def F7_2(y): return y*(3 - 5*y)/(12*(y - 1)**2) + y*(3*y - 2)/(6*(y - 1)**3)*np.log(y)
def F8_1(y): return y*(2 + 5*y - y**2)/(8*(y - 1)**3) - 3*y**2/(4*(y - 1)**4)*np.log(y)
def F8_2(y): return y*(3 - y)/(4*(y - 1)**2) - y/(2*(y - 1)**3)*np.log(y)


def bsg_e4(sc):
    tb, c, s = sc['tb'], np.cos(sc['phi']), np.sin(sc['phi']); C7 = C8 = 0.0
    for w, m in ((c, sc['m1']), (s, sc['m2'])):
        y = MT160**2/m**2; A2 = (tb*w)**2
        C7 += A2*(F7_1(y)/3 - F7_2(y)); C8 += A2*(F8_1(y)/3 - F8_2(y))
    return 3.36 - 8.22*C7 - 1.99*C8


def bsg_pull(sc):
    return (bsg_e4(sc) - BSG_EXP)/np.hypot(BSG_ERR_EXP, BSG_ERR_TH)
