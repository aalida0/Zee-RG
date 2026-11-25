"""Gate 1: HG (2016 inputs) consistency checks of the restricted general Zee model.
Restriction as in HG: f_emu = 0; Y2^{ee,emu,mue,mumu} = 0; f real after rephasing.
"""
import sys, numpy as np
from scipy.optimize import least_squares
_R = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
sys.path.insert(0, _R)
from zee import neutrino as N, observables as O, dipoles as P, stu as S
from zee.neutrino import oscillation_observables

V = 246.21965
ML = O.M_LEP
OSC = {  # HG Table 4 (NuFIT 3.0): value, sigma
    "NO": dict(s12=(0.306, .012), s13=(0.02166, .00075), s23=(0.441, .027), dm21=(7.50e-5, .19e-5), dm3=(2.524e-3, .040e-3), d=(261., 59.)),
    "IO": dict(s12=(0.306, .012), s13=(0.02179, .00076), s23=(0.587, .024), dm21=(7.50e-5, .19e-5), dm3=(-2.514e-3, .041e-3), d=(277., 46.)),
}
STU_HG = np.array([0.05, 0.09, 0.01]); SIG = np.array([.11, .13, .11])
COV = np.array([[1, .90, -.59], [.90, 1, -.83], [-.59, -.83, 1]]) * np.outer(SIG, SIG)
LCH = np.linalg.cholesky(COV)
ENT = [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]


def unpack(x):
    fe, fm = x[0], x[1]
    f = np.zeros((3, 3), complex); f[0, 2], f[1, 2] = fe, fm; f -= f.T
    Y2 = np.zeros((3, 3), complex)
    for k, (a, b) in enumerate(ENT):
        Y2[a, b] = x[2 + 2 * k] + 1j * x[3 + 2 * k]
    tb = np.exp(x[12]); s2 = x[13]
    m1 = 10 ** x[14]; m2 = m1 * np.exp(x[15])
    mu = s2 * (m2**2 - m1**2) / (np.sqrt(2) * V)
    mH, mA, cba = 10 ** x[16], 10 ** x[17], x[18]
    return f, Y2, tb, m1, m2, mu, mH, mA, cba


def osc_res(M, ordering):
    o = oscillation_observables(M, ordering)
    t = OSC[ordering]
    dd = (o["delta_deg"] - t["d"][0] + 180.) % 360. - 180.
    return np.array([(o["s12sq"] - t["s12"][0]) / t["s12"][1], (o["s13sq"] - t["s13"][0]) / t["s13"][1],
                     (o["s23sq"] - t["s23"][0]) / t["s23"][1], (o["dm21_eV2"] - t["dm21"][0]) / t["dm21"][1],
                     (o["dm3l_eV2"] - t["dm3"][0]) / t["dm3"][1], dd / t["d"][1]])


def bound_res(x, with_stu):
    f, Y2, tb, m1, m2, mu, mH, mA, cba = unpack(x)
    phi = 0.5 * np.arcsin(min(1., x[13]))  # doublet-lighter branch
    phi = np.pi / 2 - phi if x[19] > 0.5 else phi
    out = []
    def rel(val, lim):
        return 3.0 * max(0.0, np.log(max(val, 1e-300) / lim))
    for (i, j), lim in {(1, 0): 4.2e-13, (2, 0): 3.3e-8, (2, 1): 4.4e-8}.items():
        out.append(rel(P.br(i, j, f, Y2, tb, cba, mH, mA, m1, m2, phi), lim / 1.645))
    out.append(rel(O.higgs_lfv_br(1, 2, Y2, tb, cba), 0.012 / 1.949))
    out.append(rel(O.higgs_lfv_br(0, 2, Y2, tb, cba), 0.0069 / 1.949))
    out.append(rel(mu, 1500.0))
    if with_stu:
        s, t, u = S.stu(m1, m2, phi, mH, mA, cba)
        out.extend(np.linalg.solve(LCH, np.array([s, t, u]) - STU_HG))
    return np.array(out), phi


def make_res(ordering, with_stu):
    def res(x):
        f, Y2, tb, *_ = unpack(x)
        M = N.mnu_hg_leading(f, ML, Y2, tb, unpack(x)[3], unpack(x)[4], unpack(x)[5])
        b, _ = bound_res(x, with_stu)
        return np.concatenate([osc_res(M, ordering), b])
    return res

LO = np.array([-.1, -.1] + [-.1] * 10 + [np.log(.3), 0.01, 2.0, 0.01, 2.0, 2.0, -.1, 0.0])
HI = np.array([.1, .1] + [.1] * 10 + [np.log(50.), 1.0, 3.4, 3.5, 3.5, 3.5, .1, 1.0])


def run(ordering, nstart=150, seed=0, with_stu=False, starts=None):
    rng = np.random.default_rng(seed)
    best = []
    inits = starts if starts is not None else [LO + (HI - LO) * rng.random(len(LO)) for _ in range(nstart)]
    fn = make_res(ordering, with_stu)
    def prep(x0):
        x0 = np.clip(x0, LO, HI).copy()
        f, Y2, tb, m1, m2, mu, *_ = unpack(x0)
        M = N.mnu_hg_leading(f, ML, Y2, tb, m1, m2, mu)
        sv = np.linalg.svd(M, compute_uv=False)[0]
        if sv > 0:
            x0[:2] = np.clip(x0[:2] * 0.05e-9 / sv, -.1, .1)
        return x0
    for x0 in inits:
        x0 = prep(x0)
        try:
            r = least_squares(fn, np.clip(x0, LO, HI), bounds=(LO, HI), max_nfev=250, x_scale=np.maximum(np.abs(HI - LO), 1e-3) * .1)
            best.append((float(np.sum(r.fun[:6] ** 2)), float(np.sum(r.fun ** 2)), r.x))
        except Exception as e:
            print('fail', repr(e)[:80]); continue
    best.sort(key=lambda t: t[1])
    return best
