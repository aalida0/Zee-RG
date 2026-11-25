"""Independent check of the H+-W-+ gamma Barr-Zee sets against GM2Calc 2.x (full two-loop aligned 2HDM a_mu).
Needs GM2Calc (build it and set GM2CALC=/path/to/gm2calc.x, GM2INPUT=/path/to/input/example.thdm).
Setup: exact alignment, tan(beta)=1, lambda6=lambda7=0, M^2 = M_H+^2, aligned Yukawas zeta_u = zeta_d, flavour-universal zeta_l
(GM2Calc Yukawa type 5; output block 'Delta(g-2)_muon/2' is a_mu).  The code side uses phi=0, a decoupled second charged state,
Dd = sqrt2 m_l zeta_l/V, Lam_H = -Dd/sqrt2, Lam_A = i Dd/sqrt2, quark couplings zeta (A: -zeta_u, +zeta_d), lam3 = (m_h^2 + 2 M_H+^2 - 2 M^2)/V^2.
Result (results/gm2calc_check.json): code with HW_SCALE=+1 is 0.90-0.98 of GM2Calc over M_H+ = 90-400 GeV (GM2Calc's own scheme
choice 'running couplings' moves its number by +7%); HW_SCALE = 0 gives 0.53-0.65 of it and HW_SCALE = -1 gives 0.15-0.36."""
import os, re, subprocess, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from zee import dipfull as D, V_EW as V
from zee.observables import M_LEP
M = np.asarray(M_LEP, float); MH = 125.09
BIN = os.environ.get('GM2CALC', '/tmp/gm2src/build/bin/gm2calc.x'); INP = os.environ.get('GM2INPUT', '/tmp/gm2src/input/example.thdm')


def gm2calc(mHp, mH, mA, M2, zu, zd, zl, tb=1.0, loops=2, running=1):
    t = open(INP).read()
    def setl(block, idx, val):
        nonlocal t
        m = re.search(r'(Block %s\b.*?\n)(.*?)(?=\nBlock |\Z)' % block, t, re.S)
        body = [('    %s   %.12g   #' % (idx, val)) if ln.split()[:1] == [str(idx)] else ln for ln in m.group(2).split('\n')]
        t = t[:m.start(2)] + '\n'.join(body) + t[m.end(2):]
    for i, v in ((3, tb), (16, 0.0), (17, 0.0), (18, M2 * tb / (1 + tb**2)), (20, 1.0), (21, zu), (22, zd), (23, zl), (24, 5)):
        setl('MINPAR', i, v)
    for i, v in ((25, MH), (35, mH), (36, mA), (37, mHp)):
        setl('MASS', i, v)
    setl('GM2CalcConfig', 1, loops); setl('GM2CalcConfig', 6, running)
    open('/tmp/_gm2.thdm', 'w').write(t)
    o = subprocess.run([BIN, '--thdm-input-file=/tmp/_gm2.thdm'], capture_output=True, text=True).stdout
    return float(re.search(r'^\s*0\s+(\S+)\s+# Delta', o, re.M).group(1))


def quark_aligned(sc):
    zu, zd = sc['zu'], sc['zd']
    return {"t": (172.5, 3, 2 / 3, 0.5, {"h": (172.5 / V, 0.0), "H": (-172.5 / V * zu, 0.0), "A": (0.0, -zu * 172.5 / V)}),
            "b": (4.18, 3, -1 / 3, -0.5, {"h": (4.18 / V, 0.0), "H": (-4.18 / V * zd, 0.0), "A": (0.0, zd * 4.18 / V)})}


def code(mHp, mH, mA, M2, zu, zd, zl, hw):
    D.quark_couplings = quark_aligned; D.HW_SCALE = hw
    Dd = np.diag(np.sqrt(2) * M * zl / V).astype(complex); Z = np.zeros((3, 3), complex)
    sc = dict(phi=0.0, m1=mHp, m2=2e3, sba=1.0, cba=0.0, tb=-zu, zu=zu, zd=zd, lam3=(MH**2 + 2 * mHp**2 - 2 * M2) / V**2, mu=0.0)
    Lam = {'h': np.diag(M / V).astype(complex), 'H': -Dd / np.sqrt(2), 'A': 1j * Dd / np.sqrt(2)}
    AL, AR = D.amplitudes(Lam, {'h': MH, 'H': mH, 'A': mA}, Z, Dd, sc, D.charged_trilinears(sc))
    L0, R0 = D.sm_reference(sc)
    return D.diagonal_moments(AL, AR)[0][1] - D.diagonal_moments(L0, R0)[0][1]


def heavy():
    """extension to the benchmark mass range (report v10): M_H+ = 600, 800, 900, 1000 GeV with H, A at M_H+ + (20, 40) GeV, tan(beta) = 1, several couplings -> results/gm2calc_check_heavy.json"""
    rows = []
    for mHp in (600, 800, 900, 1000):
        for zl in (10, 30):
            for zu, zd in ((-1, -1), (-2, -2)):
                g = gm2calc(mHp, mHp + 20, mHp + 40, mHp**2, zu, zd, zl)
                r = dict(mHp=mHp, zl=zl, zu=zu, zd=zd, gm2calc=g, code={str(h): code(mHp, mHp + 20, mHp + 40, mHp**2, zu, zd, zl, h) for h in (1, 0, -1)}); r['ratio'] = {h: v/g for h, v in r['code'].items()}; rows.append(r)
                print(mHp, zl, zu, zd, 'GM2Calc %.3e' % g, 'ratio HW=+1/0/-1: ' + ' '.join('%.2f' % r['ratio'][h] for h in ('1', '0', '-1')), flush=True)
    json.dump(dict(rows=rows, gm2calc_commit=os.environ.get('GM2COMMIT', '372ee01'), note='aligned 2HDM, type 5, exact alignment; diagonal Delta a_mu only'), open(os.path.join(os.path.dirname(__file__), '..', 'results', 'gm2calc_check_heavy.json'), 'w'), indent=1)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'heavy': heavy(); sys.exit()
    for mHp, mH, mA in [(90, 150, 170), (200, 250, 250), (300, 300, 300), (400, 420, 430)]:
        g = gm2calc(mHp, mH, mA, mHp**2, -1, -1, 30)
        print(mHp, mH, mA, 'GM2Calc %.3e' % g, ' code HW=0/+1/-1: ' + ' '.join('%.3e' % code(mHp, mH, mA, mHp**2, -1, -1, 30, h) for h in (0, 1, -1)))
