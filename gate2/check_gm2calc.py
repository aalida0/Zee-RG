"""Independent check of the H+-W-+ gamma Barr-Zee sets against GM2Calc 2.x (full two-loop aligned 2HDM a_mu).
Needs GM2Calc (build it and set GM2CALC=/path/to/gm2calc.x, GM2INPUT=/path/to/input/example.thdm).
Setup: exact alignment, tan(beta)=1, lambda6=lambda7=0, M^2 = M_H+^2, aligned Yukawas zeta_u = zeta_d, flavour-universal zeta_l
(GM2Calc Yukawa type 5; output block 'Delta(g-2)_muon/2' is a_mu).  The code side uses phi=0, a decoupled second charged state,
Dd = sqrt2 m_l zeta_l/V, Lam_H = -Dd/sqrt2, Lam_A = i Dd/sqrt2, quark couplings zeta (A: -zeta_u, +zeta_d), lam3 = (m_h^2 + 2 M_H+^2 - 2 M^2)/V^2.
Result (results/gm2calc_check.json): code with HW_SCALE=+1 is 0.90-0.98 of GM2Calc over M_H+ = 90-400 GeV (GM2Calc's own scheme
choice 'running couplings' moves its number by +7%); HW_SCALE = 0 gives 0.53-0.65 of it and HW_SCALE = -1 gives 0.15-0.36."""
import os, re, subprocess, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zee import dipfull as D, V_EW as V
from zee.observables import M_LEP
M = np.asarray(M_LEP, float); MH = 125.09
BIN = os.environ.get('GM2CALC', '/tmp/gm2src/build/bin/gm2calc.x'); INP = os.environ.get('GM2INPUT', '/tmp/gm2src/input/example.thdm')


def gm2calc(mHp, mH, mA, M2, zu, zd, zl, tb=1.0, loops=2, running=1, sba=1.0, l6=0.0, l7=0.0, m122=None):
    t = open(INP).read()
    def setl(block, idx, val):
        nonlocal t
        m = re.search(r'(Block %s\b.*?\n)(.*?)(?=\nBlock |\Z)' % block, t, re.S)
        body = [('    %s   %.12g   #' % (idx, val)) if ln.split()[:1] == [str(idx)] else ln for ln in m.group(2).split('\n')]
        t = t[:m.start(2)] + '\n'.join(body) + t[m.end(2):]
    for i, v in ((3, tb), (16, l6), (17, l7), (18, M2 * tb / (1 + tb**2) if m122 is None else m122), (20, sba), (21, zu), (22, zd), (23, zl), (24, 5)):
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


GM2V = 242.18397      # GM2Calc's v (running couplings scheme, SM inputs of the example file), read from the lambda printer below


def offalign(mHp, mH, mA, sba, l6, l7, tb, m122, zu, zl, hw=1.0):
    """Away from alignment (c_{b-a} != 0): lambda_1..5 of the Z2-basis potential are solved from the spectrum (gate2/thdm_basis.py, which reproduces GM2Calc's own lambdas
    to 1e-9 for the same inputs), rotated to the Higgs basis (Z3, Z7 fitted from the quartic form), and fed to the code with the neutral couplings
    h: s_ba m/v + c_ba Dd/sqrt2, H: c_ba m/v - s_ba Dd/sqrt2, A: i Dd/sqrt2 and the quark couplings h: (s_ba + c_ba zeta), H: (c_ba - s_ba zeta), A: -zeta_u / +zeta_d.
    The code is run with v = GM2Calc's v so that all normalisations coincide."""
    import thdm_basis as T
    lam = list(T.invert(mHp, mH, mA, sba, l6, l7, tb, m122, v=GM2V)) + [l6, l7]; Z = T.higgs_basis(lam, tb); ph = T.physical(lam, tb, m122, GM2V)
    sb, cb = ph['sba'], ph['cba']; V0 = D.V
    def quarks(sc):
        out = {}
        for q, (m, nc, Q, T3, zeta, sgn) in {'t': (172.5, 3, 2/3, .5, zu, -1), 'b': (4.18, 3, -1/3, -.5, zu, +1)}.items():
            out[q] = (m, nc, Q, T3, {"h": (m/GM2V*(sb + cb*zeta), 0.0), "H": (m/GM2V*(cb - sb*zeta), 0.0), "A": (0.0, sgn*zeta*m/GM2V)})
        return out
    D.V = GM2V; D.quark_couplings = quarks; D.HW_SCALE = hw
    try:
        Dd = np.diag(np.sqrt(2) * M * zl / GM2V).astype(complex); Zm = np.zeros((3, 3), complex)
        sc = dict(phi=0.0, m1=mHp, m2=2e3, sba=sb, cba=cb, tb=-zu, zu=zu, zd=zu, lam3=Z[2], mu=0.0)
        Lam = {'h': sb*np.diag(M/GM2V).astype(complex) + cb*Dd/np.sqrt(2), 'H': cb*np.diag(M/GM2V).astype(complex) - sb*Dd/np.sqrt(2), 'A': 1j*Dd/np.sqrt(2)}
        tri = D.charged_trilinears(sc, l7=Z[6])
        AL, AR = D.amplitudes(Lam, {'h': ph['mh'], 'H': ph['mH'], 'A': ph['mA']}, Zm, Dd, sc, tri)
        L0, R0 = D.sm_reference(sc)
        return D.diagonal_moments(AL, AR)[0][1] - D.diagonal_moments(L0, R0)[0][1], dict(Z3=Z[2], Z7=Z[6], Z6=Z[5], cba=cb, masses=ph)
    finally: D.V = V0


def offalign_scan():
    rows = []
    for mHp, dH, dA in ((300, 30, 10), (600, 30, 10), (800, 40, 20)):
        for sba in (0.9999, 0.999, 0.99, 0.95):
            for tb, l6, l7, zu, zl in ((1.0, 0.0, 0.0, -1, 30), (1.0, 0.2, -0.1, -1, 30), (2.0, -0.2, 0.15, -2, 20), (0.5, 0.1, 0.3, -1, -30)):
                M2 = mHp**2; m122 = M2 * tb / (1 + tb**2)
                g = gm2calc(mHp, mHp + dH, mHp + dA, M2, zu, zu, zl, tb=tb, sba=sba, l6=l6, l7=l7, m122=m122)
                try: c = {str(h): offalign(mHp, mHp + dH, mHp + dA, sba, l6, l7, tb, m122, zu, zl, h) for h in (1.0, 0.0, -1.0)}
                except Exception as e: print('fail', mHp, sba, tb, e); continue
                r = dict(mHp=mHp, sba=sba, cba=c['1.0'][1]['cba'], tb=tb, l6=l6, l7=l7, zu=zu, zl=zl, gm2calc=g, code={h: v[0] for h, v in c.items()}, ratio={h: v[0]/g for h, v in c.items()}, Z3=c['1.0'][1]['Z3'], Z7=c['1.0'][1]['Z7'])
                rows.append(r); print(mHp, sba, tb, l6, l7, zu, zl, 'GM2Calc %.3e' % g, 'ratio +1/0/-1: ' + ' '.join('%.2f' % r['ratio'][h] for h in ('1.0', '0.0', '-1.0')), flush=True)
    json.dump(dict(rows=rows, v_gm2calc=GM2V, note='aligned type-5 THDM away from alignment; diagonal Delta a_mu only; code run with v = GM2Calc v'), open(os.path.join(os.path.dirname(__file__), '..', 'results', 'gm2calc_check_offalign.json'), 'w'), indent=1)


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
    if len(sys.argv) > 1 and sys.argv[1] == 'offalign': offalign_scan(); sys.exit()
    for mHp, mH, mA in [(90, 150, 170), (200, 250, 250), (300, 300, 300), (400, 420, 430)]:
        g = gm2calc(mHp, mH, mA, mHp**2, -1, -1, 30)
        print(mHp, mH, mA, 'GM2Calc %.3e' % g, ' code HW=0/+1/-1: ' + ' '.join('%.3e' % code(mHp, mH, mA, mHp**2, -1, -1, 30, h) for h in (0, 1, -1)))
