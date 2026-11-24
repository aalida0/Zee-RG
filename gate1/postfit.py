"""Post-fit observables for the final NO sample: B->X_s gamma (charged scalars), HiggsBounds with charged scalars, neutrino predictions.
Usage: ZEE_INPUTS=modern python gate1/postfit.py OUT.csv"""
import sys, pickle, numpy as np
sys.path.insert(0, '/home/claude/zee_fit'); sys.path.insert(0, '/home/claude/zee_fit/gate1')
import Higgs.predictions as HP
import ht
from zee import hgmodel as H, neutrino as N
V = H.V; MT, MW, MB = 172.5, 80.379, 4.18; MT160 = 163.0; GF = 1/(np.sqrt(2)*V**2)
def F7_1(y): return y*(7 - 5*y - 8*y**2)/(24*(y - 1)**3) + y**2*(3*y - 2)/(4*(y - 1)**4)*np.log(y)
def F7_2(y): return y*(3 - 5*y)/(12*(y - 1)**2) + y*(3*y - 2)/(6*(y - 1)**3)*np.log(y)
def F8_1(y): return y*(2 + 5*y - y**2)/(8*(y - 1)**3) - 3*y**2/(4*(y - 1)**4)*np.log(y)
def F8_2(y): return y*(3 - y)/(4*(y - 1)**2) - y/(2*(y - 1)**3)*np.log(y)
def bsg(sc):
    """BR(B->X_s gamma) x 1e4 with Misiak-Steinhauser (1702.04571): 3.36 - 8.22 C7 - 1.99 C8 at mu0=160 GeV.
    HG quark sector: Higgs-basis couplings of H2+ to quarks are type-I-like with A_u = A_d = tan(beta) (generic-basis tan beta)."""
    tb, c, s = sc['tb'], np.cos(sc['phi']), np.sin(sc['phi']); C7 = C8 = 0.0
    for w, m in ((c, sc['m1']), (s, sc['m2'])):
        y = MT160**2/m**2; A2 = (tb*w)**2
        C7 += A2*(F7_1(y)/3 - F7_2(y)); C8 += A2*(F8_1(y)/3 - F8_2(y))
    return 3.36 - 8.22*C7 - 1.99*C8
def charged(p, x, sc, Y):
    f, _ = H.unpack(np.array(x)); D = N.d_matrix(H.M_LEP, Y, sc['tb']); y2 = 2*f
    tb, c, s = sc['tb'], np.cos(sc['phi']), np.sin(sc['phi']); out = []
    for name, w, ws, m in (('Hp1', c, s, sc['m1']), ('Hp2', s, c, sc['m2'])):
        A = tb*w; X = p.addParticle(HP.ChargedScalar(name)); X.setMass(m)
        lep = [m/(16*np.pi)*(w**2*np.sum(abs(D[:, i])**2) + ws**2*np.sum(abs(y2[:, i])**2)) for i in range(3)]
        tbw = 3*GF*m*MT**2*A**2/(4*np.sqrt(2)*np.pi)*(1 - MT**2/m**2)**2 if m > MT + MB else 0.0
        for dec, g in zip((HP.Decay.enu, HP.Decay.munu, HP.Decay.taunu), lep): X.setDecayWidth(dec, g)
        X.setDecayWidth(HP.Decay.tb, tbw)
        brt = A**2*(1 - m**2/MT**2)**2/((1 - MW**2/MT**2)**2*(1 + 2*MW**2/MT**2)) if m < MT - MB else 0.0
        brt = brt/(1 + brt)
        X.setCxn(HP.Collider.LHC13, HP.Production.Hpmtb, HP.EffectiveCouplingCxns.ppHpmtb(HP.Collider.LHC13, m, A, A, brt))
        if brt > 0: X.setCxn(HP.Collider.LHC13, HP.Production.brtHpb, brt)
        out.append((brt, lep[2]/max(sum(lep) + tbw, 1e-300)))
    return out
if __name__ == '__main__':
    rows = []
    d = pickle.load(open('/tmp/mfilt4w_NO.pkl', 'rb')); S = d['S']; keep = {t[0] for t in pickle.load(open('/tmp/lmax_final.pkl', 'rb')) if t[4]}
    for i, r in d['res']:
        if i not in keep: continue
        x = S[i][1]; e = H.evaluate(list(x), 'NO'); sc = e['sc']; f, Y = H.unpack(np.array(x))
        M = N.mnu_hg_leading(f, H.M_LEP, Y, sc['tb'], sc['m1'], sc['m2'], sc['mu']); o = N.oscillation_observables(M, 'NO')
        m, U = N.takagi(M); k = o['masses_eV'][-1]/m[-1]
        ch = ht.run(list(x), extra=lambda pp: charged(pp, x, sc, Y))
        rows.append((i, bsg(sc), ch[1], ch[3][0][0], ch[3][0][1], abs(M[0, 0])*k, o['masses_eV'].min(), o['masses_eV'].sum(), e['obs']['m_beta'],
                     o['delta_deg'] % 360, o['s23sq'], o['s12sq'], o['s13sq']))
    np.savetxt(sys.argv[1], np.array(rows), delimiter=',', comments='',
               header='idx,BRbsg_e4,HBok_charged,BRtHpb_1,BRtaunu_1,mbb_eV,mlight_eV,summ_eV,mbeta_eV,delta_deg,s23sq,s12sq,s13sq')
    print('done', len(rows))
