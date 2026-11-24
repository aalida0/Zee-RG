"""Re-check the Condor envelope points with the current code: chi2, HiggsSignals/HiggsBounds (incl. charged scalars),
and the stored quartic witness (unitarity, BFB, perturbativity along the RG flow; 80-start global-minimum test).
Usage: ZEE_INPUTS=modern python gate1/env_verify.py OUTDIR RESULT.pkl"""
import sys, glob, json, pickle, numpy as np
sys.path.insert(0, '/home/claude/zee_fit'); sys.path.insert(0, '/home/claude/zee_fit/gate1')
from multiprocessing import Pool
import lmax2 as L, m_filter_l4w as M, ht, postfit as PF
from zee import vac
from m_filter_l4w import H
M.LIM = pickle.load(open('/tmp/mbest_NO.pkl', 'rb'))[0][0] + 3.84
HSMIN = min(t[3] for t in pickle.load(open('/tmp/lmax_final.pkl', 'rb')))
def check(a):
    f, d = a; b = d['best']; x = np.array(b['x']); eta = np.array(b['eta'])
    try:
        e = H.evaluate(list(x), 'NO'); sc = e['sc']; f2, Y = H.unpack(x)
        chi, okb, _, _ = ht.run(list(x), extra=lambda p: PF.charged(p, x, sc, Y))
        c = np.zeros(11); c[0], c[2], c[3], c[4], c[5] = sc['lam1'], sc['lam3'], sc['lam4'], sc['lam5'], sc['lam6']
        c[1], c[6], c[7], c[8], c[9], c[10] = eta
        t0 = np.log(min(sc['m1'], sc['mA'], sc['mH']))
        gg = e['chi2'] + ((e['mus']['gg']*M.rgg_gen(sc, c[6], c[7], c[9])/e['rgg'] - 1.14)/0.185)**2 - ((e['mus']['gg'] - 1.14)/0.185)**2
        Lv = L.lam_valid(c, t0, -sc['tb']); gok = vac.global_ok(c, sc, nstart=80, rng=np.random.default_rng(1))[0]
        return f, d['obs'], d['logL'], float(e['obs'][d['obs']]), float(e['chi2']), bool(e['chi2'] <= M.LIM and gg <= M.LIM), bool(okb), float(chi - HSMIN), float(Lv), bool(gok)
    except Exception as ex: return f, d['obs'], d['logL'], None, None, False, False, 99, -9, False
if __name__ == '__main__':
    J = []
    for f in glob.glob(sys.argv[1] + '/*.json'):
        d = json.load(open(f))
        if 'best' in d: J.append((f, d))
    with Pool(2) as P: R = P.map(check, J, chunksize=4)
    pickle.dump(R, open(sys.argv[2], 'wb')); print('done', len(R))
