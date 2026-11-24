"""RG input sensitivity: stable-to-Lambda fractions of region points with the old fixed gauge/top inputs vs SM-evolved
inputs from m_t (g'=0.3583, g2=0.6479, g3=1.1666, y_t=0.9369 at mu=172.5 GeV, MSbar), and y_t -/+5%.
Usage: ZEE_INPUTS=modern python rg_sens.py STRIDE NPROC"""
import sys, pickle, numpy as np
sys.path.insert(0, '/home/claude/zee_fit'); sys.path.insert(0, '/home/claude/zee_fit/gate1')
from multiprocessing import Pool
from scipy.integrate import solve_ivp
from zee import rgfast
import m_filter_l4w as M
from m_filter_l4w import H
BS = np.array([41/6, -19/6, -7.0]); K = 1/(16*np.pi**2)
def sm_inputs(t0, ytscale=1.0):
    def f(t, z):
        g1, g2, g3, yt = z
        return np.concatenate([K*BS*np.array([g1, g2, g3])**3, [K*yt*(4.5*yt**2 - 8*g3**2 - 2.25*g2**2 - 17/12*g1**2)]])
    s = solve_ivp(f, (np.log(172.5), t0), [0.3583, 0.6479, 1.1666, 0.9369], rtol=1e-9, atol=1e-11).y[:, -1]
    return s[:3], s[3]*ytscale
VAR = None
_run0 = rgfast.run
def run_var(c0, t0, t1=np.log(1e19), g=None, yt=None, **kw):
    if VAR is None: return _run0(c0, t0, t1=t1, **kw)
    gg, y = sm_inputs(t0, VAR)
    return _run0(c0, t0, t1=t1, g=gg, yt=y, **kw)
rgfast.run = run_var
def job(a):
    global VAR
    var, (i, x) = a; VAR = var
    return var, M.one((i, x))[1]
if __name__ == '__main__':
    stride, npr = int(sys.argv[1]), int(sys.argv[2])
    S = pickle.load(open('/tmp/lam4wide_NO.pkl', 'rb'))
    best = pickle.load(open('/tmp/mbest_NO.pkl', 'rb'))[0][0]; M.LIM = best + 3.84
    pts = [(i, S[i][1]) for i in range(0, len(S), stride) if S[i][0] < best + 3.84]
    print(len(pts), 'points', flush=True)
    jobs = [(v, p) for p in pts for v in (None, 1.0, 0.95, 1.05)]
    with Pool(npr) as P: R = P.map(job, jobs, chunksize=1)
    pickle.dump(R, open('/tmp/rgsens.pkl', 'wb'))
    for v in (None, 1.0, 0.95, 1.05):
        r = [o for vv, o in R if vv == v]
        print(v, len(r), {L: round(np.mean([o[L] for o in r]), 3) for L in (1e4, 1e8, 1e16)}, 'ew', round(np.mean([o['ew'] for o in r]), 3))
