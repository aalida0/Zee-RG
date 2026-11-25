"""Run the robustness pieces of gate2/bench.py for the chosen benchmarks, one piece per call (resumable).
Usage: ZEE_INPUTS=modern python gate2/run_bench.py   (loops until all pieces are stored in /tmp/bench/)"""
import sys, os, json, pickle, numpy as np
_R = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, _R + '/gate2')
import bench
BM = {'BM1': '/tmp/loc/loc_mu_e_g_12_0.json', 'BM2': '/tmp/loc2/loc_mu_e_g_16_1.json', 'BM3': '/tmp/envres2/out2/env_h_tau_mu_12_23.json'}
D = '/tmp/bench'; os.makedirs(D, exist_ok=True)
G0 = np.array([0.36, 0.64, 1.05, 0.88])
def piece(name, x, eta, p):
    sc, c = bench.coup(x, eta); r = -sc['tb']; ml = min(sc['m1'], sc['mA'], sc['mH']); t0 = np.log(ml)
    if p == 'default': return bench.trajectory(c, t0, r, G0)
    if p.startswith('match'):
        f = float(p.split('_x')[1]); mu0 = max(172.5, ml*f); return dict(bench.trajectory(c, np.log(mu0), r, bench.sm_run(mu0)), mu0=mu0)
    if p == 'rtol_1e-6': return bench.trajectory(c, t0, r, G0, rtol=1e-6)
    if p == 'rtol_1e-10': return bench.trajectory(c, t0, r, G0, rtol=1e-10)
    if p == 'grid_300': return bench.trajectory(c, t0, r, G0, n_per_dec=300)
    if p == 'vacuum': return bench.deep_vacuum(c, sc, n=1500)
PIECES = ['default', 'match_x0.5', 'match_x1', 'match_x2', 'rtol_1e-6', 'rtol_1e-10', 'vacuum', 'grid_300']
if __name__ == '__main__':
    for name, f in BM.items():
        if len(sys.argv) > 1 and name not in sys.argv[1:]: continue
        b = json.load(open(f))['best']; x, eta = np.array(b['x']), np.array(b['eta'])
        for p in PIECES:
            fn = f'{D}/{name}_{p}.pkl'
            if os.path.exists(fn): continue
            pickle.dump(piece(name, x, eta, p), open(fn, 'wb')); print(name, p, flush=True)
    print('all done')
