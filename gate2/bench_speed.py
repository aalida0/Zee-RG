"""Speed test for choosing where to run: times the model evaluation, the staged gate and one certification on BM1, single core,
then N parallel copies (default = all cores) to see the scaling.  Usage: ZEE_INPUTS=modern python3 -W ignore gate2/bench_speed.py [NPAR]"""
import sys, os, time, json, platform, numpy as np
os.environ.setdefault('ZEE_INPUTS', 'modern')
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(v, '1')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
from multiprocessing import Pool
from zee import hgmodel as H
import campaign as C


def work(_):
    b = json.load(open(os.path.join(ROOT, 'results', 'benchmarks_input.json')))['BM1']; x = list(b['x']); eta = np.array(b['eta'])
    t = time.time(); n = 0
    while time.time() - t < 8: H.evaluate(x, 'NO', eta=eta); n += 1
    ev = (time.time() - t) / n * 1e3
    t = time.time(); C.stage_gate(np.array(x), eta, -1.0); g = time.time() - t
    return ev, g


if __name__ == '__main__':
    npar = int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count()
    print(platform.processor() or platform.machine(), 'cores', os.cpu_count())
    ev, g = work(0); print('1 process : evaluate %.1f ms   staged gate %.2f s' % (ev, g), flush=True)
    with Pool(npar) as P: r = P.map(work, range(npar))
    print('%d processes: evaluate %.1f ms (mean)   staged gate %.2f s (mean)' % (npar, np.mean([a for a, _ in r]), np.mean([b for _, b in r])))
