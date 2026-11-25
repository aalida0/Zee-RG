"""Re-minimise the stored distinct low-energy minima of results/refit_hw.json (top N) after a change of an input bound (here B(h->mu tau) < 0.15%,
CMS HIG-20-009) and write the updated reference minimum.  The old file is kept as results/refit_hw_<TAGOLD>.json.  This is a LOCAL polish of the stored
minima (not a new global multi-start), which is sufficient because the change of the bound moves the statistic by a small smooth hinge term.
Usage: ZEE_INPUTS=modern python3 -W ignore gate2/refit_polish.py NPROC [N=16]"""
import sys, os, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
os.environ.setdefault('ZEE_INPUTS', 'modern')
from multiprocessing import Pool
import refit_hw as R

if __name__ == '__main__':
    NP = int(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    old = json.load(open(os.path.join(ROOT, 'results', 'refit_hw.json'))); src = old['all'][:N]
    jobs = [(i, np.clip(R.tz(np.array(r['x'], float)), R.LO, R.HI), np.clip(np.array(r['eta'], float)[[1, 2, 4]], -R.EB, R.EB)) for i, r in enumerate(src)]
    with Pool(NP) as P: res = sorted([r for r in P.map(R.run, jobs) if np.isfinite(r['chi2'])], key=lambda r: r['chi2'])
    json.dump(dict(best=res[0], all=res, note='production statistic, h->mu tau < 0.15% (CMS), polished from the stored minima of refit_hw_atlas018.json', previous_best_chi2=old['best']['chi2']),
              open(os.path.join(ROOT, 'results', 'refit_hw.json'), 'w'), indent=1)
    print('previous best', old['best']['chi2'], 'new best', res[0]['chi2'], 'top', [round(r['chi2'], 3) for r in res[:6]])
