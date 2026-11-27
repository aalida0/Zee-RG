"""Re-evaluate the archived verified campaign points and the 59 finalists with the production acceptance gate (gate2/accept.accept) under the
CORRECTED oblique-T statistic and a stated reference minimum (V13 referee P0).  It is a re-evaluation of the stored parameter vectors: the RG/matching
classification of a vector does not depend on the likelihood and is carried over from the campaign record; chi^2, the hard low-energy limits, the DY screen,
HiggsTools (HB/HS/h->tau mu) and the vacuum search are recomputed.  It is NOT a new campaign and the old PASS label is not reused: output keys 'status_new'/'chi2_new'
sit next to the archived 'status_old'/'chi2_old'.
Usage: ZEE_INPUTS=modern ZEE_DATA=<hbdataset/hsdataset dir> [ZEE_STU=rpp2026 ZEE_Z95=1.95996 ZEE_SUMM_LIMIT=0.0642] python3 -W ignore gate2/reeval_archive.py CMIN NPROC OUT.json [--limit N]
(set the same environment variables as the reference fit that produced CMIN; the environment is stored in the output)"""
import sys, os, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
from multiprocessing import Pool
import numpy as np
import accept as AC
from zee import hgmodel as H


def one(a):
    key, x, eta, cmin = a
    t0 = time.time()
    try:
        r = AC.accept(x, eta, cmin, logL=None, collider=True)
        return dict(key=key, x=list(map(float, x)), eta=list(map(float, eta)), chi2_new=r['chi2'], chi2_parts=r.get('chi2_parts'), status_new=r['status'], failed_new=r.get('failed_cuts', []),
                    DY=r.get('DY', {}).get('status'), HS=r.get('collider', {}).get('HS_chi2'), HB=r.get('collider', {}).get('HB_allowed'), seconds=time.time() - t0)
    except Exception as ex:
        return dict(key=key, x=list(map(float, x)), eta=list(map(float, eta)), status_new='ERROR', error='%s: %s' % (type(ex).__name__, ex))


if __name__ == '__main__':
    cmin = float(sys.argv[1]); NP = int(sys.argv[2]); out = sys.argv[3]
    lim = int(sys.argv[sys.argv.index('--limit') + 1]) if '--limit' in sys.argv else None
    S = json.load(open(os.path.join(ROOT, 'results', 'prod1', 'campaign_summary.json'))); F = json.load(open(os.path.join(ROOT, 'results', 'prod1', 'campaign_finalists.json')))
    pts = {}; meta = {}
    for i, v in enumerate(S['verified']):
        k = tuple(np.round(v['x'], 12)) + tuple(np.round(v['eta'], 12))
        if k not in pts: pts[k] = ('V%d' % i, v['x'], v['eta']); meta[k] = dict(group=v.get('group'), chi2_old=v.get('chi2'), status_old=v.get('status'), reach_class=v.get('reach_class'), screen13=v.get('screen13'), obs=v.get('obs'))
    for name, v in F.items():
        k = tuple(np.round(v['x'], 12)) + tuple(np.round(v['eta'], 12))
        if k not in pts: pts[k] = (name, v['x'], v['eta']); meta[k] = dict(group='finalist')
        else: meta[k]['also_finalist'] = name
    jobs = [(p[0], p[1], p[2], cmin) for p in pts.values()]; keys = list(pts)
    if lim: jobs, keys = jobs[:lim], keys[:lim]
    t0 = time.time(); print('points', len(jobs), 'cmin', cmin, flush=True)
    with Pool(NP) as P: R = P.map(one, jobs, chunksize=1)
    for k, r in zip(keys, R): r.update(meta[k])
    env = {k: v for k, v in os.environ.items() if k.startswith('ZEE_')}
    json.dump(dict(cmin=cmin, env=env, n=len(R), records=R), open(out, 'w'), indent=1)
    from collections import Counter
    print('done %.0fs' % (time.time() - t0), Counter(r['status_new'] for r in R), flush=True)
