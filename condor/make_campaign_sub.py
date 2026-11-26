"""Write condor/campaign.sub.
Usage: python3 make_campaign_sub.py RUN_ID [pilot|pilot2|full|topup] [WALL=2400]
  topup  : the `full` list minus the jobs that already have an output in $HOME/zee_env/runs/RUN_ID/out (same RUN_ID, same frozen checkout); writes campaign.sub for the
           remainder and expected_jobs_union_RUN_ID.json = existing outputs + new jobs (use it as EXPECT_FILE). For a run launched with an older job list.
  full   : job list from the pilot-1 evidence (see SAMPLE_SEEDS / MAX_SEEDS below): sample chains at logL -1 [low-energy baseline, nominal tree-level
           boundary only], 0 [theory-prefiltered reference], 6, 8, 10, 12 and maximise chains (mu_e_g, tau_mu_g, h_tau_mu) at logL 8, 10, 12, 14.
           Seeds are allocated per (logL, start): full quotas where a start can stand inside the target, small probe quotas where pilot 1 showed
           (or the start score suggests) that the chain cannot climb to the cutoff (start 2; start 4 at logL 12; logL 14 for every start).
  pilot  : the first 20 chains (every mode, every start, hard target logL=14); already run as pilot1.
  pilot2 : 11 chains re-testing what pilot 1 flagged (new start/logL pairs, the throttled certification of maximiser bests, a second seed of the
           low-acceptance sample chain); same seeds as pilot 1 where a comparison is wanted.
Every job needs RUN_ID (an immutable checkout from make_run.sh); outputs go to $HOME/zee_env/runs/RUN_ID/out."""
import sys, json
RUN = sys.argv[1]; KIND = sys.argv[2] if len(sys.argv) > 2 else 'pilot'; WALL = int(sys.argv[3]) if len(sys.argv) > 3 else 2400
head = f"""universe = vanilla
executable = campaign_job.sh
arguments = $(mode) $(logL) $(seed) {WALL} $(start) $(margin)
environment = "RUN_ID={RUN}"
output = $ENV(HOME)/zee_env/logs/{RUN}.out.$(Cluster).$(Process)
error = $ENV(HOME)/zee_env/logs/{RUN}.err.$(Cluster).$(Process)
log = $ENV(HOME)/zee_env/logs/{RUN}.log
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = campaign_job.sh
+queue = "short"
request_cpus = 1
request_memory = 3GB
request_disk = 2GB
"""
NST = 5                                           # results/campaign_starts.json: BM1, BM2 and three further starts passing the staged gate
# Coarse reach scores of the five starts (campaign.coarse_reach with the rescaled boundaries, 2026-11-26): 10.75, 12.51, 6.75, 11.48, 11.77.
# Pilot 1+2: start 2 cannot climb to logL 8 or 10 (sample and maximiser chains identical, ~1400 climbing moves, no stored point); start 4 stalls at logL 12 (82 accepted moves);
# no start climbs to logL 14 (starts 0, 1, 3, 4: 69, 18, 12, 82 accepted moves); start 3 DOES work at logL 12 (pilot 2: 40 stored points, post-burn acceptance 0.26).
# Dead pairs get probe quotas only (or none); the freed chains go to the productive pairs.
SAMPLE_SEEDS = {-1: (26, 26, 26, 26, 26), 0: (26, 26, 26, 26, 26), 6: (26, 26, 26, 26, 26),            # seeds per start 0..4
                8: (36, 36, 2, 36, 36), 10: (46, 46, 0, 46, 46), 12: (70, 70, 0, 40, 3)}
MAX_SEEDS = {8: (22, 22, 2, 22, 22), 10: (36, 36, 0, 36, 36), 12: (46, 46, 0, 30, 3), 14: (0, 2, 0, 0, 0)}  # per observable; margin 1 only for the productive pairs at logL <= 10
MARGIN1 = {8: (0, 1, 3, 4), 10: (0, 1, 3, 4)}                                                          # (logL: starts whose chains may use walk margin 1; half of their seeds do)
jobs = []      # (mode, logL, seed, start, margin); margin = walk prefilter logL + margin of the maximiser (0: the requested cutoff itself)
if KIND in ('full', 'topup'):
    for logL, ns in SAMPLE_SEEDS.items():
        for st in range(NST):
            for s in range(ns[st]): jobs.append(('sample', logL, st*1000 + s, st, 0))
    for obs in ('mu_e_g', 'tau_mu_g', 'h_tau_mu'):
        for logL, ns in MAX_SEEDS.items():
            for st in range(NST):
                for s in range(ns[st]):
                    m1 = st in MARGIN1.get(logL, ()) and s < ns[st]//2
                    jobs.append(('max:' + obs, logL, st*1000 + s, st, 1.0 if m1 else 0.0))
elif KIND == 'pilot2':
    for logL, st, s in ((8, 2, 1), (8, 0, 1), (12, 1, 1), (12, 3, 1), (-1, 0, 2)): jobs.append(('sample', logL, st*1000 + s, st, 0))
    for m, logL, st, s in (('mu_e_g', 10, 3, 1), ('h_tau_mu', 10, 1, 1), ('mu_e_g', 12, 0, 1), ('tau_mu_g', 12, 1, 1), ('h_tau_mu', 14, 3, 1), ('tau_mu_g', 8, 2, 1)):
        jobs.append(('max:' + m, logL, st*1000 + s, st, 0.0))
else:
    for logL, st in ((-1, 0), (0, 1), (10, 0), (10, 1), (10, 2), (10, 3), (10, 4), (12, 0), (12, 4)): jobs.append(('sample', logL, st*1000 + 1, st, 0))
    for m, logL, st, mg in (('mu_e_g', 10, 0, 1.0), ('mu_e_g', 10, 3, 0.0), ('mu_e_g', 14, 0, 1.0), ('mu_e_g', 14, 1, 0.0),
                            ('tau_mu_g', 10, 2, 1.0), ('tau_mu_g', 10, 4, 0.0), ('tau_mu_g', 14, 1, 1.0), ('tau_mu_g', 14, 4, 0.0),
                            ('h_tau_mu', 10, 1, 0.0), ('h_tau_mu', 12, 0, 0.0), ('h_tau_mu', 14, 1, 0.0)):
        jobs.append(('max:' + m, logL, st*1000 + 1, st, mg))
if KIND == 'topup':
    import glob, os
    done = {}
    for f in glob.glob(os.path.expanduser('~/zee_env/runs/%s/out/camp_*.json' % RUN)):
        d = json.load(open(f)); done[(d['mode'], float(d['logL']), d['seed'])] = dict(mode=d['mode'], logL=float(d['logL']), seed=d['seed'], start=d['start'], margin=d.get('margin', 0.0))
    todo = [j for j in jobs if (j[0], float(j[1]), j[2]) not in done]
    print('%d existing outputs; the full list wants %d, of which %d are new' % (len(done), len(jobs), len(todo)))
    union = list(done.values()) + [dict(mode=m, logL=float(l), seed=sd, start=st, margin=mg) for m, l, sd, st, mg in todo]
    json.dump(union, open('expected_jobs_union_%s.json' % RUN, 'w'), indent=0); jobs = todo
body = ''.join(f"mode={m}\nlogL={l}\nseed={sd}\nstart={st}\nmargin={mg}\nqueue\n" for m, l, sd, st, mg in jobs)
open('campaign.sub', 'w').write(head + body)
import json; json.dump([dict(mode=m, logL=float(l), seed=sd, start=st, margin=mg) for m, l, sd, st, mg in jobs], open('expected_jobs_%s_%s.json' % (KIND, RUN), 'w'), indent=0)   # exact expected job IDs for the collector (EXPECT_FILE)
print(len(jobs), 'jobs (%s), run %s, wall %d s' % (KIND, RUN, WALL))
