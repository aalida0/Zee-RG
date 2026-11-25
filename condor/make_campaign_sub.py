"""Write condor/campaign.sub.
Usage: python3 make_campaign_sub.py RUN_ID [pilot|full] [WALL=1380]
  full  : 1980 chains = 780 sample chains (logL -1 [low-energy baseline, nominal tree-level boundary only], 0 [theory-prefiltered reference: nominal and
          quartic-rescaled boundaries valid at the start, no scale requirement], 6, 8, 10, 12 x 5 starts x 26 seeds)
          + 1200 maximise chains (mu_e_g, tau_mu_g, h_tau_mu x logL 8,10,12,14 x 5 starts x 20 seeds; half with walk margin 1, half with margin 0)
  pilot : 20 chains covering every mode, every start and the hard target logL=14 (run this first, on the worker nodes, and inspect it with collect_campaign.py).
Every job needs RUN_ID (an immutable checkout from make_run.sh); outputs go to $HOME/zee_env/runs/RUN_ID/out."""
import sys
RUN = sys.argv[1]; KIND = sys.argv[2] if len(sys.argv) > 2 else 'pilot'; WALL = int(sys.argv[3]) if len(sys.argv) > 3 else 1380
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
jobs = []      # (mode, logL, seed, start, margin); margin = walk prefilter logL + margin of the maximiser (0: the requested cutoff itself)
NST = 5                                           # results/campaign_starts.json: BM1, BM2 and three further starts passing the staged gate
if KIND == 'full':
    # logL = -1: low-energy baseline (nominal tree-level boundary only); logL = 0: theory-prefiltered reference (nominal + quartic-rescaled boundaries valid at the start)
    for logL in (-1, 0, 6, 8, 10, 12):
        for st in range(NST):
            for s in range(26): jobs.append(('sample', logL, st*1000 + s, st, 0))
    for obs in ('mu_e_g', 'tau_mu_g', 'h_tau_mu'):
        for logL in (8, 10, 12, 14):
            for st in range(NST):
                for s in range(20): jobs.append(('max:' + obs, logL, st*1000 + s, st, 1.0 if s < 10 else 0.0))
else:
    for logL, st in ((-1, 0), (0, 1), (10, 0), (10, 1), (10, 2), (10, 3), (10, 4), (12, 0), (12, 4)): jobs.append(('sample', logL, st*1000 + 1, st, 0))
    for m, logL, st, mg in (('mu_e_g', 10, 0, 1.0), ('mu_e_g', 10, 3, 0.0), ('mu_e_g', 14, 0, 1.0), ('mu_e_g', 14, 1, 0.0),
                            ('tau_mu_g', 10, 2, 1.0), ('tau_mu_g', 10, 4, 0.0), ('tau_mu_g', 14, 1, 1.0), ('tau_mu_g', 14, 4, 0.0),
                            ('h_tau_mu', 10, 1, 0.0), ('h_tau_mu', 12, 0, 0.0), ('h_tau_mu', 14, 1, 0.0)):
        jobs.append(('max:' + m, logL, st*1000 + 1, st, mg))
body = ''.join(f"mode={m}\nlogL={l}\nseed={sd}\nstart={st}\nmargin={mg}\nqueue\n" for m, l, sd, st, mg in jobs)
open('campaign.sub', 'w').write(head + body); print(len(jobs), 'jobs (%s), run %s' % (KIND, RUN))
