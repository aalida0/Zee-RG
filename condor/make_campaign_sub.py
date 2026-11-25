"""Write condor/campaign.sub.
Usage: python3 make_campaign_sub.py RUN_ID [pilot|full] [WALL=1380]
  full  : 2000 chains = 800 sample chains (logL 0 [low-energy baseline: no high-scale requirement], 6, 8, 10, 12 x 5 starts x 32 seeds)
          + 1200 maximise chains (mu_e_g, tau_mu_g, h_tau_mu x logL 8,10,12,14 x 5 starts x 20 seeds)
  pilot : 20 chains covering every mode and every start (run this first, on the worker nodes, and inspect it with collect_campaign.py).
Every job needs RUN_ID (an immutable checkout from make_run.sh); outputs go to $HOME/zee_env/runs/RUN_ID/out."""
import sys
RUN = sys.argv[1]; KIND = sys.argv[2] if len(sys.argv) > 2 else 'pilot'; WALL = int(sys.argv[3]) if len(sys.argv) > 3 else 1380
head = f"""universe = vanilla
executable = campaign_job.sh
arguments = $(mode) $(logL) $(seed) {WALL} $(start)
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
jobs = []
NST = 5                                           # results/campaign_starts.json: BM1, BM2 and three further starts passing the staged gate
if KIND == 'full':
    for logL in (0, 6, 8, 10, 12):
        for st in range(NST):
            for s in range(32): jobs.append(('sample', logL, st*1000 + s, st))
    for obs in ('mu_e_g', 'tau_mu_g', 'h_tau_mu'):
        for logL in (8, 10, 12, 14):
            for st in range(NST):
                for s in range(20): jobs.append(('max:' + obs, logL, st*1000 + s, st))
else:
    for logL in (0, 10):
        for st in range(NST): jobs.append(('sample', logL, st*1000 + 1, st))
    for obs in ('mu_e_g', 'tau_mu_g', 'h_tau_mu'):
        for st in (0, 2, 3): jobs.append(('max:' + obs, 10, st*1000 + 1, st))
    jobs.append(('sample', 12, 1, 0))
body = ''.join(f"mode={m}\nlogL={l}\nseed={sd}\nstart={st}\nqueue\n" for m, l, sd, st in jobs)
open('campaign.sub', 'w').write(head + body); print(len(jobs), 'jobs (%s), run %s' % (KIND, RUN))
