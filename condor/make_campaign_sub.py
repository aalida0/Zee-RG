"""Write condor/campaign.sub: 2000 jobs = 800 sample-mode chains (logL 6,8,10,12 x 2 starts x 100 seeds)
+ 1200 maximise-mode chains (mu_e_g, tau_mu_g, h_tau_mu x logL 8,10,12,14 x 2 starts x 50 seeds).  WALL seconds per chain."""
import sys
WALL = int(sys.argv[1]) if len(sys.argv) > 1 else 1380
head = f"""universe = vanilla
executable = campaign_job.sh
arguments = $(mode) $(logL) $(seed) {WALL} $(start)
output = $ENV(HOME)/zee_env/logs/c.out.$(Cluster).$(Process)
error = $ENV(HOME)/zee_env/logs/c.err.$(Cluster).$(Process)
log = $ENV(HOME)/zee_env/logs/log.campaign
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = campaign_job.sh
+queue = "short"
request_cpus = 1
request_memory = 3GB
request_disk = 2GB
"""
jobs = []
for logL in (6, 8, 10, 12):
    for st in (0, 1):
        for s in range(100): jobs.append(('sample', logL, st*1000 + s, st))
for obs in ('mu_e_g', 'tau_mu_g', 'h_tau_mu'):
    for logL in (8, 10, 12, 14):
        for st in (0, 1):
            for s in range(50): jobs.append(('max:' + obs, logL, st*1000 + s, st))
body = ''.join(f"mode={m}\nlogL={l}\nseed={sd}\nstart={st}\nqueue\n" for m, l, sd, st in jobs)
open('campaign.sub', 'w').write(head + body); print(len(jobs), 'jobs')
