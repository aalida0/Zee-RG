import sys,pickle; sys.path.insert(0,'/home/claude/zee_fit/gate1')
import numpy as np, gate1 as G
rng=np.random.default_rng(11)
r=pickle.load(open('/tmp/gate1_stage1.pkl','rb'))['IO']
starts=[]
for a,c,x in r[:10]:
    for _ in range(150):
        y=x.copy(); y[:14]+=rng.normal(size=14)*np.maximum(np.abs(x[:14]),1e-4)*rng.choice([.05,.2,.6]); starts.append(y)
for _ in range(1500):
    x=G.LO+(G.HI-G.LO)*rng.random(len(G.LO))
    for k in range(5):
        rr=10**rng.uniform(-4,-1); th=rng.uniform(0,2*np.pi); x[2+2*k]=rr*np.cos(th); x[3+2*k]=rr*np.sin(th)
    x[0:2]=rng.choice([-1,1],2)*10**rng.uniform(-4,-1,2); starts.append(x)
b=G.run("IO",starts=starts)
pickle.dump([(a,c,x) for a,c,x in b[:20]],open('/tmp/gate1_IO_refined.pkl','wb'))
print("IO best",[(round(a,2),round(c,2)) for a,c,_ in b[:8]],flush=True)
