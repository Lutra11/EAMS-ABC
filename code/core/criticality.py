import numpy as np
from numba import njit

@njit(cache=True)
def features(starts,ends,ms,timelines,lengths,offsets,counts,proc,idle,makespan):
    size=len(ms); latest=np.full(size,makespan); succ=np.full(size,-1,np.int64)
    gaps=np.zeros(size)
    for k in range(len(lengths)):
        for h in range(lengths[k]-1):
            a=timelines[k,h]; b=timelines[k,h+1]; succ[a]=b
            g=max(0.,starts[b]-ends[a])*idle[k]
            gaps[a]+=g; gaps[b]+=g
    js=np.full(size,-1,np.int64)
    for j in range(len(counts)):
        for q in range(counts[j]-1): js[offsets[j]+q]=offsets[j]+q+1
    for o in np.argsort(starts)[::-1]:
        finish=makespan
        if js[o]>=0: finish=min(finish,latest[js[o]])
        if succ[o]>=0: finish=min(finish,latest[succ[o]])
        latest[o]=finish-(ends[o]-starts[o])
    slack=np.maximum(0.,latest-starts)
    tc=1.-slack/(np.max(slack)+1e-12)
    pec=(ends-starts)*proc[ms]; pec=pec/(np.max(pec)+1e-12)
    iec=gaps/(np.max(gaps)+1e-12)
    return tc,pec,iec

def criticality(p,s,alpha=.5,beta=.7,gamma=2.):
    if s.critical is None:
        s.critical=features(s.starts,s.ends,s.ms,s.timelines,s.lengths,p.offsets,p.counts,p.proc,p.idle,s.f[0])
    tc,pec,iec=s.critical
    score=(alpha*tc+(1-alpha)*(beta*pec+(1-beta)*iec)+.02)**gamma
    return score/score.sum()
