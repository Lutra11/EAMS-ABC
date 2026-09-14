import numpy as np

def initialize(p,rng,index=0):
    os=np.repeat(np.arange(p.n,dtype=np.int64),p.counts); rng.shuffle(os)
    ms=np.empty(p.size,dtype=np.int64); load=np.zeros(p.m); mode=index%4
    for o in rng.permutation(p.size):
        ks=p.eligible[o]; ts=p.times[o,ks]
        if mode==0: k=rng.choice(ks)
        elif mode==1: k=ks[np.argmin(ts+rng.random(len(ks))*.2)]
        elif mode==2: k=ks[np.argmin(ts*p.proc[ks]+rng.random(len(ks))*.2)]
        else: k=ks[np.argmin(load[ks]+ts)]
        ms[o]=k; load[k]+=p.times[o,k]
    return os,ms
