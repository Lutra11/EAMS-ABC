"""Discrete POX crossover and eligible-machine uniform crossover."""
import numpy as np

def crossover(p,a,b,rng):
    subset=rng.random(p.n)<.5
    keep=subset[a.os]; child=np.full(p.size,-1,np.int64); child[keep]=a.os[keep]
    child[~keep]=b.os[~subset[b.os]]
    ms=np.where(rng.random(p.size)<.5,a.ms,b.ms)
    if rng.random()<.9:
        i,j=rng.integers(p.size,size=2); child[i],child[j]=child[j],child[i]
    for o in np.flatnonzero(rng.random(p.size)<1/p.size): ms[o]=rng.choice(p.eligible[o])
    return child,ms
