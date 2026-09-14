"""Four single-evaluation neighborhoods; no hidden candidate evaluations."""
import numpy as np
from .criticality import criticality

def move(p,s,rng,operator,guided=True,alpha=.5,gap=True,uniform_selection=False):
    os=s.os.copy(); ms=s.ms.copy()
    prob=criticality(p,s,alpha) if guided and not uniform_selection else None
    o=int(rng.choice(p.size,p=prob)); a=int(np.flatnonzero(s.order==o)[0])
    if operator<2:
        if operator==0:
            b=int(rng.integers(p.size)); os[a],os[b]=os[b],os[a]
        else:
            b=int(rng.integers(p.size)); os=np.insert(np.delete(os,a),b,os[a])
    elif operator==2:
        ks=p.eligible[o]; ks=ks[ks!=ms[o]]
        if len(ks):
            if guided:
                t=p.times[o,ks]; e=t*p.proc[ks]
                load=np.bincount(ms,weights=s.ends-s.starts,minlength=p.m); load[ms[o]]-=s.ends[o]-s.starts[o]
                w=rng.random(); score=w*t/(max(t)+1e-12)+(1-w)*e/(max(e)+1e-12)+.2*load[ks]/(max(load)+1e-12)
                k=int(ks[np.argmin(score)]) if rng.random()<.8 else int(rng.choice(ks))
            else: k=int(rng.choice(ks))
            ms[o]=k
    else:
        # A gap supplies a proposal only. Destination and donor energy are evaluated globally.
        candidates=[]
        if gap:
            for k in range(p.m):
                ids=s.timelines[k,:s.lengths[k]]
                for left,right in zip(ids[:-1],ids[1:]):
                    length=s.starts[right]-s.ends[left]
                    if length>0: candidates.append((length*p.idle[k],k,int(left),int(right)))
        if candidates:
            candidates.sort(reverse=True); _,k,left,right=candidates[int(rng.integers(min(4,len(candidates))))]
            feasible=np.flatnonzero((p.times[:,k]>0)&(p.times[:,k]<=s.starts[right]-s.ends[left])&(ms!=k))
            if len(feasible):
                o=int(rng.choice(feasible)); a=int(np.flatnonzero(s.order==o)[0]); b=int(np.flatnonzero(s.order==right)[0])
                ms[o]=k; os=np.insert(np.delete(os,a),b-(a<b),os[a])
            else:
                a=int(np.flatnonzero(s.order==right)[0]); b=int(np.flatnonzero(s.order==left)[0]); os=np.insert(np.delete(os,a),b,os[a])
        else:
            b=int(rng.integers(p.size)); os=np.insert(np.delete(os,a),b,os[a]); ms[o]=int(rng.choice(p.eligible[o]))
    return os.astype(np.int64),ms

def reconstruct(p,s,rng,ratio=.2,alpha=.5):
    os=s.os.copy(); ms=s.ms.copy(); count=max(1,int(p.size*ratio))
    ids=rng.choice(p.size,size=count,replace=False,p=criticality(p,s,alpha))
    # Selected OS positions are permuted; all unselected positions are preserved.
    positions=np.flatnonzero(np.isin(s.order,ids)); values=os[positions].copy(); rng.shuffle(values); os[positions]=values
    for o in ids:
        ks=p.eligible[o]; t=p.times[o,ks]; e=t*p.proc[ks]; w=rng.random()
        score=w*t/(max(t)+1e-12)+(1-w)*e/(max(e)+1e-12)
        ms[o]=ks[np.argmin(score)] if rng.random()<.7 else rng.choice(ks)
    return os,ms
