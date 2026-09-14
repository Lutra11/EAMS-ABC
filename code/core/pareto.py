import numpy as np
from numba import njit

def dominates(a,b): return bool(np.all(a<=b+1e-10) and np.any(a<b-1e-10))

@njit(cache=True)
def ranks(F):
    n=len(F); rank=np.full(n,-1,np.int64); remaining=n; level=0
    while remaining:
        selected=[]
        for i in range(n):
            if rank[i]>=0: continue
            dominated=False
            for j in range(n):
                if rank[j]>=0: continue
                if F[j,0]<=F[i,0] and F[j,1]<=F[i,1] and (F[j,0]<F[i,0] or F[j,1]<F[i,1]):
                    dominated=True; break
            if not dominated: selected.append(i)
        for i in selected: rank[i]=level
        remaining-=len(selected); level+=1
    return rank

def crowding(F):
    n=len(F); out=np.zeros(n)
    if n<=2: return np.full(n,np.inf)
    for k in range(2):
        ids=np.argsort(F[:,k]); out[ids[0]]=out[ids[-1]]=np.inf
        span=F[ids[-1],k]-F[ids[0],k]
        if span>0: out[ids[1:-1]]+=(F[ids[2:],k]-F[ids[:-2],k])/span
    return out

class Archive:
    def __init__(self,capacity=80): self.capacity=capacity; self.items=[]; self.F=np.empty((0,2))
    def add(self,s):
        if len(self.F):
            if np.any(np.all(self.F<=s.f+1e-10,axis=1)): return False
            keep=~np.all(s.f<=self.F+1e-10,axis=1)
            self.items=[x for x,k in zip(self.items,keep) if k]
        self.items.append(s); self.F=np.array([x.f for x in self.items])
        if len(self.items)>self.capacity:
            ix=int(np.argmin(crowding(self.F))); self.items.pop(ix); self.F=np.delete(self.F,ix,axis=0)
        return any(x is s for x in self.items)

def environmental(items,n):
    F=np.array([x.f for x in items]); r=ranks(F); selected=[]
    for level in range(int(max(r))+1):
        ids=np.flatnonzero(r==level)
        if len(selected)+len(ids)<=n: selected.extend(ids)
        else:
            selected.extend(ids[np.argsort(-crowding(F[ids]),kind='stable')[:n-len(selected)]]); break
    return [items[i] for i in selected]
