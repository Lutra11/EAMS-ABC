"""Earliest feasible insertion decoder and independent feasibility audit."""
from dataclasses import dataclass
import numpy as np
from numba import njit

@njit(cache=True)
def decode_arrays(os,ms,offsets,times,proc,idle,n,m):
    size=len(ms); starts=np.zeros(size); ends=np.zeros(size)
    counters=np.zeros(n,np.int64); ready=np.zeros(n)
    timelines=np.full((m,size),-1,np.int64); lengths=np.zeros(m,np.int64)
    order=np.zeros(size,np.int64)
    for t in range(size):
        job=os[t]; op=offsets[job]+counters[job]; counters[job]+=1; order[t]=op
        machine=ms[op]; duration=times[op,machine]; start=ready[job]
        at=lengths[machine]
        for h in range(lengths[machine]):
            other=timelines[machine,h]
            if start+duration<=starts[other]+1e-9: at=h; break
            start=max(start,ends[other])
        for h in range(lengths[machine],at,-1): timelines[machine,h]=timelines[machine,h-1]
        timelines[machine,at]=op; lengths[machine]+=1
        starts[op]=start; ends[op]=start+duration; ready[job]=ends[op]
    # Backward compaction: keep every machine's last completion fixed and
    # right-shift earlier operations within job/machine successor bounds.
    # Processing energy and makespan stay fixed; each active span cannot grow.
    successor=np.full(size,-1,np.int64)
    job_successor=np.full(size,-1,np.int64)
    for j in range(n):
        for q in range(counters[j]-1): job_successor[offsets[j]+q]=offsets[j]+q+1
    for k in range(m):
        for h in range(lengths[k]-1): successor[timelines[k,h]]=timelines[k,h+1]
    for op in np.argsort(starts)[::-1]:
        if successor[op]<0: continue
        latest_end=starts[successor[op]]
        if job_successor[op]>=0: latest_end=min(latest_end,starts[job_successor[op]])
        duration=ends[op]-starts[op]
        if latest_end>ends[op]: starts[op]=latest_end-duration; ends[op]=latest_end
    processing=0.; waiting=0.; energy_by_machine=np.zeros((m,2))
    for k in range(m):
        work=0.
        for h in range(lengths[k]):
            op=timelines[k,h]; work+=ends[op]-starts[op]
        ep=proc[k]*work; ei=0.
        if lengths[k]>0:
            span=ends[timelines[k,lengths[k]-1]]-starts[timelines[k,0]]
            ei=idle[k]*max(0.,span-work)
        processing+=ep; waiting+=ei; energy_by_machine[k,0]=ep; energy_by_machine[k,1]=ei
    return np.array([np.max(ends),processing+waiting]),starts,ends,timelines,lengths,order,energy_by_machine

@dataclass
class Solution:
    os: np.ndarray
    ms: np.ndarray
    f: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    timelines: np.ndarray
    lengths: np.ndarray
    order: np.ndarray
    energy: np.ndarray
    critical: object=None

def decode(problem,os,ms):
    return Solution(os,ms,*decode_arrays(os,ms,problem.offsets,problem.times,problem.proc,problem.idle,problem.n,problem.m))

def validate(p,s):
    assert len(s.os)==p.size and np.array_equal(np.bincount(s.os,minlength=p.n),p.counts)
    assert len(s.ms)==p.size and np.all(p.times[np.arange(p.size),s.ms]>0)
    assert np.all(s.starts>=-1e-8)
    assert np.allclose(s.ends-s.starts,p.times[np.arange(p.size),s.ms])
    for j in range(p.n):
        ids=np.arange(p.offsets[j],p.offsets[j]+p.counts[j])
        assert np.all(s.starts[ids[1:]]>=s.ends[ids[:-1]]-1e-8)
    ep=ei=0.
    for k in range(p.m):
        ids=np.flatnonzero(s.ms==k); ids=ids[np.argsort(s.starts[ids])]
        if not len(ids): continue
        assert np.all(s.starts[ids[1:]]>=s.ends[ids[:-1]]-1e-8)
        work=sum(p.times[o,k] for o in ids)
        ep+=p.proc[k]*work
        ei+=p.idle[k]*(s.ends[ids[-1]]-s.starts[ids[0]]-work)
    assert np.allclose(s.f,[max(s.ends),ep+ei])
    return True
