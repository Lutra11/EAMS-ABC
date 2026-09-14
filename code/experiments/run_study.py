"""Deterministic, resumable experiments. Each JSON is one independent run."""
import bootstrap
from pathlib import Path
import argparse,json,os,platform,hashlib,time
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor,as_completed
from eams_abc.problem import read_problem
from eams_abc.solver import Config,run

ROOT=Path(__file__).resolve().parents[2]
MAIN=['EAMS-ABC','DABC','NSGA-II','MOEA-D','MA-NSGA-II']
ABLATION=['no-critical','no-adaptive','no-rebuild','no-gap','no-exchange']

def task(args):
    name,method,seed,scenario,ratio,cfg,stage=args
    folder=ROOT/'code'/'output'/'raw'/stage; folder.mkdir(parents=True,exist_ok=True)
    out=folder/f'{name}_{scenario}_{ratio}_{method}_{seed}.json'
    if out.exists(): return 'cached'
    p=read_problem(ROOT/'dataset'/'raw'/'brandimarte'/f'{name}.txt',scenario,ratio)
    cfg=replace(Config(),**cfg)
    if method=='DABC': cfg=replace(cfg,guided=False,adaptive=False,rebuild=False,gap=False,elite_exchange=False)
    if method=='no-critical': cfg=replace(cfg,uniform_selection=True)
    if method=='no-adaptive': cfg=replace(cfg,adaptive=False)
    if method=='no-rebuild': cfg=replace(cfg,rebuild=False)
    if method=='no-gap': cfg=replace(cfg,gap=False)
    if method=='no-exchange': cfg=replace(cfg,elite_exchange=False)
    result=run(p,method,seed,cfg); result.update(scenario=scenario,idle_ratio=ratio,stage=stage)
    temp=out.with_suffix('.tmp'); temp.write_text(json.dumps(result,separators=(',',':')),encoding='utf-8'); temp.replace(out)
    return f'{name} {method} {seed} {result["seconds"]:.1f}s'

def make_tasks(stage,budget,runs):
    if stage in ['main','ablation']:
        methods=MAIN if stage=='main' else ABLATION
        return [(f'mk{i:02d}',m,1000+s,'independent',.15,{'evaluations':budget},stage) for i in range(1,11) for s in range(runs) for m in methods]
    if stage=='power':
        return [(f'mk{i:02d}',m,2000+s,'speed',.15,{'evaluations':budget},stage) for i in range(1,11) for s in range(runs) for m in MAIN]
    if stage=='sensitivity':
        settings=[('alpha',v) for v in [.3,.5,.7]]+[('rho',v) for v in [.1,.2,.4]]+[('limit',v) for v in [15,30,60]]+[('reconstruction',v) for v in [.1,.2,.3]]
        return [(f'mk{i:02d}','EAMS-ABC',3000+s,'independent',.15,{'evaluations':budget,key:val},f'sensitivity/{key}_{val}') for i in [1,6,10] for s in range(runs) for key,val in settings]
    if stage=='idle':
        return [(f'mk{i:02d}',m,4000+s,'independent',r,{'evaluations':budget},stage) for i in [1,6,10] for s in range(runs) for r in [.05,.3] for m in MAIN]
    if stage=='development':
        return [(f'mk{i:02d}','EAMS-ABC',s,'independent',.15,{'evaluations':budget},stage) for i in [1,6,10] for s in range(runs)]
    if stage=='heldout':
        return [(f'mk{i:02d}',m,5000+s,'independent',.15,{'evaluations':budget},stage) for i in range(11,16) for s in range(runs) for m in MAIN]
    if stage=='timing':
        return [(f'mk{i:02d}',m,6000+s,'independent',.15,{'evaluations':budget},stage) for i in [1,6,10] for s in range(runs) for m in MAIN]
    raise ValueError(stage)

if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--stage',default='main'); a.add_argument('--evaluations',type=int,default=6000); a.add_argument('--runs',type=int,default=30); a.add_argument('--workers',type=int,default=6); a.add_argument('--smoke',action='store_true'); args=a.parse_args()
    tasks=make_tasks(args.stage,args.evaluations,args.runs)
    if args.smoke: tasks=[t for t in tasks if t[0]=='mk01'][:5]
    log=ROOT/'code'/'output'/f'run_{args.stage}.log'; t0=time.time()
    with log.open('a',encoding='utf-8') as f, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(task,t) for t in tasks]
        for count,future in enumerate(as_completed(futures),1):
            line=f'{count}/{len(tasks)} {future.result()} elapsed={time.time()-t0:.1f}'
            f.write(line+'\n'); f.flush()
            if count%20==0 or count==len(tasks): print(line,flush=True)
