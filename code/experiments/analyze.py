import bootstrap
from pathlib import Path
import json,csv,collections
import numpy as np
from scipy.stats import rankdata,friedmanchisquare
from metrics import nondominated,hv,igd_plus,paired,holm
from eams_abc.problem import read_problem
from eams_abc.decoder import decode,validate
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'code'/'output'
MAIN=['EAMS-ABC','DABC','NSGA-II','MOEA-D','MA-NSGA-II']
def save_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    runs=[]
    for path in (OUT/'raw').rglob('*.json'):
        d=json.loads(path.read_text()); d['_path']=str(path.relative_to(ROOT)); runs.append(d)
    groups=collections.defaultdict(list)
    for d in runs: groups[(d['instance'],d['scenario'],d['idle_ratio'])].append(d)
    refs={}; rows=[]; selected={}
    for key,ds in groups.items():
        # Freeze a common box from all reported final and initial archives in this scenario.
        allF=np.vstack([np.array(d['front']) for d in ds]+[np.array(d['history'][0]['front']) for d in ds])
        lo=allF.min(0); hi=allF.max(0); span=np.maximum(hi-lo,1e-12)
        pooled=nondominated(np.vstack([d['front'] for d in ds])); R=(pooled-lo)/span
        refs['|'.join(map(str,key))]=dict(lower=lo.tolist(),upper=hi.tolist(),reference=[1.1,1.1],empirical_front=pooled.tolist(),runs=len(ds))
        for d in ds:
            F=np.array(d['front']); Z=(F-lo)/span; ix=int(np.argmin(np.linalg.norm(Z,axis=1)))
            rows.append(dict(instance=d['instance'],scenario=d['scenario'],idle_ratio=d['idle_ratio'],stage=d['stage'],method=d['method'],seed=d['seed'],hv=hv(Z),igd_plus=igd_plus(Z,R),cmin=float(F[:,0].min()),emin=float(F[:,1].min()),compromise_c=float(F[ix,0]),compromise_e=float(F[ix,1]),archive_size=len(F),seconds=d['seconds'],evaluations=d['evaluations'],path=d['_path']))
            d['_hv']=rows[-1]['hv']; d['_compromise']=ix
        if key[1:] == ('independent',.15):
            for method in MAIN:
                choices=sorted([d for d in ds if d['method']==method and d['stage']=='main'],key=lambda d:d['_hv'])
                if choices:
                    d=choices[(len(choices)-1)//2]; selected[f'{key[0]}|{method}']=dict(path=d['_path'],seed=d['seed'],solution=d['solutions'][d['_compromise']],hv=d['_hv'])
    summary=[]; buckets=collections.defaultdict(list)
    for row in rows: buckets[(row['stage'],row['instance'],row['method'],row['scenario'],row['idle_ratio'])].append(row)
    for key,rs in buckets.items():
        row=dict(zip(['stage','instance','method','scenario','idle_ratio'],key)); row['runs']=len(rs)
        for metric in ['hv','igd_plus','cmin','emin','compromise_c','compromise_e','archive_size','seconds']:
            values=np.array([r[metric] for r in rs]); row[metric+'_mean']=float(values.mean());row[metric+'_std']=float(values.std(ddof=1)) if len(values)>1 else 0.;row[metric+'_median']=float(np.median(values))
        summary.append(row)
    stats={}
    for metric,reverse in [('hv',True),('igd_plus',False)]:
        datasets=sorted({r['instance'] for r in rows if r['stage']=='main'})
        if not datasets:continue
        data=np.array([[next(r[metric+'_median'] for r in summary if r['stage']=='main' and r['instance']==name and r['method']==method) for method in MAIN] for name in datasets])
        pairs=[]
        rng=np.random.default_rng(731)
        for j in range(1,len(MAIN)):
            p,effect=paired(data[:,0],data[:,j]); delta=data[:,0]-data[:,j]
            boots=np.mean(delta[rng.integers(len(delta),size=(10000,len(delta)))],axis=1)
            pairs.append(dict(method=MAIN[j],p=p,rank_biserial=effect,mean_paired_delta=float(delta.mean()),ci95=np.quantile(boots,[.025,.975]).tolist(),wins=int(np.sum(delta>1e-10) if reverse else np.sum(delta< -1e-10)),ties=int(np.sum(abs(delta)<=1e-10))))
        for pair,adj in zip(pairs,holm([x['p'] for x in pairs])):pair['p_holm']=float(adj)
        test=friedmanchisquare(*data.T)
        stats[metric]=dict(blocks=datasets,friedman_statistic=float(test.statistic),friedman_p=float(test.pvalue),average_ranks=dict(zip(MAIN,np.mean([rankdata(-v if reverse else v) for v in data],axis=0).tolist())),pairs=pairs)
    abstats=[]
    for method in ['no-critical','no-adaptive','no-rebuild','no-gap','no-exchange']:
        aa=[];bb=[]
        for name in sorted({r['instance'] for r in summary if r['stage']=='ablation'}):
            a=next((r for r in summary if r['stage']=='main' and r['method']=='EAMS-ABC' and r['instance']==name),None)
            b=next((r for r in summary if r['stage']=='ablation' and r['method']==method and r['instance']==name),None)
            if a and b: aa.append(a['hv_median']);bb.append(b['hv_median'])
        if aa:
            p,effect=paired(aa,bb);abstats.append(dict(method=method,p=p,rank_biserial=effect,mean_delta=float(np.mean(np.array(aa)-bb)),wins=int(np.sum(np.array(aa)>np.array(bb))),blocks=len(aa)))
    for d,adj in zip(abstats,holm([x['p'] for x in abstats])):d['p_holm']=float(adj)
    report=dict(total_runs=len(runs),total_evaluations=sum(r['evaluations'] for r in runs),stage_counts=dict(collections.Counter(d['stage'] for d in runs)),main_statistics=stats,ablation_statistics=abstats)
    save_csv(OUT/'run_metrics.csv',rows);save_csv(OUT/'summary.csv',summary)
    (OUT/'normalization.json').write_text(json.dumps(refs,indent=2),encoding='utf-8')
    (OUT/'selected_schedules.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    (OUT/'statistical_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
