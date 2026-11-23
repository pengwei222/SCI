"""Paired SCI benchmark with frozen algorithms, direct branch audits and full costs.

Float64 comparisons are numerical validation, not interval certificates.
Timing repetitions and initialization perturbations are not independent samples.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,csv,hashlib,json,platform,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_limits,threadpool_info
import original_core as old
import optimized_core as slsqp
from fast_core import FastSCI
import newton_solver
from audit_reference import reference,evaluate
threadpool_limits(1)
ROOT=Path(__file__).resolve().parent
METHODS={'original-fw':(old.SCI,old.solve,{'rule':'fw'}),
 'blas-fw':(FastSCI,slsqp.solve,{'rule':'fw'}),
 'newton-fw':(FastSCI,newton_solver.solve,{'rule':'fw'}),
 'newton-cheap':(FastSCI,newton_solver.solve,{'rule':'cheap','gate_factor':4.})}

def jsonable(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return jsonable(v.item())
 if isinstance(v,float) and not np.isfinite(v):return None
 if isinstance(v,dict):return {k:jsonable(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [jsonable(x) for x in v]
 return v

def monitored(cls):
 class M(cls):
  def __init__(self,*a,**kw):
   self.phase='construction';self.profile={p:{'seconds':0.,'calls':0,'hessian_requests':0} for p in ['construction','solver','fusion']}
   super().__init__(*a,**kw)
  def oracle(self,w,hessian=True):
   t=time.perf_counter()
   try:return super().oracle(w,hessian)
   finally:
    x=self.profile[self.phase];x['seconds']+=time.perf_counter()-t;x['calls']+=1;x['hessian_requests']+=int(hessian)
 return M
MONITORED={n:(monitored(c),s,k) for n,(c,s,k) in METHODS.items()}

def time_solve(c,name,tol):
 cls,solver,kw=MONITORED[name]
 t0=time.perf_counter();model=cls(c['A'],c['B'],c['mode'],c.get('heading_length',1.),pose_dimension=c.get('pose_dimension'));t1=time.perf_counter()
 model.phase='solver';r=solver(model,tol=tol,maxiter=150,history=True,**kw);t2=time.perf_counter()
 model.phase='fusion';x,P=model.fuse(r['w'],c['means']);t3=time.perf_counter()
 hist=r.get('history',[])
 ct=int(sum(np.isfinite(v.get('gq',np.nan)) for v in hist))+int(r['certificate_failures'])
 timing={'construction_seconds':t1-t0,'solver_seconds':t2-t1,'fusion_seconds':t3-t2,'end_to_end_seconds':t3-t0,
  'oracle_construction_seconds':model.profile['construction']['seconds'],'oracle_solver_seconds':model.profile['solver']['seconds'],
  'oracle_fusion_seconds':model.profile['fusion']['seconds'],'hessian_requests':model.profile['solver']['hessian_requests'],
  'oracle_requests':model.profile['solver']['calls'],'certificate_attempts':ct}
 return r,x,P,timing

def persist_cases(cases):
 target=ROOT/'inputs';target.mkdir(exist_ok=True)
 for c in cases:
  p=target/(c['case_id']+'.npz')
  arr={k:v for k,v in c.items() if isinstance(v,np.ndarray)}
  meta={k:v for k,v in c.items() if not isinstance(v,np.ndarray)}
  if not p.exists():np.savez_compressed(p,**arr,metadata=json.dumps(jsonable(meta)))
  else:
   with np.load(p,allow_pickle=False) as saved:
    if any(k not in saved or not np.array_equal(v,saved[k]) for k,v in arr.items()) or json.loads(str(saved['metadata']))!=jsonable(meta):
     raise RuntimeError('Persisted input differs from current case: '+c['case_id'])
 return {c['case_id']:hashlib.sha256((target/(c['case_id']+'.npz')).read_bytes()).hexdigest() for c in cases}

def load_recorded(path):
 cases=[]
 for p in sorted(Path(path).rglob('bank_*.npz')):
  meta=json.loads((p.parent/'manifest.json').read_text())
  with np.load(p,allow_pickle=False) as z:
   for j,epoch in enumerate(z['epoch']):
    if int(epoch)%5 or z['time'][j]<30:continue
    ds,seed=int(meta['dataset']),int(meta['seed'])
    c={k:z[k][j].copy() for k in ['A','B','means']}
    if 'ref_w' in z:c['ref_w']=z['ref_w'][j].copy()
    c.update(case_id=f'recorded_d{ds:02d}_s{seed}_e{int(epoch):05d}',panel='recorded',regime='recorded',d=c['A'].shape[1],s=len(c['A']),seed=seed,dataset=ds,epoch=int(epoch),mode='trace',pose_dimension=15,time=float(z['time'][j]))
    cases.append(c)
 return cases

def load_saved(path):
 cases=[]
 for p in sorted(Path(path).glob('*.npz')):
  with np.load(p,allow_pickle=False) as z:
   c=json.loads(str(z['metadata']))
   c.update({k:z[k].copy() for k in z.files if k!='metadata'})
   cases.append(c)
 return cases

def summarize(out):
 df=pd.read_csv(out/'timing.csv')
 metrics=['end_to_end_seconds','construction_seconds','solver_seconds','fusion_seconds','oracle_solver_seconds','certificate_seconds','checks','oracle_calls','certificate_attempts','hessian_requests','audit_error_upper']
 med=df.groupby(['case_id','method','tolerance'],as_index=False)[metrics].median()
 attrs=df[['case_id','panel','regime','d','s','seed','mode','dataset','epoch']].drop_duplicates('case_id')
 med=med.merge(attrs,on='case_id',validate='many_to_one')
 ok=df.groupby(['case_id','method','tolerance'],as_index=False).agg(success=('success','all'),reference_resolved=('reference_resolved','all'),timing_complete=('timed_solve_completed','all'),repeats=('repeat','size'))
 med=med.merge(ok,on=['case_id','method','tolerance'])
 med.to_csv(out/'per_case.csv',index=False)
 summary=[]
 for key,a in med.groupby(['panel','regime','d','s','mode','dataset','method','tolerance'],dropna=False):
  rec=dict(zip(['panel','regime','d','s','mode','dataset','method','tolerance'],key));rec.update(cases=len(a),successes=int(a.success.sum()),complete_timings=int(a.timing_complete.sum()),reference_unresolved=int((~a.reference_resolved).sum()))
  # Timing summaries retain all attempted cases; success count is always paired.
  for metric in metrics:
   rec['median_'+metric]=a[metric].median()
  for q in [.1,.9,.95,.99]:rec[f'p{round(q*100)}_seconds']=a.end_to_end_seconds.quantile(q)
  summary.append(rec)
 pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
 keys=['case_id','tolerance'];tables={m:a.set_index(keys) for m,a in med.groupby('method')};pairs=[]
 for new,baseline,label in [('blas-fw','original-fw','matrix'),('newton-fw','blas-fw','newton'),('newton-cheap','newton-fw','certificate'),('newton-cheap','original-fw','complete')]:
  a,b=tables[new],tables[baseline];idx=a.index.intersection(b.index)
  p=a.loc[idx].reset_index();p['paired_speedup']=b.loc[idx].end_to_end_seconds.to_numpy()/a.loc[idx].end_to_end_seconds.to_numpy();p['paired_success']=(a.loc[idx].success.to_numpy()&b.loc[idx].success.to_numpy());p['checks_saved']=b.loc[idx].checks.to_numpy()-a.loc[idx].checks.to_numpy();p['contrast']=label
  pairs.append(p)
 pairs=pd.concat(pairs,ignore_index=True);pairs.to_csv(out/'paired.csv',index=False)
 pairsummary=[]
 for key,a in pairs.groupby(['panel','regime','d','s','mode','dataset','tolerance','contrast'],dropna=False):
  rec=dict(zip(['panel','regime','d','s','mode','dataset','tolerance','contrast'],key));valid=a[a.paired_success]
  rec.update(cases=len(a),paired_successes=len(valid),median_speedup=valid.paired_speedup.median(),p10_speedup=valid.paired_speedup.quantile(.1),p90_speedup=valid.paired_speedup.quantile(.9),fraction_faster=(valid.paired_speedup>1).mean(),fraction_checks_saved=(valid.checks_saved>0).mean(),mean_checks_saved=valid.checks_saved.mean())
  pairsummary.append(rec)
 pd.DataFrame(pairsummary).to_csv(out/'paired_summary.csv',index=False)
 quality=dict(rows=len(df),unique_inputs=df.case_id.nunique(),successes=int(df.success.sum()),failed=int((~df.success).sum()),solver_nonconverged=int((~df.converged).sum()),reference_unresolved=int((~df.reference_resolved).sum()),accuracy_upper_unconfirmed=int((~df.accuracy_upper_confirmed).sum()),bound_violations=int(df.bound_violation.sum()),max_reference_gap=float(df.reference_gap.max()),max_oracle_f_error=float(df.oracle_f_error.max()),max_covariance_relative_error=float(df.covariance_relative_error.max()),max_fused_relative_error=float(df.fused_relative_error.max()),numerical_not_interval=True)
 (out/'quality.json').write_text(json.dumps(quality,indent=2));print(json.dumps(quality),flush=True)

def run(cases,args):
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 if (out/'timing.csv').exists():raise RuntimeError('Use a fresh output directory; existing results are never overwritten')
 if len({c['case_id'] for c in cases})!=len(cases):raise ValueError('Duplicate input case_id')
 if args.limit:cases=cases[:args.limit]
 cases=list(cases);rng=np.random.default_rng(20261116);rng.shuffle(cases)
 sources={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['original_core.py','optimized_core.py','fast_core.py','newton_solver.py','synthetic_inputs.py','audit_reference.py','run_expanded.py']}
 hashes=persist_cases(cases)
 protocol={'inputs':len(cases),'methods':list(METHODS),'tolerances':args.tolerances,'repeats':args.repeats,'seed':20261116,'maxiter':150,'gate':4.,'initialization':'uniform; no reference warm starts online','precision':'float64','reference_margin':'direct FW gap <= min_tol/20','timed':'fresh constructor+solve+fusion; identical profiling wrapper in each implementation','oracle_times':'inclusive of oracle calls inside certificate; do not add to certificate time','success':'solver bound met AND direct-reference upper error <=tol+numerical_allowance AND reference resolved AND direct outputs agree','numerical_allowance':'max(1e-10,1e-7*tol)','source_hashes':sources,'input_sha256':hashes,'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform(),'threads':threadpool_info()}
 (out/'manifest.json').write_text(json.dumps(protocol,indent=2))
 for name in METHODS:
  try:time_solve(cases[0],name,max(args.tolerances))
  except Exception:pass  # Warm-up errors do not remove the formal attempt.
 nrows=0;beg=time.perf_counter();refs=[]
 fields=['case_id','panel','regime','d','s','seed','mode','dataset','epoch','method','tolerance','repeat','construction_seconds','solver_seconds','fusion_seconds','end_to_end_seconds','oracle_construction_seconds','oracle_solver_seconds','oracle_fusion_seconds','hessian_requests','oracle_requests','certificate_attempts','checks','oracle_calls','certificate_seconds','certificate_failures','fallback_used','reported_gap','fw_gap','converged','objective','direct_objective','reference_f','reference_gap','reference_resolved','audit_error_lower','audit_error_upper','accuracy_upper_confirmed','bound_violation','oracle_f_error','simplex_error','covariance_relative_error','fused_relative_error','success','timed_solve_completed','weights','status']
 with (out/'timing.csv').open('w',newline='') as file:
  writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
  for i,c in enumerate(cases):
   try:ref=reference(c,min_tol=min(args.tolerances),target_gap=min(1e-9,min(args.tolerances)/100))
   except Exception as exc:ref=dict(ref_f=float('nan'),lower=-float('inf'),gap=float('inf'),verified=False,scale=None,status='reference_unresolved',errors=[type(exc).__name__+': '+str(exc)])
   resolved=bool(ref['verified'] and ref['gap']<=min(args.tolerances)/20)
   refs.append(dict(case_id=c['case_id'],**jsonable(ref)))
   jobs=[(m,t,r) for m in METHODS for t in args.tolerances for r in range(args.repeats)];rng.shuffle(jobs)
   for name,tol,repeat in jobs:
    row={k:c.get(k,-1) for k in ['case_id','panel','regime','d','s','seed','mode','dataset','epoch']};row.update(method=name,tolerance=tol,repeat=repeat,reference_f=ref['ref_f'],reference_gap=ref['gap'],reference_resolved=resolved)
    attempt_start=time.perf_counter()
    row['timed_solve_completed']=False
    try:
     r,x,P,times=time_solve(c,name,tol)
     row.update(**times,timed_solve_completed=True)
     audit=evaluate(c,r['w'],scale=ref['scale']);af=audit['f'];ap=audit['P'];ax=ap@sum(ji@mu for ji,mu in zip(audit['Ji'],c['means']))
     lo=max(0.,af-ref['ref_f']);hi=max(0.,af-ref['lower']);slack=max(1e-10,1e-7*tol)
     ferr=abs(af-r['f']);perr=np.linalg.norm(P-ap)/max(np.linalg.norm(ap),1e-30);xerr=np.linalg.norm(x-ax)/max(np.linalg.norm(ax),1e-30);simp=max(abs(sum(r['w'])-1),max(0.,-min(r['w'])))
     good=bool(r['converged'] and resolved and hi<=tol+slack and ferr<=max(slack,1e-8) and perr<=1e-6 and xerr<=1e-6 and simp<=1e-10 and r['gap']+slack>=lo)
     row.update(**times,checks=r['checks'],oracle_calls=r['oracle_calls'],certificate_seconds=r['certificate_seconds'],certificate_failures=r['certificate_failures'],fallback_used=r.get('fallback_used',False),reported_gap=r['gap'],fw_gap=r['fw'],converged=bool(r['converged']),objective=r['f'],direct_objective=af,audit_error_lower=lo,audit_error_upper=hi,accuracy_upper_confirmed=bool(hi<=tol+slack),bound_violation=bool(r['gap']+slack<lo),oracle_f_error=ferr,simplex_error=simp,covariance_relative_error=perr,fused_relative_error=xerr,success=good,weights=json.dumps(r['w'].tolist()),status=r['status'])
    except Exception as exc:
     row.setdefault('end_to_end_seconds',time.perf_counter()-attempt_start)
     row.update(converged=False,success=False,accuracy_upper_confirmed=False,bound_violation=False,status=type(exc).__name__+': '+str(exc),weights='[]')
    writer.writerow(row);nrows+=1
   file.flush()
   if (i+1)%10==0 or i==len(cases)-1:
    (out/'references.json').write_text(json.dumps(refs,indent=2));print(f'{i+1}/{len(cases)} inputs; {nrows} solves; {time.perf_counter()-beg:.1f}s',flush=True)
 (out/'completion.json').write_text(json.dumps({'inputs':len(cases),'rows':nrows,'elapsed_seconds':time.perf_counter()-beg,'sources_unchanged':all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in sources.items())},indent=2))
 summarize(out)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--suite',choices=['synthetic','recorded','saved'],required=True);p.add_argument('--banks',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=3);p.add_argument('--tolerances',nargs='+',type=float,default=[1e-2,1e-3,1e-5]);p.add_argument('--limit',type=int,default=0);args=p.parse_args()
 if args.suite=='synthetic':
  from synthetic_inputs import make_cases
  cases=make_cases()
 elif args.suite=='recorded':cases=load_recorded(args.banks)
 else:cases=load_saved(args.banks)
 run(cases,args)
