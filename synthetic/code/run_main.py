import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import sys, json, time, platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_limits,threadpool_info
from benchmark_core import *
threadpool_limits(1)
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'data'; OUT.mkdir(exist_ok=True)
rows=[]; refs=[]; configs=[]; repeats=3
start=time.perf_counter(); index=0
for d in [2,4,8,16]:
 for s in [3,6,10]:
  for cond in [10.,1000.]:
   for kind in ['balanced','heterogeneous']:
    for seedidx in range(2):
     seed=19092026+index*7919
     gen=generate(d,s,cond,seed,kind)
     As,Bs=gen.covariances(); p=from_covariances(As,Bs,gen.meta)
     common=p.common_preprocessing_seconds
     setup=p.build_lift()
     configs.append(dict(index=index,d=d,s=s,cond=cond,kind=kind,seed=seed,
                         common_preprocessing_seconds=common,lift_setup_seconds=setup))
     np.savez_compressed(OUT/f'instance_{index:03d}.npz',A=np.array(As),B=np.array(Bs))
     for mode in ['trace','logdet']:
      ref=reference(p,mode); wstar=ref['w']; Fstar=ref['f']; fr=ref['gfw']
      f0=p.oracle(np.ones(s)/s,mode,0)
      refs.append(dict(index=index,d=d,s=s,cond=cond,kind=kind,seed=seed,mode=mode,
                       fstar=Fstar,reference_fw=fr,reference_converged=ref['converged'],
                       active=int(np.sum(wstar>1e-7)),interior=bool(np.min(wstar)>1e-7),
                       improvement_from_uniform=f0-Fstar,wstar=json.dumps(wstar.tolist())))
      for eta in [1.,.1,.01,.001,.0001]:
       c=feasible((1-eta)*wstar+eta*np.ones(s)/s)
       measures={}
       # Randomized within-candidate order. No reuse of scale/certificate outputs.
       jobs=['vertex','cheap']*repeats; np.random.default_rng(seed+int(1/eta)).shuffle(jobs)
       for typ in jobs:
        result=certificate(p,c,mode,bkind=typ,tol=1e-13)
        measures.setdefault(typ,[]).append(result)
       v=measures['vertex'][0]; cheap=measures['cheap'][0]
       f,g=p.oracle(c,mode,1)
       t=time.perf_counter(); _=fw_gap(g,c); fw_arithmetic=time.perf_counter()-t
       base=dict(index=index,d=d,s=s,cond=cond,kind=kind,seed=seed,mode=mode,eta=eta,
                 true_gap_estimate=f-Fstar,reference_fw=fr,interior=np.min(wstar)>1e-7,
                 active=int(np.sum(wstar>1e-7)),gfw=v['gfw'],gq=v['gq'],gq_cheap=cheap['gq'],
                 b=v['b'],b_cheap=cheap['b'],qp_gap=v['qp_gap'],qp_status=v['qp_status'],
                 cheap_qp_gap=cheap['qp_gap'],qp_iterations=v['qp_iterations'],
                 weights=json.dumps(c.tolist()),anchor=json.dumps(v['z'].tolist()),
                 common_preprocessing_seconds=common,lift_setup_seconds=setup,
                 fw_extra_seconds=fw_arithmetic)
       for typ, prefix in [('vertex','vertex'),('cheap','cheap')]:
        for key in ['oracle_seconds','scale_seconds','qp_seconds','total_seconds']:
         base[prefix+'_'+key]=float(np.median([r[key] for r in measures[typ]]))
       base['violation_vs_reference']=max(0.,f-Fstar-v['gq'])
       rows.append(base)
     index+=1
     if index%8==0:
      pd.DataFrame(rows).to_csv(OUT/'certificate_results.csv',index=False)
      pd.DataFrame(refs).to_csv(OUT/'references.csv',index=False)
      pd.DataFrame(configs).to_csv(OUT/'instances.csv',index=False)
      print(f'completed {index}/96 instances, {len(rows)} candidate tests, wall={time.perf_counter()-start:.2f}s',flush=True)
pd.DataFrame(rows).to_csv(OUT/'certificate_results.csv',index=False)
pd.DataFrame(refs).to_csv(OUT/'references.csv',index=False)
pd.DataFrame(configs).to_csv(OUT/'instances.csv',index=False)
env=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,
         threadpools=threadpool_info(),main_elapsed_seconds=time.perf_counter()-start,
         timing_repetitions=repeats,instances=index,optimization_problems=len(refs),candidate_tests=len(rows))
(OUT/'environment.json').write_text(json.dumps(env,indent=2))
print(json.dumps(env,indent=2),flush=True)
