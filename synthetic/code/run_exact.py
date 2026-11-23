import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,time,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'))
from sci_exact import RationalSCI,quantize_simplex
from sci_solver import NumericSCI
from benchmark_core import reference,qp,newton
import numpy as np
import sympy as sp
import pandas as pd
from threadpoolctl import threadpool_limits
threadpool_limits(1)
I=[[1,0],[0,1]];O=[[0,0],[0,0]]
cases=[
('noncommuting', [[[2,1],[1,3]],[[3,-1],[-1,2]],[[1,0],[0,4]]], [[[3,1],[1,2]],[[2,0],[0,1]],[[2,-1],[-1,3]]]),
('singular', [[[1,0],[0,0]],[[1,1],[1,1]],O], [[[1,1],[1,3]],[[2,0],[0,1]],[[3,1],[1,2]]]),
('ordinary_CI', [[[2,1],[1,3]],[[3,-1],[-1,2]],[[1,0],[0,4]]],[O,O,O]),
('flat_scalar', [[[4]],[[4]],[[1]]],[[[0]],[[0]],[[2]]]),
('flat_matrix', [[[4,0],[0,4]],[[4,0],[0,4]],I],[O,O,[[2,0],[0,2]]]),
('constant', [O,O,O],[I,[[2,0],[0,2]],[[3,1],[1,2]]]),
('four_estimators', [[[2,1],[1,3]],[[3,-1],[-1,2]],[[1,0],[0,4]],[[5,2],[2,2]]], [[[3,1],[1,2]],[[2,0],[0,1]],[[2,-1],[-1,3]],[[1,0],[0,3]]]),
]
class Adapter:
 def __init__(self,p):self.p=p;self.s=p.s;self.d=p.d
 def oracle(self,w,mode='logdet',order=2):
  f,g,H,P=self.p.oracle(w,mode)
  if order==0:return f/self.d
  if order==1:return f/self.d,g/self.d
  return f/self.d,g/self.d,H/self.d,P
finalrows=[];finalcerts=[]
for name,A,B in cases:
 st=time.perf_counter();ex=RationalSCI(A,B); setup=time.perf_counter()-st
 num=Adapter(NumericSCI(ex))
 for mode in ['trace','logdet']:
  # A complete small-scale numerical proposal -> exact verification workflow.
  target=1e-6; start=time.perf_counter()
  answer=newton(num,mode,eps=target/4,stopping='cheap_gated')
  proposal_seconds=time.perf_counter()-start
  cfinal=quantize_simplex(answer['w'],bits=40);wfinal=np.array(cfinal,dtype=float).ravel()
  ff,gg,HH,PP=num.oracle(wfinal,mode)
  start=time.perf_counter()
  if min(wfinal)>0:
   bfinal=1/min(wfinal);bp=None
  else:
   _,bfinal=num.p.scales(wfinal);bp=str(bfinal)
  lowfinal=qp(wfinal,gg,HH/bfinal,atol=1e-14)
  zfinal=quantize_simplex(lowfinal.z,bits=40)
  floating_posterior_seconds=time.perf_counter()-start
  start=time.perf_counter();vc=ex.certificate(cfinal,zfinal,mode,b_proposal=bp)
  checking_seconds=time.perf_counter()-start
  finalgap=sp.Rational(vc['gap']['numerator'],vc['gap']['denominator'])/ex.d
  finalrows.append(dict(case=name,d=ex.d,s=ex.s,mode=mode,target=target,
                        numerical_converged=answer['converged'],exact_target_met=bool(finalgap<=sp.Rational(1,1000000)),
                        exact_gap=float(finalgap),candidate_seconds=proposal_seconds,
                        posterior_float_seconds=floating_posterior_seconds,
                        posterior_exact_seconds=checking_seconds,exact_input_setup_seconds=setup,
                        end_to_end_seconds=setup+proposal_seconds+floating_posterior_seconds+checking_seconds))
  finalcerts.append(dict(case=name,dimension_normalization=ex.d,certificate=vc))

pd.DataFrame(finalrows).to_csv(ROOT/"data"/"exact_final_workflows.csv",index=False)
(ROOT/"data"/"exact_final_certificates.json").write_text(json.dumps(finalcerts,indent=2))
