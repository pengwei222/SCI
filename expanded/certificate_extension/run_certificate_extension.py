"""Seven-point same-candidate certificate diagnostics on the frozen D2/D3 inputs.

Actual certificate evaluation, never interpolation. The archived reference
candidate is reconstructed by the original SLSQP/FW rule, and anchor-point
agreement is checked. Failed certificates retain FW. Float64 diagnostics only.
Input I/O is outside any computation; this script reports no performance claim.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits,threadpool_info
import original_core as old
threadpool_limits(1)
ROOT=Path(__file__).resolve().parent
ETAS=[.1,.03,.01,.003,.001,.0003,.0001]
ANCHORS=[.1,.01,.001]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read_case(path):
 with np.load(path,allow_pickle=False) as z:
  m=json.loads(str(z['metadata']))
  if m.get('panel')!='recorded' or m.get('dataset') not in [2,3]:return None
  return dict(m,**{k:z[k].copy() for k in ['A','B','ref_w']})

def evaluate(case,archived,refs):
 ident={k:int(case[k]) for k in ['dataset','seed','epoch']};idx=tuple(ident.values())
 model=old.SCI(case['A'],case['B'],'trace',1.,pose_dimension=15)
 ref=old.solve(model,tol=1e-10,rule='fw',maxiter=200,w0=case['ref_w'])
 past=refs.loc[idx];boundary=bool(ref['w'].min()<1e-5)
 assert boundary==bool(past.reference_boundary),(idx,'classification changed')
 assert abs(ref['f']-past.reference_f)<1e-11,(idx,'reference objective changed')
 reference=dict(**ident,reference_f=ref['f'],reference_gap=ref['gap'],reference_boundary=boundary,reference_weights=json.dumps(ref['w'].tolist()),reconstructed_reference_f_delta=abs(ref['f']-past.reference_f),reconstructed_reference_gap_delta=abs(ref['gap']-past.reference_gap))
 out=[];anchors=[];u=np.ones(5)/5
 for eta in ETAS:
  w=old.simplex((1-eta)*ref['w']+eta*u);f,g,H,P=model.oracle(w,True);fw=max(0.,float(g@w-g.min()))
  lo=max(0.,f-ref['f']);hi=max(0.,f-ref['f']+ref['gap'])
  for rule in ['cheap','vertex']:
   error='';b=inner=gq=np.nan
   try:
    b=1/float(w.min()) if rule=='cheap' else model.vertex_b(w)
    _,dual,inner=old.qp(w,g,H/b);gq=max(0.,-dual);effective=min(fw,gq)
   except (ArithmeticError,np.linalg.LinAlgError,ValueError) as exc:
    error=f'{type(exc).__name__}: {exc}';effective=fw
   ratio=1. if fw==0 and effective==0 else fw/max(effective,1e-16)
   row=dict(**ident,time=float(case['time']),eta=eta,rule=rule,reference_boundary=boundary,f=f,fw_gap=fw,gq_gap=gq,effective_gap=effective,tightening_ratio=ratio,objective_gap_lower=lo,objective_gap_upper=hi,reference_gap=ref['gap'],b=b,inner_gap=inner,certificate_failed=bool(error),error=error,bound_violation=bool(effective+2e-9<lo),reference_resolution_sufficient=bool(hi-lo<=max(lo,1e-16)/20))
   out.append(row)
   if eta in ANCHORS:
    oldrow=archived.loc[idx+(eta,rule)]
    rec=dict(**ident,eta=eta,rule=rule,classification_equal=boundary==bool(oldrow.reference_boundary),failure_equal=bool(error)==bool(oldrow.certificate_failed))
    for key in ['f','fw_gap','gq_gap','effective_gap','tightening_ratio','b','inner_gap']:
     a,bv=float(row[key]),float(oldrow[key]);rec[key+'_abs_delta']=abs(a-bv) if np.isfinite(a) and np.isfinite(bv) else 0. if np.isnan(a) and np.isnan(bv) else np.inf
    # Anchors require directly recomputed objective/gaps/scales, with absolute
    # floors for tiny near-optimum quantities. Ratios are reported separately.
    rec['numerical_match']=bool(rec['classification_equal'] and rec['failure_equal'] and all(np.isclose(float(row[key]),float(oldrow[key]),rtol=2e-7,atol=2e-12,equal_nan=True) for key in ['f','fw_gap','gq_gap','effective_gap','b','inner_gap']))
    anchors.append(rec)
 return out,reference,anchors

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,default=ROOT.parent/'inputs');p.add_argument('--output',type=Path,default=ROOT/'results');p.add_argument('--limit',type=int,default=0);args=p.parse_args()
 args.output.mkdir(parents=True,exist_ok=True)
 if (args.output/'same_point_tightness.csv').exists():raise FileExistsError('Choose fresh output directory')
 archived=pd.read_csv(ROOT/'archived_three_point.csv').set_index(['dataset','seed','epoch','eta','rule'])
 refs=pd.read_csv(ROOT/'archived_references.csv').set_index(['dataset','seed','epoch'])
 paths=sorted(args.inputs.glob('recorded_d0[23]_*.npz'))
 if args.limit:paths=paths[:args.limit]
 else:assert len(paths)==1246,len(paths)
 manifest=dict(etas=ETAS,anchor_etas=ANCHORS,inputs=len(paths),selection='Exactly original D2/D3 inputs; no outcome filtering',reference='Original SLSQP/FW, tol1e-10,maxiter200, saved ref_w start',boundary='min(reference_w)<1e-5',certificates='Original qp and vertex_b; failed certificate retains FW; 0/0 gain defined as1',normalization='pose trace divided by uniform pose trace',timing_claim=False,numerical_only=True,source_sha256={name:sha(ROOT/name) for name in ['original_core.py','run_certificate_extension.py','archived_three_point.csv','archived_references.csv']},input_sha256={path.name:sha(path) for path in paths},threads=threadpool_info())
 (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2));rows=[];reference_rows=[];anchor_rows=[];t=time.perf_counter()
 for i,path in enumerate(paths):
  case=read_case(path);a,b,c=evaluate(case,archived,refs);rows+=a;reference_rows.append(b);anchor_rows+=c
  if (i+1)%50==0 or i==len(paths)-1:
   pd.DataFrame(rows).to_csv(args.output/'same_point_tightness.csv',index=False);pd.DataFrame(reference_rows).to_csv(args.output/'reference_weights.csv',index=False);pd.DataFrame(anchor_rows).to_csv(args.output/'anchor_comparison.csv',index=False)
   print(f'{i+1}/{len(paths)} inputs; {len(rows)} certificate evaluations; elapsed {time.perf_counter()-t:.1f}s',flush=True)
 df=pd.DataFrame(rows);anc=pd.DataFrame(anchor_rows)
 df.groupby(['dataset','eta','rule','reference_boundary']).agg(cases=('epoch','size'),median_tightening=('tightening_ratio','median'),p10_tightening=('tightening_ratio',lambda x:x.quantile(.1)),p90_tightening=('tightening_ratio',lambda x:x.quantile(.9)),median_fw_gap=('fw_gap','median'),median_effective_gap=('effective_gap','median'),median_objective_gap_lower=('objective_gap_lower','median'),median_objective_gap_upper=('objective_gap_upper','median'),certificate_failures=('certificate_failed','sum'),observed_violations=('bound_violation','sum')).reset_index().to_csv(args.output/'tightness_summary.csv',index=False)
 df[df.certificate_failed].to_csv(args.output/'certificate_failures.csv',index=False)
 quality=dict(inputs=len(paths),candidates=len(paths)*len(ETAS),evaluations=len(df),anchors=len(anc),anchor_matches=int(anc.numerical_match.sum()),anchor_failures=int((~anc.numerical_match).sum()),certificate_failures=int(df.certificate_failed.sum()),bound_violations=int(df.bound_violation.sum()),max_reference_gap=float(df.reference_gap.max()),max_reconstructed_reference_f_delta=max(x['reconstructed_reference_f_delta'] for x in reference_rows),elapsed_seconds=time.perf_counter()-t,source_unchanged=all(sha(ROOT/name)==h for name,h in manifest['source_sha256'].items()),note='Float64 engineering checks; old reference resolution is not assumed exact near optimum.')
 quality['anchor_max_deltas']={c:float(anc[c].max()) for c in anc.columns if c.endswith('_abs_delta')}
 (args.output/'quality.json').write_text(json.dumps(quality,indent=2));print(json.dumps(quality),flush=True)
if __name__=='__main__':main()
