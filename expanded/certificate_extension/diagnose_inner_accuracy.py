"""Ancillary fixed-candidate QP-accuracy diagnostic; does not modify primary data."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import argparse,hashlib,json
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
import original_core as old
import diagnostic_active_core as active
threadpool_limits(1)
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,default=ROOT.parent/'inputs');p.add_argument('--results',type=Path,default=ROOT/'results');args=p.parse_args()
 d=pd.read_csv(args.results/'same_point_tightness.csv');refs=pd.read_csv(args.results/'reference_weights.csv')
 chosen=d[(~d.reference_boundary)&d.eta.isin([.001,.0001])]
 lookup=chosen.set_index(['dataset','seed','epoch','eta','rule']);rows=[]
 for ref in refs[~refs.reference_boundary].itertuples():
  key=(int(ref.dataset),int(ref.seed),int(ref.epoch));name=f'recorded_d{key[0]:02d}_s{key[1]}_e{key[2]:05d}.npz'
  with np.load(args.inputs/name,allow_pickle=False) as z:model=old.SCI(z['A'],z['B'],'trace',1.,pose_dimension=15)
  wr=np.array(json.loads(ref.reference_weights));u=np.ones(5)/5
  for eta in [.001,.0001]:
   w=old.simplex((1-eta)*wr+eta*u);f,g,H,P=model.oracle(w,True)
   for rule in ['cheap','vertex']:
    archived=lookup.loc[key+(eta,rule)];assert abs(f-archived.f)<1e-12
    failed=bool(archived.certificate_failed)
    if failed:gq=inner=np.nan;effective=archived.fw_gap
    else:
     _,dual,inner=active.active_qp(w,g,H/archived.b);gq=max(0.,-dual);effective=min(archived.fw_gap,gq)
    gain=1. if archived.fw_gap==0 and effective==0 else archived.fw_gap/max(effective,1e-16)
    rows.append(dict(dataset=key[0],seed=key[1],epoch=key[2],eta=eta,rule=rule,original_gain=archived.tightening_ratio,active_gain=gain,original_gq=archived.gq_gap,active_gq=gq,original_inner_gap=archived.inner_gap,active_inner_gap=inner,original_inner_fraction=archived.inner_gap/archived.gq_gap if archived.gq_gap>0 else np.nan,active_inner_fraction=inner/gq if gq>0 else np.nan,certificate_failed=failed,bound_violation=bool(effective+2e-9<archived.objective_gap_lower)))
 out=pd.DataFrame(rows);out.to_csv(args.results/'auxiliary_accuracy_diagnostic.csv',index=False)
 summary=out.groupby(['dataset','eta','rule']).agg(cases=('epoch','size'),median_original_gain=('original_gain','median'),median_active_gain=('active_gain','median'),median_original_inner_fraction=('original_inner_fraction','median'),median_active_inner_fraction=('active_inner_fraction','median'),median_original_inner_gap=('original_inner_gap','median'),median_active_inner_gap=('active_inner_gap','median'),spectral_failures=('certificate_failed','sum'),observed_bound_violations=('bound_violation','sum')).reset_index()
 summary.to_csv(args.results/'auxiliary_accuracy_summary.csv',index=False)
 provenance={'scope':'Ancillary after-run diagnosis; same frozen candidate, gradient, Hessian and scale; original and active QP only','primary_outputs_modified':False,'candidate_rows':len(out),'inputs':int(len(refs[~refs.reference_boundary])),'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'original_core.py',ROOT/'diagnostic_active_core.py',Path(__file__)]},'observed_bound_violations':int(out.bound_violation.sum()),'exact_arithmetic':False}
 (args.results/'auxiliary_accuracy_provenance.json').write_text(json.dumps(provenance,indent=2));print(summary.to_string(index=False));print(json.dumps(provenance))
if __name__=='__main__':main()
