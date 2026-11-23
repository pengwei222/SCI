"""Recompute Appendix K from the retained frozen-reference diagnostics."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent

def compute():
    rows=[]
    data=pd.read_csv(ROOT/'data/d2_d3_reference_uniform.csv')
    for ds,g in data.groupby('dataset'):
        for method in (['reference','uniform'] if ds==3 else ['reference']):
            # Original protocol: RMSE within each seed, then mean over seeds.
            per_seed=g.groupby('seed').agg(mse=(method+'_position_mse','mean'),coverage=(method+'_coverage95','mean'))
            rows.append(dict(dataset=int(ds),method=method,robot=0,inputs=len(g),seeds=len(per_seed),position_rmse_m=float(np.sqrt(per_seed.mse).mean()),coverage95=float(per_seed.coverage.mean())))
    for ds in [4,5]:
        with np.load(ROOT/f'data/d{ds}_reference.npz',allow_pickle=False) as z:
            error=(z['ref_mean']-z['truth']).reshape(-1,5,3)[:,:,:2]
            nees=np.empty(error.shape[:2]);P=z['ref_P']
            for robot in range(5):
                ids=np.array([3*robot,3*robot+1]);block=P[:,ids[:,None],ids[None,:]]
                white=np.linalg.solve(np.linalg.cholesky(block),error[:,robot,:,None])[...,0]
                nees[:,robot]=np.sum(white**2,axis=1)
            for robot in [0,1,2,3,4,5]:
                e=error if robot==0 else error[:,robot-1,:];n=nees if robot==0 else nees[:,robot-1]
                rows.append(dict(dataset=ds,method='reference',robot=robot,inputs=len(error),seeds=1,position_rmse_m=float(np.sqrt(np.mean(np.sum(e*e,axis=-1)))),coverage95=float(np.mean(n<=-2*np.log(.05)))))
    return pd.DataFrame(rows)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);args=parser.parse_args()
    table=compute();print(table.to_string(index=False))
    expected={2:.0961,3:.0471,4:.0406,5:.6441}
    for ds,value in expected.items():
        got=table.query('dataset==@ds and method=="reference" and robot==0').iloc[0]
        assert abs(got.position_rmse_m-value)<.000051,(ds,got.position_rmse_m)
    d3=table.query('dataset==3').set_index('method')
    assert abs(d3.loc['reference','position_rmse_m']*100-4.7054)<.000051
    assert abs(d3.loc['uniform','position_rmse_m']*100-4.7271)<.000051
    for ds,pct in [(3,83.95),(4,81.62),(5,74.88)]:
        got=table.query('dataset==@ds and method=="reference" and robot==0').coverage95.iloc[0]
        assert abs(100*got-pct)<.0051
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);table.to_csv(args.output,index=False)
