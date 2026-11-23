"""Read each compressed bank array once, then call the unchanged timing harness.

This adapter affects only untimed input loading. Case IDs, fields, selection,
order before the fixed shuffle, and all timed algorithms are unchanged.
"""
import os
for name in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"]:os.environ[name]="1"
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import run_expanded as benchmark


def load_cached(path):
    cases=[]
    for p in sorted(Path(path).rglob('bank_*.npz')):
        meta=json.loads((p.parent/'manifest.json').read_text())
        with np.load(p,allow_pickle=False) as z:
            arrays={k:z[k] for k in ['epoch','time','A','B','means']}
            if 'ref_w' in z: arrays['ref_w']=z['ref_w']
        for j,epoch in enumerate(arrays['epoch']):
            if int(epoch)%5 or arrays['time'][j]<30: continue
            ds,seed=int(meta['dataset']),int(meta['seed'])
            c={k:arrays[k][j].copy() for k in ['A','B','means']}
            if 'ref_w' in arrays:c['ref_w']=arrays['ref_w'][j].copy()
            c.update(case_id=f'recorded_d{ds:02d}_s{seed}_e{int(epoch):05d}',panel='recorded',regime='recorded',d=c['A'].shape[1],s=len(c['A']),seed=seed,dataset=ds,epoch=int(epoch),mode='trace',pose_dimension=15,time=float(arrays['time'][j]))
            cases.append(c)
    return cases

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--banks',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--tolerances',nargs='+',type=float,default=[.01,.003,.001,.0003,.0001,.00003,.00001])
    args=p.parse_args();args.limit=0
    cases=load_cached(args.banks)
    benchmark.run(cases,args)
    (args.output/'untimed_input_adapter.json').write_text(json.dumps({
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Compressed bank loading only; identical case construction and unchanged timed harness',
        'inputs':len(cases)},indent=2))
