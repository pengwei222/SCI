"""CLI: python run.py --help. Windows/Linux, CPU float64, no CUDA dependency."""
import argparse
import json
import os
from pathlib import Path
# Set before importing NumPy; avoid tiny-matrix BLAS oversubscription.
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name,'1')
from threadpoolctl import threadpool_limits
from sciexp.data import fetch,calibrate,atomic_json
from sciexp.replay import replay
from sciexp.benchmark import benchmark,summarize,plot_trajectories


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['download','calibrate','replay','benchmark','summarize','all'])
    p.add_argument('--data',default='data')
    p.add_argument('--out',default='runs/main')
    p.add_argument('--config',default=str(Path(__file__).parent/'configs/default.json'))
    p.add_argument('--calibration',default='calibration.json')
    p.add_argument('--datasets',nargs='+',type=int,help='Defaults: 1..9 for download, configured test set otherwise')
    p.add_argument('--seeds',nargs='+',type=int)
    p.add_argument('--source',choices=['auto','official','mirror'],default='auto')
    p.add_argument('--max-seconds',type=float,help='Only for code smoke tests, stored in replay signature')
    p.add_argument('--stride',type=int,help='Override paired benchmark sampling stride')
    p.add_argument('--repeats',type=int)
    p.add_argument('--burn-in',type=float)
    args=p.parse_args();config=json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.stride is not None:config['benchmark_stride']=args.stride
    if args.repeats is not None:config['timing_repeats']=args.repeats
    if args.burn_in is not None:config['burn_in_seconds']=args.burn_in
    if args.seeds is not None:config['seeds']=args.seeds
    if config['benchmark_stride']<1 or config['timing_repeats']<1:raise ValueError('Positive stride/repeats required')
    ds=args.datasets or (list(range(1,10)) if args.stage=='download' else config['test_datasets'])
    if any(d not in range(1,10) for d in ds):raise ValueError('Datasets must be 1..9')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.stage=='download':fetch(args.data,ds,args.source);return
    if args.stage=='calibrate':
        path=Path(args.calibration)
        if path.exists():raise FileExistsError(f'{path}: calibration is frozen; choose a new filename to recalibrate')
        cal=calibrate(args.data,config['training_dataset']);atomic_json(path,cal)
        print(f'Saved training-only calibration to {path}',flush=True);return
    if args.stage=='summarize':
        summarize(out/'benchmark',out/'figures');plot_trajectories(out/'banks',out/'figures');return
    cal=json.loads(Path(args.calibration).read_text(encoding='utf-8'))
    if cal['training_dataset']!=config['training_dataset']:raise ValueError('Calibration split mismatch')
    if args.stage in ('replay','all'):
        for d in ds:
            for seed in config['seeds']:
                replay(args.data,out/'banks',d,seed,config,cal,args.max_seconds)
    if args.stage in ('benchmark','all'):
        benchmark(out/'banks',out/'benchmark',config,cal,ds,config['seeds'])
    if args.stage=='all':
        summarize(out/'benchmark',out/'figures');plot_trajectories(out/'banks',out/'figures')

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
