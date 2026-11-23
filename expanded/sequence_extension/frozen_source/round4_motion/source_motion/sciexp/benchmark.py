from __future__ import annotations
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .core import SCI,solve,inv
from .data import atomic_json
from .replay import error_metrics,signature

RULES={'fw':'SCI-FW','cheap':'SCI-GQ-cheap','vertex':'SCI-GQ-vertex'}

def benchmark(bank_root,out,config,cal,datasets,seeds):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    token=signature(config,cal)
    meta=out/'benchmark_manifest.json'
    if meta.exists() and json.loads(meta.read_text())['signature']!=token:
        raise RuntimeError('Benchmark configuration changed; use a new output directory')
    atomic_json(meta,dict(signature=token,config=config,platform=platform.platform(),
        python=platform.python_version(),numpy=np.__version__,
        arithmetic='float64 with engineering margins; no exact or interval verification',
        metric='time includes fresh model construction, optimizer and stopping overhead',
        scope='paired fusion-epoch outputs on a common reference feedback trajectory'))
    warmed=False
    for ds in datasets:
      for seed in seeds:
        bank=Path(bank_root)/f'dataset{ds:02d}_seed{seed}'
        if not (bank/'manifest.json').exists(): raise FileNotFoundError(bank)
        bank_meta=json.loads((bank/'manifest.json').read_text())
        if not bank_meta.get('complete'):raise RuntimeError(f'{bank}: replay incomplete')
        # Verify common calibration/config/source independently of max_seconds token.
        if bank_meta['config']!=config or bank_meta.get('base_signature')!=token:
            raise RuntimeError(f'{bank}: replay calibration/config/source mismatch')
        for path in sorted(bank.glob('bank_*.npz')):
            dest=out/f'ds{ds:02d}_seed{seed}_{path.stem}.csv'
            receipt=dest.with_suffix('.json')
            input_hash=hashlib.sha256(path.read_bytes()).hexdigest()
            if dest.exists() and receipt.exists():
                old=json.loads(receipt.read_text())
                if old['signature']==token and old['input_sha256']==input_hash:continue
                raise RuntimeError(f'Stale result: {dest}')
            rows=[]
            with np.load(path,allow_pickle=False) as batch:
              selected=[j for j,k in enumerate(batch['epoch'])
                        if int(k)%config['benchmark_stride']==0 and batch['time'][j]>=config['burn_in_seconds']]
              for j in selected:
                A,B,means=batch['A'][j],batch['B'][j],batch['means'][j]
                truth=batch['truth'][j];epoch=int(batch['epoch'][j])
                base=dict(dataset=ds,seed=seed,epoch=epoch,time=float(batch['time'][j]),
                    ref_gap=float(batch['ref_gap'][j]),ref_converged=bool(batch['ref_converged'][j]),
                    reference_boundary=bool(batch['ref_w'][j].min()<1e-5),
                    active_cameras=int(np.count_nonzero(batch['accepted'][j,:5])),
                    accepted=int(batch['accepted'][j,:5].sum()),rejected=int(batch['rejected'][j,:5].sum()))
                fref=float(batch['ref_f'][j]);refgap=float(batch['ref_gap'][j])
                if not warmed:
                    for rule in RULES:
                        solve(SCI(A,B,config['objective'],config['heading_length']),rule=rule,
                              tol=config['tolerances'][0],maxiter=config['max_iterations'])
                    warmed=True
                jobs=[(tol,rule,rep) for tol in config['tolerances'] for rule in RULES
                      for rep in range(config['timing_repeats'])]
                rng=np.random.default_rng(seed*100000+epoch);rng.shuffle(jobs)
                for tol,rule,rep in jobs:
                    t0=time.perf_counter()
                    model=SCI(A,B,config['objective'],config['heading_length'])
                    construction=time.perf_counter()-t0
                    result=solve(model,tol=tol,rule=rule,maxiter=config['max_iterations'])
                    total=construction+result['seconds']
                    mean,P=model.fuse(result['w'],means)
                    # fref is a feasible value, not an assumed exact optimum.
                    lower=max(0.,result['f']-fref)
                    upper=max(0.,result['f']-fref+refgap)
                    violation=result['gap']+2e-9<lower
                    rows.append(dict(**base,method=RULES[rule],tolerance=tol,repeat=rep,
                        **error_metrics(mean,P,truth),seconds=total,construction_seconds=construction,
                        certificate_seconds=result['certificate_seconds'],checks=result['checks'],
                        gap=result['gap'],fw_gap=result['fw'],gq_gap=result['gq'],b=result['b'],
                        objective=result['f'],suboptimality_lower=lower,suboptimality_upper=upper,
                        converged=result['converged'],bound_violation=violation,
                        certificate_failures=result['certificate_failures'],status=result['status']))
                # Estimation-context baselines on the same inputs, evaluated once.
                model=SCI(A,B,config['objective'],config['heading_length'])
                uniform=np.ones(5)/5;um,up=model.fuse(uniform,means)
                ci=SCI(A+B,np.zeros_like(B),config['objective'],config['heading_length'])
                cr=solve(ci,tol=config['reference_tolerance'],rule='fw',maxiter=config['max_iterations'])
                cm,cp=ci.fuse(cr['w'],means)
                infos=np.array([inv(t) for t in A+B]);np_=inv(infos.sum(axis=0))
                nm=np_@np.einsum('sij,sj->i',infos,means)
                baselines=[('SCI-reference',batch['ref_mean'][j],batch['ref_P'][j],base['ref_converged']),
                           ('SCI-uniform',um,up,True),('CI',cm,cp,cr['converged']),
                           ('Naive-independent',nm,np_,True),
                           ('Centralized-epoch',batch['central_mean'][j],batch['central_P'][j],True)]
                for name,mean,P,ok in baselines:
                    rows.append(dict(**base,method=name,tolerance=0.,repeat=0,
                                     **error_metrics(mean,P,truth),converged=ok))
            tmp=dest.with_suffix('.csv.tmp')
            pd.DataFrame(rows).to_csv(tmp,index=False);tmp.replace(dest)
            atomic_json(receipt,dict(signature=token,input_sha256=input_hash,rows=len(rows)))
            print(f'{dest.name}: {len(selected)} inputs, {len(rows)} rows',flush=True)


def summarize(results,out,main_tol=.001):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    files=sorted(Path(results).glob('ds*_bank_*.csv'))
    frames=[pd.read_csv(p) for p in files if p.stat().st_size>2]
    if not frames: raise ValueError('No completed benchmark rows')
    raw=pd.concat(frames,ignore_index=True)
    keys=['dataset','seed','epoch','method','tolerance']
    # Repeats are timing repetitions, never independent accuracy samples.
    numeric=['seconds','construction_seconds','certificate_seconds','checks','gap','fw_gap',
             'position_mse','heading_mse','position_nees','coverage95',
             'suboptimality_lower','suboptimality_upper','reference_boundary','active_cameras']
    paired=raw.groupby(keys,dropna=False)[numeric].median().reset_index()
    status=raw.groupby(keys).agg(converged=('converged','all'),
                                bound_violation=('bound_violation','max'),
                                certificate_failures=('certificate_failures','max')).reset_index()
    paired=paired.merge(status,on=keys)
    paired.to_csv(out/'paired_epoch_results.csv',index=False)
    groups=['dataset','seed','method','tolerance']
    seq=paired.groupby(groups).agg(position_mse=('position_mse','mean'),heading_mse=('heading_mse','mean'),
        coverage95=('coverage95','mean'),position_nees=('position_nees','mean'),
        median_ms=('seconds',lambda s:1000*s.median() if s.notna().any() else np.nan),
        p95_ms=('seconds',lambda s:1000*s.quantile(.95) if s.notna().any() else np.nan),
        median_checks=('checks','median'),success_rate=('converged','mean'),
        bound_violations=('bound_violation','sum'),epochs=('epoch','count')).reset_index()
    seq['position_rmse_m']=np.sqrt(seq.pop('position_mse'))
    seq['heading_rmse_deg']=np.sqrt(seq.pop('heading_mse'))*180/np.pi
    seq.to_csv(out/'per_sequence.csv',index=False)
    # Equal weight per sequence; not per overlapping sensor observation.
    summary=seq.groupby(['method','tolerance']).agg(
        position_rmse_m=('position_rmse_m','mean'),heading_rmse_deg=('heading_rmse_deg','mean'),
        coverage95=('coverage95','mean'),position_nees=('position_nees','mean'),
        median_ms=('median_ms','mean'),p95_ms=('p95_ms','mean'),
        median_checks=('median_checks','mean'),success_rate=('success_rate','mean'),
        bound_violations=('bound_violations','sum'),sequences=('dataset','nunique')).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    times=paired[paired.method.isin(RULES.values())].pivot(index=['dataset','seed','epoch','tolerance'],columns='method',values='seconds')
    speed=pd.DataFrame(index=times.index)
    for method in ['SCI-GQ-cheap','SCI-GQ-vertex']:
        speed[method]=times['SCI-FW']/times[method]
    speed.reset_index().to_csv(out/'paired_speedups.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for method in RULES.values():
        z=summary[summary.method==method].sort_values('tolerance')
        axes[0].plot(z.tolerance,z.median_ms,'o-',label=method)
    axes[0].set(xscale='log',yscale='log',xlabel='Normalized objective tolerance',ylabel='End-to-end fusion optimization (ms)')
    axes[0].legend(fontsize=8)
    names=['SCI-reference','SCI-FW','SCI-GQ-cheap','SCI-GQ-vertex','SCI-uniform','CI','Naive-independent','Centralized-epoch']
    sel=summary[np.isclose(summary.tolerance,main_tol)|np.isclose(summary.tolerance,0)].set_index('method').reindex(names)
    labels=['SCI ref.','SCI FW','GQ cheap','GQ vertex','Uniform','CI','Naive','Central epoch']
    for ax,col,ylabel in zip(axes[1:],['position_rmse_m','coverage95'],['Position RMSE (m)','95% position-ellipse coverage']):
        ax.bar(np.arange(len(names)),sel[col],color=['#566573','#4477AA','#66CCEE','#228833','#CCBB44','#AA3377','#EE6677','#BBBBBB'])
        ax.set_xticks(np.arange(len(names)),labels,rotation=50,ha='right');ax.set_ylabel(ylabel)
    axes[2].axhline(.95,color='black',ls='--',lw=1);axes[2].set_ylim(0,1.02)
    for ext in ('pdf','png'):fig.savefig(out/f'real_tracking_main.{ext}',dpi=220)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    for method in RULES.values():
        z=paired[(paired.method==method)&np.isclose(paired.tolerance,main_tol)]
        vals=np.sort(z.seconds.dropna().to_numpy()*1000)
        if len(vals):axes[0].plot(vals,np.arange(1,len(vals)+1)/len(vals),label=method)
    axes[0].set(xscale='log',xlabel='Optimization latency (ms)',ylabel='Empirical CDF');axes[0].legend(fontsize=8)
    main=paired[np.isclose(paired.tolerance,main_tol)&paired.method.isin(RULES.values())]
    for ii,method in enumerate(RULES.values()):
        vals=[]
        for boundary in (False,True):
            z=main[(main.method==method)&(main.reference_boundary==boundary)]
            ratio=z.fw_gap/np.maximum(z.gap,1e-14);vals.append(float(ratio.median()) if len(z) else np.nan)
        axes[1].bar(np.arange(2)+(ii-1)*.24,vals,width=.24,label=method)
    axes[1].set_xticks([0,1],['Interior reference','Boundary reference'])
    axes[1].set(yscale='log',ylabel='Median current FW gap / retained bound');axes[1].legend(fontsize=8)
    for ext in ('pdf','png'):fig.savefig(out/f'certificate_behavior.{ext}',dpi=220)
    plt.close(fig)
    fails=paired[(~paired.converged)|(paired.bound_violation.eq(True))]
    fails.to_csv(out/'failures.csv',index=False)
    atomic_json(out/'quality_report.json',dict(rows=len(raw),paired_rows=len(paired),
        sequences=sorted(raw.dataset.unique().astype(int).tolist()),
        unconverged=int((~paired.converged).sum()),
        observed_bound_violations=int(paired.bound_violation.eq(True).sum()),
        reference_unconverged_rows=int((~raw.ref_converged).sum()),
        interpretation='No failure excluded from latency/accuracy aggregates. Repeated timings collapsed first.'))
    print(summary.to_string(index=False),flush=True)


def plot_trajectories(bank_root,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for bank in sorted(Path(bank_root).glob('dataset*_seed*')):
        shards=sorted(bank.glob('bank_*.npz'))
        if not shards:continue
        times=[];truth=[];est=[];accepted=[];rejected=[];refok=[]
        for shard in shards:
            with np.load(shard,allow_pickle=False) as z:
                times.append(z['time']);truth.append(z['truth']);est.append(z['ref_mean'])
                accepted.append(z['accepted']);rejected.append(z['rejected']);refok.append(z['ref_converged'])
        t=np.concatenate(times);gt=np.concatenate(truth);x=np.concatenate(est)
        ac=np.concatenate(accepted);rj=np.concatenate(rejected)
        fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
        for i,ax in enumerate(axes.flat):
            if i<5:
                k=3*i;ax.plot(gt[:,k],gt[:,k+1],lw=1,label='Motion capture')
                ax.plot(x[:,k],x[:,k+1],lw=.8,alpha=.8,label='SCI reference feedback')
                ax.set(title=f'Robot {i+1}',xlabel='x (m)',ylabel='y (m)',aspect='equal')
            else:
                err=(x-gt).reshape(-1,5,3)
                ax.plot(t,np.sqrt(np.mean(np.sum(err[:,:,:2]**2,axis=2),axis=1)),lw=.8)
                ax.set(xlabel='Elapsed time (s)',ylabel='Team position RMSE (m)')
        axes[0,0].legend(fontsize=8);fig.suptitle(bank.name+' | common SCI-reference feedback')
        for ext in ('pdf','png'):fig.savefig(out/(bank.name+'_trajectory.'+ext),dpi=200)
        plt.close(fig)
        counts=ac.sum(axis=0)+rj.sum(axis=0)
        atomic_json(out/(bank.name+'_tracking_diagnostics.json'),dict(
            epochs=len(t),accepted_per_branch=ac.sum(axis=0).tolist(),
            rejected_per_branch=rj.sum(axis=0).tolist(),
            rejection_fraction=(rj.sum(axis=0)/np.maximum(counts,1)).tolist(),
            unconverged_reference_epochs=int((~np.concatenate(refok)).sum()),
            note='Branches 1..5 are camera-local; branch 6 is centralized epoch comparator.'))
