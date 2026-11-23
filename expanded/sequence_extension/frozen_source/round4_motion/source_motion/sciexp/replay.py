from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .core import SCI,solve,inv,sym
from .data import Dataset,wrap,atomic_json
from .motion import ControlResponse

POSE_DIM=15
STATE_DIM=25
HEADINGS=np.arange(2,POSE_DIM,3)


def observe(x,i,subject,landmarks,range_geometry=None,range_robot_target_offset=0.):
    j=3*i; p=x[j:j+3]
    target=x[3*(subject-1):3*(subject-1)+2] if subject<=5 else landmarks[subject]
    dx,dy=target-p[:2];q=dx*dx+dy*dy
    if q<1e-8: raise ArithmeticError('Near-zero observation range')
    r=np.sqrt(q);H=np.zeros((2,len(x)))
    H[:,j:j+3]=[[-dx/r,-dy/r,0],[dy/q,-dx/q,-1]]
    if subject<=5:
        k=3*(subject-1);H[:,k:k+2]=[[dx/r,dy/r],[-dy/q,dx/q]]
    theta=wrap(np.arctan2(dy,dx)-p[2])
    if range_geometry is not None:
        c0,cx,cy=range_geometry
        a=cx*np.cos(theta)+cy*np.sin(theta)
        da=-cx*np.sin(theta)+cy*np.cos(theta)
        H[0]=a*H[0]+r*da*H[1]
        r=c0+a*r
    if subject<=5:r+=range_robot_target_offset
    h=np.array([r,theta])
    if len(x)==STATE_DIM:
        j=POSE_DIM+2*i
        h+=x[j:j+2];h[1]=wrap(h[1]);H[:,j:j+2]=np.eye(2)
    return h,H


def predict(x,A,B,u,dt,density,tau,bias_variance):
    for branch in range(len(x)):
        th=x[branch,HEADINGS].copy();mid=th+.5*u[:,1]*dt
        F=np.eye(STATE_DIM);Q=np.zeros((STATE_DIM,STATE_DIM))
        for i in range(5):
            k=3*i;v=u[i,0];c,s=np.cos(mid[i]),np.sin(mid[i])
            F[k, k+2]=-v*dt*s; F[k+1,k+2]=v*dt*c
            rot=np.array([[c,-s,0],[s,c,0],[0,0,1]])
            Q[k:k+3,k:k+3]=rot@np.diag(density[i]*dt)@rot.T
        phi=np.exp(-dt/tau)
        F[POSE_DIM:,POSE_DIM:]=np.diag(phi)
        Q[POSE_DIM:,POSE_DIM:]=np.diag(bias_variance*(-np.expm1(-2*dt/tau)))
        x[branch,0:POSE_DIM:3]+=u[:,0]*dt*np.cos(mid)
        x[branch,1:POSE_DIM:3]+=u[:,0]*dt*np.sin(mid)
        x[branch,HEADINGS]=wrap(th+u[:,1]*dt)
        x[branch,POSE_DIM:]*=phi
        A[branch]=sym(F@A[branch]@F.T+Q)
        B[branch]=sym(F@B[branch]@F.T)


def update(x,A,B,k,i,subject,z,R,landmarks,gate,range_geometry=None,
           range_robot_target_offset=0.,range_variance_law=None,measurement_inflation=1.):
    h,H=observe(x[k],i,subject,landmarks,range_geometry,range_robot_target_offset)
    if range_variance_law is not None:
        # Heteroscedastic total range residual variance, learned on Dataset1.
        # Evaluate at predicted sensor depth, never the observed noisy range.
        # Range OU is disabled for these candidates; bearing OU remains active.
        depth=h[0]-(x[k,POSE_DIM+2*i] if len(x[k])==STATE_DIM else 0.)
        depth=max(0.,float(depth))
        variance=(range_variance_law['floor_variance']+
                  range_variance_law['coefficient'][i]*depth**range_variance_law['power'])
        R=R.copy();R[0,0]=measurement_inflation*variance
        if R[0,1]!=0. or R[1,0]!=0.:
            raise ValueError('Current heteroscedastic+OU implementation requires diagonal white R')
    r=z-h;r[1]=wrap(r[1]);P=A[k]+B[k]
    S=H@P@H.T+R;Si=inv(S);nis=float(r@Si@r)
    # Predeclared soft-gating ablation: retain the original chi-square threshold,
    # but bound the influence instead of permanently rejecting a recovery cue.
    # This data-dependent robust covariance is an empirical EKF approximation,
    # not a new claim of exact SCI independence or statistical consistency.
    if nis>gate:
        R=R*(nis/gate)
        S=H@P@H.T+R;Si=inv(S)
    K=P@H.T@Si; C=np.eye(len(P))-K@H
    x[k]+=K@r;x[k,HEADINGS]=wrap(x[k,HEADINGS])
    A[k]=sym(C@A[k]@C.T)
    B[k]=sym(C@B[k]@C.T+K@R@K.T)
    return True,nis


def error_metrics(mean,P,truth):
    e=(mean[:POSE_DIM]-truth).reshape(5,3).copy();e[:,2]=wrap(e[:,2])
    pos_nees=[]
    for i in range(5):
        ep=e[i,:2];pos_nees.append(float(ep@inv(P[3*i:3*i+2,3*i:3*i+2])@ep))
    return dict(position_mse=float(np.mean(np.sum(e[:,:2]**2,axis=1))),
                heading_mse=float(np.mean(e[:,2]**2)),
                position_nees=float(np.mean(pos_nees)),
                coverage95=float(np.mean(np.array(pos_nees)<=5.991464547107979)))


def signature(config,cal):
    source_dir=Path(__file__).parent
    code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_dir.glob('*.py'))}
    return hashlib.sha256(json.dumps(dict(config=config,calibration=cal,code=code),sort_keys=True).encode()).hexdigest()


def replay(root,out,ds,seed,config,cal,max_seconds=None):
    out=Path(out)/f'dataset{ds:02d}_seed{seed}';out.mkdir(parents=True,exist_ok=True)
    data=Dataset(root,ds);controls=ControlResponse(data.odom,cal);token=signature(config,cal)
    token=hashlib.sha256((token+str(max_seconds)).encode()).hexdigest()
    meta=out/'manifest.json'
    if meta.exists():
        old=json.loads(meta.read_text())
        if old['signature']!=token: raise RuntimeError(f'{out}: configuration/source changed; use a new output directory')
        if old.get('complete'): print(f'{out.name}: complete, skipping',flush=True);return
    atomic_json(meta,dict(signature=token,base_signature=signature(config,cal),complete=False,dataset=ds,seed=seed,config=config,
        mode='25-state SCI feedback with persistent per-camera OU bias; own closed-loop trajectory',
        truth_use='known map + one initial pose, thereafter evaluation only'))
    period=config['fusion_period'];start=data.start+1.
    end=min(data.end,start+max_seconds) if max_seconds is not None else data.end
    n=int((end-start)//period)
    meas=[r for r in data.measurements if start<r[0]<=start+n*period]
    mt=np.array([r[0] for r in meas])
    rng=np.random.default_rng(seed)
    sd=np.tile([config['initial_position_sd'],config['initial_position_sd'],config['initial_heading_sd']],5)
    pose=data.truth(start)+rng.normal(size=15)*sd;pose[HEADINGS]=wrap(pose[HEADINGS])
    tau=np.array(cal['colored_tau_s']).reshape(10)
    stationary=np.array(cal['colored_stationary_covariance'])
    # Strictly PD A is required by this SCI implementation. White-only selected
    # components receive only a disclosed 1e-12 numerical nuisance variance.
    bias_variance=np.maximum(np.diagonal(stationary,axis1=1,axis2=2).reshape(10),1e-12)
    bias_variance*=config['measurement_inflation']
    feedback_x=np.r_[pose,np.zeros(10)]
    feedback_P=np.diag(np.r_[sd**2,bias_variance])
    begin=0;shards=sorted(out.glob('bank_*.npz'))
    if shards:
        with np.load(shards[-1],allow_pickle=False) as ck:
            if str(ck['signature'])!=token: raise RuntimeError('Checkpoint mismatch')
            begin=int(ck['next_epoch']);feedback_x=ck['feedback_x'];feedback_P=ck['feedback_P']
        print(f'{out.name}: resuming epoch {begin}',flush=True)
    R=np.array(cal['colored_white_covariance'])*config['measurement_inflation']
    bias=np.array(cal['measurement_bias'])
    range_geometry=cal.get('range_geometry')
    range_robot_target_offset=cal.get('range_robot_target_offset',0.)
    range_variance_law=cal.get('range_variance_law')
    if range_variance_law is not None and np.any(stationary[:,0,0]>0):
        raise ValueError('Total range variance law must not double-count active range OU')
    density=np.array(cal['process_density'])*config['process_inflation']
    rows=[];wall=time.perf_counter()
    def save():
        if not rows:return
        arrays={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
        first=int(rows[0]['epoch'])
        path=out/f'bank_{first:06d}.npz';tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as h:
            np.savez_compressed(h,**arrays,feedback_x=feedback_x,feedback_P=feedback_P,
                                next_epoch=int(rows[-1]['epoch'])+1,signature=np.array(token))
        tmp.replace(path);rows.clear()
    for k in range(begin,n):
        a,b=start+k*period,start+(k+1)*period
        # Five camera-local team estimators plus a centralized epoch comparator.
        x=np.tile(feedback_x,(6,1));A=np.tile(feedback_P,(6,1,1));B=np.zeros_like(A)
        left,right=np.searchsorted(mt,[a,b],side='right')
        observations=meas[left:right]
        ticks=np.unique(np.r_[np.arange(a+config['motion_step'],b,config['motion_step']),
                               [r[0] for r in observations],controls.changes_between(a,b),b])
        pos=0;previous=a;accepted=np.zeros(6,int);rejected=np.zeros(6,int);downweighted=np.zeros(6,int)
        for t in ticks:
            if t>previous:
                u=controls.average(previous,t)
                predict(x,A,B,u,t-previous,density,tau,bias_variance);previous=t
            while pos<len(observations) and observations[pos][0]<=t:
                _,i,subject,r,ang=observations[pos];z=np.array([r,ang])-bias[i]
                for branch in (i,5):
                    ok,nis=update(x,A,B,branch,i,subject,z,R[i],data.landmarks,config['innovation_gate'],
                                None if range_geometry is None else range_geometry[i],
                                range_robot_target_offset,range_variance_law,config['measurement_inflation'])
                    if nis>config['innovation_gate']:downweighted[branch]+=1
                    if ok:accepted[branch]+=1
                    else:rejected[branch]+=1
                pos+=1
        means=x[:5].copy()
        means[:,HEADINGS]=means[0,HEADINGS]+wrap(means[:,HEADINGS]-means[0,HEADINGS])
        model=SCI(A[:5],B[:5],config['objective'],config['heading_length'],pose_dimension=POSE_DIM)
        reference=solve(model,tol=config['reference_tolerance'],rule='vertex',maxiter=config['max_iterations'])
        feedback_x,feedback_P=model.fuse(reference['w'],means)
        feedback_x[HEADINGS]=wrap(feedback_x[HEADINGS])
        rows.append(dict(epoch=k,time=b-start,A=A[:5].copy(),B=B[:5].copy(),means=means,
            truth=data.truth(b),ref_w=reference['w'],ref_f=reference['f'],ref_gap=reference['gap'],
            ref_converged=reference['converged'],ref_mean=feedback_x[:POSE_DIM].copy(),ref_P=feedback_P[:POSE_DIM,:POSE_DIM].copy(),
            bias_mean=feedback_x[POSE_DIM:].copy(),bias_P=feedback_P[POSE_DIM:,POSE_DIM:].copy(),
            central_mean=x[5,:POSE_DIM].copy(),central_P=(A[5]+B[5])[:POSE_DIM,:POSE_DIM],accepted=accepted,rejected=rejected,downweighted=downweighted,
            reference_seconds=reference['seconds'],reference_checks=reference['checks']))
        if len(rows)>=128 or k==n-1:save()
        if (k+1)%50==0 or k==n-1:
            elapsed=time.perf_counter()-wall;done=k+1-begin
            print(f'{out.name}: {k+1}/{n} epochs, {elapsed/max(done,1):.3f} s/epoch, last ref gap={reference["gap"]:.2e}',flush=True)
    atomic_json(meta,dict(signature=token,base_signature=signature(config,cal),max_seconds=max_seconds,complete=True,dataset=ds,seed=seed,epochs=n,config=config,
        duration=n*period,unknown_barcodes=data.unknown_barcodes,
        replay_seconds_this_invocation=time.perf_counter()-wall,
        interpretation='Own closed-loop 25-state SCI estimator; pose-only trace objective; bias uncertainty and innovations are shared latent process terms in A; private white measurement noise enters B. Empirical nonlinear robust approximation, not a consistency proof.'))
