from __future__ import annotations
import concurrent.futures
import ftplib
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import zipfile
import numpy as np

OFFICIAL = 'https://asrl.utias.utoronto.ca/datasets/mrclam/index.html'
MIRROR_REPO = 'doclorenzo/SLAM-Project'
MIRROR_COMMIT = '729e8e35c27e9cfcd86ecefa294fca86aa59e576'
FILES = ['Barcodes.dat','Landmark_Groundtruth.dat'] + [f'Robot{i}_{k}.dat' for i in range(1,6)
                for k in ['Groundtruth','Odometry','Measurement']]

def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8');tmp.replace(path)

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fetch(root, datasets, source='auto'):
    """Official FTP first; optional pinned third-party mirror of data only.
    No code from the mirror is executed. Manual official .dat files also work.
    """
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    for ds in datasets:
        target=root/f'MRCLAM_Dataset{ds}';target.mkdir(exist_ok=True)
        if all((target/f).exists() for f in FILES):
            print(f'Dataset {ds}: complete local files, reusing',flush=True);continue
        used=source
        if source in ('official','auto'):
            try:
                with ftplib.FTP('asrl3.utias.utoronto.ca',timeout=20) as ftp:
                    ftp.login();ftp.cwd('MRCLAM')
                    names=[n for n in ftp.nlst() if re.search(rf'dataset[_ -]?0?{ds}(?:\D|$)',n,re.I) and n.lower().endswith('.zip')]
                    if len(names)!=1: raise RuntimeError(f'Ambiguous official archive: {names}')
                    archive=root/f'official_dataset{ds}.zip.part'
                    with archive.open('wb') as h: ftp.retrbinary('RETR '+names[0],h.write)
                with zipfile.ZipFile(archive) as z:
                    for f in FILES:
                        members=[n for n in z.namelist() if Path(n).name==f]
                        if len(members)!=1: raise RuntimeError(f'Missing/ambiguous {f}')
                        (target/f).write_bytes(z.read(members[0]))
                archive.unlink();used='official_ftp'
            except Exception as e:
                if source=='official': raise
                print(f'Official FTP unavailable ({e}); using pinned data mirror.',flush=True)
                used='mirror'
        if used=='mirror':
            base=f'https://raw.githubusercontent.com/{MIRROR_REPO}/{MIRROR_COMMIT}/data/raw/MRCLAM_Dataset{ds}/'
            def one(name):
                path=target/name
                if path.exists(): return
                last=None
                for _ in range(3):
                    try:
                        with urllib.request.urlopen(base+name,timeout=60) as r: data=r.read()
                        if not data or b'<html' in data[:100].lower(): raise ValueError('Unexpected payload')
                        tmp=path.with_suffix('.dat.part');tmp.write_bytes(data);tmp.replace(path);return
                    except Exception as e: last=e
                raise RuntimeError(f'Download failed: {name}: {last}')
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                list(ex.map(one,FILES))
        for f in FILES:
            # Parser check catches truncated responses and HTML error pages.
            a=np.loadtxt(target/f,comments='#',ndmin=2)
            if len(a)==0 or not np.isfinite(a).all(): raise ValueError(f'Invalid {f}')
        atomic_json(target/'provenance.json',dict(source=used,official=OFFICIAL,
            mirror_repo=MIRROR_REPO if used=='mirror' else None,
            mirror_commit=MIRROR_COMMIT if used=='mirror' else None,
            official_checksums_verified=False,sha256={f:digest(target/f) for f in FILES}))
        print(f'Dataset {ds}: downloaded and parsed 17 files ({used})',flush=True)


def wrap(x): return (x+np.pi)%(2*np.pi)-np.pi

class Dataset:
    def __init__(self,root,ds):
        self.root=Path(root)/f'MRCLAM_Dataset{ds}'; self.ds=ds
        if not self.root.exists(): raise FileNotFoundError(f'{self.root}: run download or place official .dat files here')
        def read(name):
            a=np.loadtxt(self.root/name,comments='#',ndmin=2)
            if not np.isfinite(a).all(): raise ValueError(name)
            return a
        bc=read('Barcodes.dat'); self.barcodes={int(b):int(a) for a,b in bc[:,:2]}
        self.landmarks={int(r[0]):r[1:3] for r in read('Landmark_Groundtruth.dat')}
        self.gt=[];self.odom=[];self.measurements=[]
        self.unknown_barcodes=0
        for i in range(1,6):
            gt=read(f'Robot{i}_Groundtruth.dat');gt=gt[np.argsort(gt[:,0],kind='stable')]
            gt[:,3]=np.unwrap(gt[:,3]);self.gt.append(gt)
            od=read(f'Robot{i}_Odometry.dat');od=od[np.argsort(od[:,0],kind='stable')]
            # Duplicate control timestamps: retain the last logged command.
            od=od[np.r_[np.diff(od[:,0])>0,True]];self.odom.append(od)
            m=read(f'Robot{i}_Measurement.dat')
            for row in m:
                subject=self.barcodes.get(int(row[1]))
                if subject is None: self.unknown_barcodes+=1;continue
                if subject==i or row[2]<=0: continue
                self.measurements.append((float(row[0]),i-1,subject,float(row[2]),float(row[3])))
        self.measurements.sort(key=lambda r:r[0])
        self.gt_times=[g[:,0].copy() for g in self.gt]
        self.gt_columns=[[g[:,k].copy() for k in (1,2,3)] for g in self.gt]
        self.start=max(max(g[0,0] for g in self.gt),max(o[0,0] for o in self.odom))
        self.end=min(min(g[-1,0] for g in self.gt),min(o[-1,0] for o in self.odom))
        self.cumulative=[]
        for od in self.odom:
            cs=np.vstack([np.zeros(2),np.cumsum(np.diff(od[:,0])[:,None]*od[:-1,1:3],axis=0)])
            self.cumulative.append(cs)

    def truth(self,t):
        return np.array([np.interp(t,tt,col) for tt,cols in zip(self.gt_times,self.gt_columns) for col in cols])

    def integrated_controls(self,t):
        out=[]
        for od,cs in zip(self.odom,self.cumulative):
            j=np.clip(np.searchsorted(od[:,0],t,side='right')-1,0,len(od)-1)
            out.append(cs[j]+(t-od[j,0])*od[j,1:3])
        return np.array(out)

    def average_controls(self,a,b,cal):
        u=(self.integrated_controls(b)-self.integrated_controls(a))/(b-a)
        return u*np.array(cal['control_scale'])+np.array(cal['control_bias'])


def calibrate(root,ds=1):
    """Only the designated training sequence may enter this routine."""
    data=Dataset(root,ds);scale=[];bias=[];q=[]
    dt=.1
    ts=np.arange(data.start+1,data.end-1,dt)
    truths=data.truth(ts).T
    controls=np.array([data.integrated_controls(t) for t in ts])
    for i in range(5):
        p=truths[:,3*i:3*i+3]; mid=(p[1:,2]+p[:-1,2])/2
        dp=np.diff(p,axis=0); u=np.diff(controls[:,i,:],axis=0)/dt
        vf=(np.cos(mid)*dp[:,0]+np.sin(mid)*dp[:,1])/dt
        vl=(-np.sin(mid)*dp[:,0]+np.cos(mid)*dp[:,1])/dt
        true=np.column_stack([vf,dp[:,2]/dt]);sc=[];bs=[];res=[]
        for k in range(2):
            X=np.column_stack([u[:,k],np.ones(len(u))])
            coef=np.linalg.lstsq(X,true[:,k],rcond=None)[0]
            er=true[:,k]-X@coef
            mask=abs(er-np.median(er))<=max(1e-3,np.quantile(abs(er-np.median(er)),.98))
            coef=np.linalg.lstsq(X[mask],true[mask,k],rcond=None)[0]
            if not .1<coef[0]<3: raise ValueError(f'Implausible control calibration robot {i+1}: {coef}')
            sc.append(float(coef[0]));bs.append(float(coef[1]));res.append(true[:,k]-X@coef)
        # Diffusion spectral densities from 0.1-s increments, with declared floors.
        errors=np.column_stack([res[0],vl,res[1]])
        lim=np.quantile(abs(errors),.99,axis=0)
        qq=np.mean(np.clip(errors,-lim,lim)**2,axis=0)*dt
        q.append(np.maximum(qq,[1e-5,1e-5,1e-5]).tolist());scale.append(sc);bias.append(bs)
    residual=[[] for _ in range(5)]
    for t,i,subject,r,b in data.measurements:
        if not data.start<=t<=data.end: continue
        x=data.truth(t); p=x[3*i:3*i+3]
        target=x[3*(subject-1):3*(subject-1)+2] if subject<=5 else data.landmarks[subject]
        dv=target-p[:2]
        residual[i].append([r-np.linalg.norm(dv),wrap(b-(np.arctan2(dv[1],dv[0])-p[2]))])
    R=[];mbias=[];counts=[];retained=[]
    for i,rr in enumerate(residual):
        rr=np.array(rr)
        if len(rr)<100: raise ValueError(f'Insufficient calibration observations robot {i+1}')
        med=np.median(rr,axis=0)
        mask=np.all(abs(rr-med)<=np.maximum(np.quantile(abs(rr-med),.98,axis=0),[.01,.005]),axis=1)
        rr_fit=rr[mask];mu=rr_fit.mean(axis=0);cov=np.cov(rr_fit,rowvar=False)
        cov+=np.diag([.02**2,.005**2])  # fixed sensor-model floors, not test tuning
        R.append(cov.tolist());mbias.append(mu.tolist());counts.append(len(rr));retained.append(int(mask.sum()))
    return dict(training_dataset=ds,control_scale=scale,control_bias=bias,
                process_density=q,measurement_covariance=R,measurement_bias=mbias,
                calibration_observations=counts,calibration_retained=retained,
                interpretation='Empirical fixed EKF model; independence and unbiasedness not established by calibration',
                input_sha256={f:digest(data.root/f) for f in FILES})
