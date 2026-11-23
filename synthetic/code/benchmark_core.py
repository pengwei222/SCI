"""Synthetic SCI benchmark. Normalized trace/logdet, float64, CPU.
No floating-point output is described as a rigorous certificate.
The construction stores covariance factors (and exact rank labels), avoiding
rank thresholding in the singular test families.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.optimize import minimize, root


def sym(X): return (X + X.T)*0.5

def feasible(w):
    w=np.maximum(np.asarray(w,dtype=float),0.0)
    if not np.isfinite(w).all() or w.sum()<=0: raise ValueError('invalid weights')
    return w/w.sum()

def invspd(X):
    return sym(cho_solve(cho_factor(sym(X),lower=True,check_finite=False),np.eye(len(X)),check_finite=False))

class SCI:
    """J_i(t)=sum_j t/(a_ij+(1-a_ij)t) z_ij z_ij^T.

    a=0 uses its continuous extension 1. a=1 gives t.
    Main objective uses W=I/d and logdet/d.
    """
    def __init__(self, Zs, aas, meta=None):
        self.Zs=[np.asarray(z,float) for z in Zs]
        self.aas=[np.asarray(a,float) for a in aas]
        self.s=len(Zs); self.d=len(Zs[0]); self.meta=meta or {}
        self.M0=None; self.Mi=None; self.vertices=None; self.setup_lift_seconds=0.
        self.calls=0
    def covariances(self):
        As=[]; Bs=[]
        for Z,a in zip(self.Zs,self.aas):
            F=np.linalg.inv(Z.T)
            As.append(sym((F*a)@F.T)); Bs.append(sym((F*(1-a))@F.T))
        return As,Bs
    def info(self,w,order=2):
        w=np.asarray(w,float)
        if len(w)!=self.s or np.min(w)<-1e-12 or abs(w.sum()-1)>1e-10:
            raise ValueError('weight outside simplex')
        J=np.zeros((self.d,self.d)); Ds=[]; Fs=[]
        for t,Z,a in zip(w,self.Zs,self.aas):
            b=1-a; den=a+b*t; zero=a==0
            safe=np.where(zero,1.,den)
            q=np.where(zero,1.,t/safe)
            J+=(Z*q)@Z.T
            if order:
                qp=np.where(zero,0.,a/safe**2)
                Ds.append(sym((Z*qp)@Z.T))
            if order>=2:
                qpp=np.where(zero,0.,-2*a*b/safe**3)
                Fs.append(sym((Z*qpp)@Z.T))
        return sym(J),np.array(Ds),np.array(Fs)
    def oracle(self,w,mode='logdet',order=2):
        self.calls+=1
        J,D,F=self.info(w,order)
        factor=cho_factor(J,lower=True,check_finite=False)
        if mode=='logdet': val=-2*np.log(np.diag(factor[0])).sum()/self.d
        P=None
        if mode=='trace' or order:
            P=sym(cho_solve(factor,np.eye(self.d),check_finite=False))
        if mode=='trace': val=np.trace(P)/self.d
        if not order: return float(val)
        if mode=='logdet':
            g=-np.einsum('ab,iba->i',P,D)/self.d
        else:
            g=-np.einsum('ab,iba->i',P@P,D)/self.d
        if order==1: return float(val),g
        LP=np.linalg.cholesky(P)
        if mode=='logdet':
            K=LP.T@D@LP
            H=K.reshape(self.s,-1)@K.reshape(self.s,-1).T/self.d
            H[np.diag_indices(self.s)]-=np.einsum('ab,iba->i',P,F)/self.d
        else:
            Y=LP.T@D@P
            H=2*Y.reshape(self.s,-1)@Y.reshape(self.s,-1).T/self.d
            H[np.diag_indices(self.s)]-=np.einsum('ab,iba->i',P@P,F)/self.d
        return float(val),g,sym(H),P
    def build_lift(self):
        start=time.perf_counter()
        rs=[np.sum((a>0)&(a<1)) for a in self.aas]
        n=self.d+sum(rs); M0=np.zeros((n,n)); Mi=np.zeros((self.s,n,n))
        off=self.d
        for i,(Z,a,r) in enumerate(zip(self.Zs,self.aas,rs)):
            zero=a==0; one=a==1; mid=(a>0)&(a<1)
            M0[:self.d,:self.d]+=Z[:,zero]@Z[:,zero].T
            Mi[i,:self.d,:self.d]+=Z[:,one]@Z[:,one].T
            if r:
                zz=Z[:,mid]; aa=a[mid]; bb=1-aa
                M0[:self.d,:self.d]+=(zz/bb)@zz.T
                L=zz*np.sqrt(aa/bb)
                M0[:self.d,off:off+r]=L; M0[off:off+r,:self.d]=L.T
                M0[off:off+r,off:off+r]=np.diag(aa)
                Mi[i,off:off+r,off:off+r]=np.diag(bb)
                off+=r
        self.M0=sym(M0); self.Mi=Mi; self.vertices=[sym(M0+V) for V in Mi]
        self.n=n; self.setup_lift_seconds=time.perf_counter()-start
        return self.setup_lift_seconds
    def lift(self,w):
        if self.M0 is None: self.build_lift()
        return sym(self.M0+np.einsum('i,ijk->jk',w,self.Mi))
    def scales(self, w, both=False, *, verified=False, diagnostics=False):
        # Near a split eigenvalue endpoint, the direct lift may lose its
        # Schur complement to cancellation, even when LAPACK reports success.
        near_endpoint = any(np.any((a > 0) & (a < 1) &
                                  (np.minimum(a, 1 - a) < np.sqrt(np.finfo(float).eps)))
                            for a in self.aas)
        if verified or near_endpoint:
            from verified_scale import FactoredScaleCheck, verified_scales
            a, b, details = verified_scales(self, FactoredScaleCheck(self), w, both)
            return (a, b, details) if diagnostics else (a, b)
        M=self.lift(w); a=1.; b=1.
        for V in self.vertices:
            if both:
                ev=eigh(V,M,eigvals_only=True,check_finite=False,driver='gvd')
                a=min(a,float(ev[0])); b=max(b,float(ev[-1]))
            else:
                ev=eigh(V,M,eigvals_only=True,check_finite=False,driver='gvx',subset_by_index=(self.n-1,self.n-1))
                b=max(b,float(ev[-1]))
        if a<=0 or not np.isfinite(b): raise ArithmeticError('nonpositive numerical lift scale')
        # Preserve the original floating-point path for well-separated inputs.
        a, b = (a/(1+1e-9) if both else 0.), b*(1+1e-9)
        details = {"scale_status": "floating_vertex", "spectral_proposal": b/(1+1e-9)}
        return (a, b, details) if diagnostics else (a, b)


def random_orthogonal(rng,d): return np.linalg.qr(rng.normal(size=(d,d)))[0]

def generate(d,s,cond,seed,kind='balanced'):
    rng=np.random.default_rng(seed)
    Zs=[]; aa=[]
    for i in range(s):
        U=random_orthogonal(rng,d); V=random_orthogonal(rng,d)
        eig=np.geomspace(1,cond,d); eig/=np.exp(np.log(eig).mean())
        if kind=='heterogeneous': scale=np.exp(rng.uniform(-2.3,2.3))
        else: scale=np.exp(rng.uniform(-.15,.15))
        Tinvroot=(U/np.sqrt(eig*scale))@U.T
        Z=Tinvroot@V
        a=rng.uniform(.15,.85,d)
        if kind=='affine': a=np.ones(d)
        elif kind=='singular':
            a=np.resize(np.array([0.,1.,.3,.7]),d); rng.shuffle(a)
        elif kind=='near_independent': a=np.full(d,1e-10)
        elif kind=='near_ci': a=np.full(d,1-1e-10)
        Zs.append(Z); aa.append(a)
    return SCI(Zs,aa,dict(d=d,s=s,cond=cond,seed=seed,kind=kind))

@dataclass
class QP:
    z: np.ndarray
    value: float
    dual: float
    gap: float
    iterations: int
    status: str


def qp(c,g,Q,atol=1e-12,max_iter=200):
    """Active-set simplex QP; returns a valid feasible-point dual formula
    even if stationarity tolerance is not reached. Zero-curvature faces use
    least squares. No PSD clipping. Roundoff residuals are recorded.
    """
    c=feasible(c); Q=sym(Q); g=np.asarray(g)-np.mean(g)
    z=c.copy(); F=set(np.flatnonzero(z>1e-14).tolist())
    def result(it,status):
        v=z-c; value=float(g@v+.5*v@Q@v); r=g+Q@v
        gap=max(0.,float(r@z-r.min()))
        return QP(z.copy(),value,value-gap,gap,it,status)
    for it in range(max_iter):
        r=g+Q@(z-c); gap=max(0.,float(r@z-r.min()))
        if gap<=atol: return result(it,'gap_met')
        ids=np.array(sorted(F),int); m=len(ids)
        K=np.empty((m+1,m+1)); K[:m,:m]=Q[np.ix_(ids,ids)]
        K[:m,m]=1; K[m,:m]=1; K[m,m]=0
        rhs=np.r_[-r[ids],0.]
        try: sol=np.linalg.solve(K,rhs)
        except np.linalg.LinAlgError: sol=np.linalg.lstsq(K,rhs,rcond=1e-13)[0]
        p=np.zeros_like(z); p[ids]=sol[:m]
        residual=np.linalg.norm(K@sol-rhs,np.inf)
        if residual>1e-8*max(1,np.linalg.norm(rhs,np.inf)):
            p[ids]=-(r[ids]-r[ids].mean())
        descent=float(r@p)
        if np.linalg.norm(p,np.inf)<=2e-14 or descent>=-1e-28:
            j=int(np.argmin(r))
            if j not in F: F.add(j); continue
            # Standard conditional-gradient fallback is always feasible.
            p=-z; p[j]+=1
            curve=float(p@Q@p)
            alpha=1 if curve<=0 else min(1.,gap/curve)
        else:
            negative=p < -1e-16
            alpha=min(1.,float(np.min(-z[negative]/p[negative]))) if np.any(negative) else 1.
            if residual>1e-8*max(1,np.linalg.norm(rhs,np.inf)):
                curv=float(p@Q@p)
                if curv>0: alpha=min(alpha,-descent/curv)
        zn=feasible(z+alpha*p)
        F=set(j for j in F if not (zn[j]<=1e-14 and p[j]<-1e-16))
        F.update(np.flatnonzero(zn>1e-14).tolist())
        if np.max(np.abs(zn-z))<2e-16:
            z=zn; return result(it+1,'stagnation')
        z=zn
    return result(max_iter,'limit')


def fw_gap(g,w): return max(0.,float(g@w-g.min()))


def certificate(model,c,mode,bkind='vertex',tol=1e-12, *, verify_scale=False):
    start=time.perf_counter(); f,g,H,P=model.oracle(c,mode,2); oracle_seconds=time.perf_counter()-start
    t=time.perf_counter()
    if bkind=='cheap':
        if min(c)<=0: raise ValueError('cheap bound requires interior')
        b=1/min(c)
        details = {'scale_status':'analytic_interior', 'spectral_proposal':None}
    else:
        _, b, details = model.scales(c, verified=verify_scale, diagnostics=True)
    scale_seconds=time.perf_counter()-t
    t=time.perf_counter(); out=qp(c,g,H/b,atol=tol)
    Gfw=fw_gap(g,c); G=max(0.,min(Gfw,-out.dual)); qp_seconds=time.perf_counter()-t
    return dict(f=f,gfw=Gfw,gq=G,b=b,qp_gap=out.gap,qp_iterations=out.iterations,
                qp_status=out.status,oracle_seconds=oracle_seconds,scale_seconds=scale_seconds,
                qp_seconds=qp_seconds,total_seconds=time.perf_counter()-start,z=out.z,**details)


def slsqp(model,mode,ftol=1e-13,maxiter=300,w0=None,trace=False):
    c=np.ones(model.s)/model.s if w0 is None else feasible(w0)
    path=[c.copy()] if trace else []
    cache={}; evaluations=0
    def fg(w):
        nonlocal evaluations
        key=np.asarray(w).tobytes()
        if cache.get('key')!=key:
            f,g=model.oracle(feasible(w),mode,1)
            cache.update(key=key,f=f,g=g); evaluations+=1
        return cache['f'],cache['g']
    def callback(w):
        if trace: path.append(feasible(w).copy())
    begin=time.perf_counter()
    res=minimize(lambda w:fg(w)[0],c,jac=lambda w:fg(w)[1],method='SLSQP',
                 constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(model.s)},
                 bounds=[(0.,1.)]*model.s,callback=callback,
                 options=dict(ftol=ftol,maxiter=maxiter,disp=False))
    seconds=time.perf_counter()-begin
    w=feasible(res.x); f,g=model.oracle(w,mode,1)
    if trace and np.linalg.norm(path[-1]-w)>0: path.append(w)
    return dict(w=w,f=f,gfw=fw_gap(g,w),seconds=seconds,success=bool(res.success),
                nit=int(res.nit),evaluations=evaluations,path=path,message=str(res.message))


def newton(model,mode,eps=1e-10,max_iter=150,w0=None,stopping='fw',trace=False):
    """Direct constrained Newton + backtracking. Uses no lift unless the
    optional stopping certificate is requested. Failed cases remain failures.
    """
    w=np.ones(model.s)/model.s if w0 is None else feasible(w0)
    path=[]; start=time.perf_counter(); cert_seconds=0.; value_calls=0
    last_gap=np.inf
    for k in range(max_iter+1):
        f,g,H,P=model.oracle(w,mode,2); Gfw=fw_gap(g,w); G=Gfw
        if trace: path.append(w.copy())
        if stopping!='fw' and Gfw>eps and ('gated' not in stopping or Gfw<=np.sqrt(eps)):
            t=time.perf_counter()
            if stopping.startswith('cheap') and np.min(w)>0: b=1/min(w)
            elif stopping.startswith('vertex'): _,b=model.scales(w)
            else: b=None
            if b is not None:
                low=qp(w,g,H/b,atol=max(eps/20,1e-14))
                G=min(Gfw,max(0.,-low.dual))
            cert_seconds+=time.perf_counter()-t
        if G<=eps:
            return dict(w=w,f=f,gap=G,gfw=Gfw,nit=k,seconds=time.perf_counter()-start,
                        cert_seconds=cert_seconds,converged=True,path=path,value_calls=value_calls)
        if k==max_iter: break
        step=qp(w,g,H,atol=max(1e-14,eps/20)).z-w
        descent=float((g-g.mean())@step)
        if descent>=0 or np.max(abs(step))<1e-16: break
        alpha=1.
        accepted=False
        for _ in range(45):
            trial=feasible(w+alpha*step); fn=model.oracle(trial,mode,0); value_calls+=1
            if fn<=f+1e-4*alpha*descent+2e-15*max(1,abs(f)):
                accepted=True; break
            alpha*=.5
        if not accepted: break
        if np.max(abs(trial-w))<2e-16: break
        w=trial; last_gap=G
    f,g=model.oracle(w,mode,1)
    return dict(w=w,f=f,gap=fw_gap(g,w),gfw=fw_gap(g,w),nit=k,seconds=time.perf_counter()-start,
                cert_seconds=cert_seconds,converged=False,path=path,value_calls=value_calls)


def reference(model,mode):
    a=slsqp(model,mode,ftol=1e-14)
    b=newton(model,mode,w0=a['w'],eps=2e-11,max_iter=100)
    if b['gfw']>2e-9:
        alt=slsqp(model,mode,w0=b['w'],ftol=1e-15,maxiter=1000)
        b2=newton(model,mode,w0=alt['w'],eps=2e-11,max_iter=150)
        if b2['gfw']<b['gfw']: b=b2
    if b['gfw']>2e-12:
        F=np.flatnonzero(b['w']>1e-8)
        if len(F)>1:
            def unpack(y):
                w=np.zeros(model.s); w[F[1:]]=y; w[F[0]]=1-np.sum(y)
                return w
            def residual(y):
                _,gg,HH,_=model.oracle(unpack(y),mode,2)
                rr=gg[F[1:]]-gg[F[0]]
                JJ=HH[np.ix_(F[1:],F[1:])]-HH[F[1:],F[0],None]-HH[F[0],F[1:]][None,:]+HH[F[0],F[0]]
                return rr,JJ
            try:
                rr=root(lambda y:residual(y)[0],b['w'][F[1:]],jac=lambda y:residual(y)[1],tol=1e-11)
                ww=unpack(rr.x)
                ff,gg=model.oracle(ww,mode,1)
                ggfw=fw_gap(gg,ww)
                if np.min(ww)>=0 and ggfw<b['gfw']:
                    b.update(w=ww,f=ff,gfw=ggfw,gap=ggfw,converged=ggfw<=2e-11)
            except (ValueError,np.linalg.LinAlgError): pass
    return b


def model_solver(model,mode,eps=1e-7,max_iter=200,w0=None,trace=False):
    w=np.ones(model.s)/model.s if w0 is None else feasible(w0)
    start=time.perf_counter(); path=[]; total_scale=0.; total_qp=0.; G=np.inf
    for k in range(max_iter+1):
        f,g,H,P=model.oracle(w,mode,2)
        st=time.perf_counter(); a,b=model.scales(w,both=True); total_scale+=time.perf_counter()-st
        st=time.perf_counter(); low=qp(w,g,H/b,atol=eps/8); total_qp+=time.perf_counter()-st
        G=min(fw_gap(g,w),max(0.,-low.dual))
        if trace: path.append(dict(k=k,f=f,gap=G,fw=fw_gap(g,w),a=a,b=b,w=w.tolist()))
        if G<=eps: break
        if k==max_iter: break
        st=time.perf_counter(); upper=qp(w,g,H/a,atol=min(eps/8,1e-12)); total_qp+=time.perf_counter()-st
        if upper.value>=0 or upper.gap>.5*(-upper.value):
            return dict(w=w,f=f,gap=G,nit=k,converged=False,reason='upper_relative_test_failed',
                        seconds=time.perf_counter()-start,path=path,scale_seconds=total_scale,qp_seconds=total_qp)
        w=upper.z
    return dict(w=w,f=f,gap=G,nit=k,converged=bool(G<=eps),reason='gap_met' if G<=eps else 'outer_limit',
                seconds=time.perf_counter()-start,path=path,scale_seconds=total_scale,qp_seconds=total_qp)


def from_covariances(As,Bs,meta=None):
    """Full common branch preprocessing for SPD benchmark inputs only.
    Known-rank singular stress tests use explicit factors, not PSD clipping.
    """
    start=time.perf_counter(); Zs=[]; aas=[]
    for A,B in zip(As,Bs):
        a,Z=eigh(sym(A),sym(A+B),check_finite=False)
        if np.min(a)<=0 or np.max(a)>=1:
            raise ArithmeticError('SPD preprocessing outside (0,1); no rank repair')
        Zs.append(Z); aas.append(a)
    obj=SCI(Zs,aas,meta); obj.common_preprocessing_seconds=time.perf_counter()-start
    return obj
