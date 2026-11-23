"""Engineering variants: same SLSQP candidates; same numerical certificate.
No exact/interval certification is asserted. All QP lower bounds use feasible
supporting planes; even an interrupted/failed active-set search is safe in exact
arithmetic. Gate changes when a certificate is attempted, never its threshold.
"""
import numpy as np
import original_core as original

class CachedSCI(original.SCI):
    def oracle(self,w,hessian=True):
        w=np.asarray(w)
        if hessian and hasattr(self,'_first') and np.array_equal(w,self._first[0]):
            _,den,D,P,G,f,g=self._first
            F=np.einsum('sik,sk,sjk->sij',self.Z,-2*self.beta/den**3,self.Z)
            LP=np.linalg.cholesky(P)
            if self.mode=='trace':
                Y=np.array([LP.T@di@P@np.diag(np.sqrt(np.diag(self.W))) for di in D]).reshape(self.s,-1)
                H=2*Y@Y.T-np.diag(np.einsum('ij,sji->s',G,F))
            else:
                Y=np.array([LP.T@di@LP for di in D]).reshape(self.s,-1)
                H=Y@Y.T-np.diag(np.einsum('ij,sji->s',P,F))
            return float(f/self.scale),g/self.scale,original.sym(H)/self.scale,P
        den=1+w[:,None]*self.beta
        Ji=np.einsum('sik,sk,sjk->sij',self.Z,w[:,None]/den,self.Z)
        D=np.einsum('sik,sk,sjk->sij',self.Z,1/den**2,self.Z)
        P=original.inv(Ji.sum(axis=0))
        if self.mode=='trace':
            f=np.trace(self.W@P);G=P@self.W@P;g=-np.einsum('ij,sji->s',G,D)
        else:
            sign,f=np.linalg.slogdet(P)
            if sign<=0:raise ArithmeticError('Invalid covariance')
            G=None;g=-np.einsum('ij,sji->s',P,D)
        self._first=(w.copy(),den,D,P,G,f,g)
        if hessian:return self.oracle(w,True)
        return float(f/self.scale),g/self.scale,None,P

def active_qp(c,g,Q):
    """Bounded active-set candidate, followed by the ORIGINAL dual-gap bound."""
    z=c.copy();s=len(c);support=set(np.flatnonzero(z>0).tolist());h=Q@c-g
    for _ in range(4*s+4):
        I=np.array(sorted(support),dtype=int)
        K=np.empty((len(I)+1,len(I)+1));K[:-1,:-1]=Q[np.ix_(I,I)]
        K[-1,:-1]=1;K[:-1,-1]=1;K[-1,-1]=0
        try:sol=np.linalg.solve(K,np.r_[h[I],1])[:-1]
        except np.linalg.LinAlgError:break
        target=np.zeros(s);target[I]=sol
        if np.min(sol)<-1e-14:
            d=target-z;bad=np.flatnonzero(d<0)
            alpha=min(1.,float(np.min(-z[bad]/d[bad])))
            z=original.simplex(z+alpha*d)
            drop=[int(i) for i in I if z[i]<1e-14]
            if not drop:break
            support.difference_update(drop)
            if not support:break
        else:
            z=original.simplex(target)
            r=g+Q@(z-c);inactive=[i for i in range(s) if i not in support]
            if not inactive:break
            j=min(inactive,key=lambda j:r[j])
            if r[j]>=float(r@z)-1e-12*max(1.,float(np.max(abs(r)))):break
            support.add(j)
    v=z-c;value=float(g@v+.5*v@Q@v)
    if value>0 or not np.isfinite(value):z=c.copy();v=z-c;value=0.
    r=g+Q@v;gap=max(0.,float(r@z-r.min()))
    return z,value-gap,gap

import time
from scipy.optimize import minimize
simplex=original.simplex
Stop=original.Stop
qp=active_qp

def solve(model, tol=1e-3, rule='fw', maxiter=150, w0=None, history=False, gate_factor=float('inf')):
    """Identical SLSQP candidate generator for all stopping rules.
    Timer includes every oracle, spectral scale, QP and certificate operation.
    Model construction time is recorded separately by the benchmark.
    """
    t0 = time.perf_counter()
    w0 = np.ones(model.s)/model.s if w0 is None else simplex(w0)
    best_lower, latest, seen, hist = -np.inf, None, None, []
    calls, checks, cert_time, failures = 0, 0, 0., 0
    cache_w, cache = None, None
    def evaluate(w):
        nonlocal cache_w,cache,calls
        if cache_w is None or not np.array_equal(w,cache_w):
            cache_w=w.copy();cache=model.oracle(w,False);calls+=1
        return cache
    def check(raw):
        nonlocal best_lower,latest,seen,checks,cert_time,failures
        w = simplex(raw)
        if seen is not None and np.array_equal(w,seen): return
        seen = w.copy(); checks += 1
        f,g,_,P = evaluate(w)
        fw = max(0.,float(g@w-g.min()))
        lower = f-fw
        b,inner_gap,gq = np.nan,np.nan,np.nan
        ct = time.perf_counter()
        if rule != 'fw' and tol < f-max(best_lower,lower) <= gate_factor*tol:
            try:
                if rule == 'cheap':
                    if w.min() <= 0: raise ArithmeticError('Cheap scale needs interior weights')
                    b = 1/float(w.min())
                elif rule == 'vertex': b = model.vertex_b(w)
                else: raise ValueError(rule)
                H = model.oracle(w,True)[2]
                if np.linalg.eigvalsh(H).min() < -1e-8*max(1.,np.linalg.norm(H,2)):
                    raise ArithmeticError('Negative numerical curvature')
                _,dual,inner_gap = qp(w,g,H/b)
                gq = max(0.,-dual)
                lower = max(lower,f-gq)
            except (ArithmeticError,np.linalg.LinAlgError,ValueError):
                failures += 1  # fail closed: retain first-order bound
        cert_time += time.perf_counter()-ct
        best_lower = max(best_lower,lower)
        gap = max(0.,f-best_lower)
        latest = dict(w=w.copy(),f=f,gap=gap,fw=fw,gq=gq,b=b,inner_gap=inner_gap,P=P)
        if history: hist.append({k:v for k,v in latest.items() if k not in ('w','P')})
        if gap <= tol: raise Stop()
    status='maxiter_or_stall'
    try:
        check(w0)
        res=minimize(lambda w:evaluate(w)[0],w0,jac=lambda w:evaluate(w)[1],
                     method='SLSQP',bounds=[(0,1)]*model.s,
                     constraints={'type':'eq','fun':lambda w:w.sum()-1,
                                  'jac':lambda w:np.ones_like(w)},callback=check,
                     options={'maxiter':maxiter,'ftol':1e-15,'disp':False})
        check(res.x)
        status=str(res.message)
    except Stop: status='bound_met'
    latest.update(seconds=time.perf_counter()-t0,certificate_seconds=cert_time,
                  oracle_calls=calls,checks=checks,certificate_failures=failures,
                  converged=latest['gap']<=tol,status=status,history=hist)
    return latest
