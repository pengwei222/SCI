"""Float64 SCI oracles and stopping rules. Numerical bounds, not interval proofs."""
from __future__ import annotations
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular, eigh
from scipy.optimize import minimize


def sym(M): return (M + M.T) * .5

def inv(M):
    return sym(cho_solve(cho_factor(sym(M), lower=True, check_finite=False),
                         np.eye(len(M)), check_finite=False))

def simplex(w):
    w = np.maximum(np.asarray(w, dtype=float), 0.)
    if not np.isfinite(w).all() or w.sum() <= 0: raise ValueError('Invalid weights')
    return w / w.sum()


def qp(c, g, Q):
    """Any feasible z gives a valid supporting-plane lower bound on min q.
    Inner optimizer success is deliberately NOT needed to issue this bound.
    """
    def fun(z):
        v = z-c
        return float(g@v + .5*v@Q@v)
    def jac(z): return g+Q@(z-c)
    res = minimize(fun, c, jac=jac, method='SLSQP', bounds=[(0,1)]*len(c),
                   constraints={'type':'eq','fun':lambda z:z.sum()-1,
                                'jac':lambda z:np.ones_like(z)},
                   options={'ftol':1e-13, 'maxiter':100})
    z = simplex(res.x) if np.isfinite(res.x).all() else c.copy()
    if fun(z) > 0: z = c.copy()
    r = jac(z)
    inner_gap = max(0., float(r@z-r.min()))
    dual = fun(z)-inner_gap
    return z, float(dual), inner_gap


class SCI:
    """This experiment has A_i positive definite, B_i semidefinite.
    General singular A is outside this implementation's scope.
    Whitening A gives J_i(t)=Z_i diag(t/(1+t beta_i)) Z_i^T.
    """
    def __init__(self, A, B, mode='trace', heading_length=1., pose_dimension=None):
        self.A, self.B = np.asarray(A), np.asarray(B)
        self.s, self.d, _ = self.A.shape
        self.mode = mode
        self.W = np.eye(self.d)
        self.W[np.arange(2,self.d,3), np.arange(2,self.d,3)] = heading_length**2
        if pose_dimension is not None:
            # Nuisance bias states are fused, but only physical pose covariance
            # enters the trace objective. Zero nuisance weights are intentional.
            if mode!='trace': raise ValueError('Pose-only cost supports trace objective')
            self.W=np.zeros((self.d,self.d))
            self.W[:pose_dimension,:pose_dimension]=np.eye(pose_dimension)
            idx=np.arange(2,pose_dimension,3);self.W[idx,idx]=heading_length**2
        self.Z, self.beta, self.roundoff_adjustment = [], [], 0.
        for a,b in zip(A,B):
            L = np.linalg.cholesky(sym(a))
            Li = solve_triangular(L, np.eye(self.d), lower=True, check_finite=False)
            ev,U = eigh(sym(Li@b@Li.T), check_finite=False)
            scale = max(1., float(abs(ev).max()))
            if ev.min() < -1e-10*scale: raise ArithmeticError('B is not PSD')
            self.roundoff_adjustment = max(self.roundoff_adjustment, max(0.,-float(ev.min()))/scale)
            self.beta.append(np.maximum(ev,0.))
            self.Z.append(Li.T@U)
        self.beta, self.Z = np.array(self.beta), np.array(self.Z)
        self.scale = 1.
        raw = self.oracle(np.ones(self.s)/self.s, hessian=False)[0]
        self.scale = raw if mode == 'trace' else float(self.d)
        if self.scale <= 0: raise ArithmeticError('Invalid objective scale')
        self._vertices = None

    def information(self,w):
        den = 1+np.asarray(w)[:,None]*self.beta
        Ji = np.einsum('sik,sk,sjk->sij',self.Z,np.asarray(w)[:,None]/den,self.Z)
        return Ji

    def oracle(self,w,hessian=True):
        w = np.asarray(w)
        den = 1+w[:,None]*self.beta
        Ji = np.einsum('sik,sk,sjk->sij',self.Z,w[:,None]/den,self.Z)
        D = np.einsum('sik,sk,sjk->sij',self.Z,1/den**2,self.Z)
        P = inv(Ji.sum(axis=0))
        if self.mode == 'trace':
            f = np.trace(self.W@P)
            G = P@self.W@P
            g = -np.einsum('ij,sji->s',G,D)
        else:
            sign,ld = np.linalg.slogdet(P)
            if sign <= 0: raise ArithmeticError('Invalid covariance')
            f = ld
            g = -np.einsum('ij,sji->s',P,D)
        H = None
        if hessian:
            F = np.einsum('sik,sk,sjk->sij',self.Z,-2*self.beta/den**3,self.Z)
            LP = np.linalg.cholesky(P)
            if self.mode == 'trace':
                Y = np.array([LP.T@di@P@np.diag(np.sqrt(np.diag(self.W))) for di in D])
                Y = Y.reshape(self.s,-1)
                H = 2*Y@Y.T-np.diag(np.einsum('ij,sji->s',G,F))
            else:
                Y = np.array([LP.T@di@LP for di in D]).reshape(self.s,-1)
                H = Y@Y.T-np.diag(np.einsum('ij,sji->s',P,F))
            H = sym(H)/self.scale
        return float(f/self.scale), g/self.scale, H, P

    def lift(self,w):
        # Stable Taylor-at-1 lift; no inverse of B or tiny 1/beta factors.
        # t/(1+bt)=(b+t)/(1+b)^2 - b(t-1)^2/[(1+b)^2(1+bt)].
        n = (self.s+1)*self.d
        M = np.zeros((n,n))
        for i,(z,b,t) in enumerate(zip(self.Z,self.beta,w)):
            sl = slice((i+1)*self.d,(i+2)*self.d)
            M[:self.d,:self.d] += (z*((b+t)/(1+b)**2))@z.T
            C = z*(np.sqrt(b)/(1+b)*(t-1))
            M[:self.d,sl],M[sl,:self.d] = C,C.T
            M[sl,sl] = np.diag(1+b*t)
        return sym(M)

    def vertex_b(self,w):
        if self._vertices is None:
            self._vertices = [self.lift(e) for e in np.eye(self.s)]
        M = self.lift(w)
        # Diagonal congruence improves scaling without altering eigenvalues.
        s = 1/np.sqrt(np.diag(M))
        M = s[:,None]*M*s[None,:]
        b = 1.
        for V in self._vertices:
            ev = eigh(s[:,None]*V*s[None,:], M, eigvals_only=True,
                      subset_by_index=[len(M)-1,len(M)-1], check_finite=False)[0]
            b = max(b,float(ev))
        if not np.isfinite(b): raise ArithmeticError('Nonfinite vertex bound')
        return b*(1+1e-9)  # Engineering margin, not outward-rounded verification.

    def fuse(self,w,means):
        P = inv(self.information(w).sum(axis=0))
        x = P@np.einsum('sij,sj->i',self.information(w),means)
        return x,P


class Stop(Exception): pass


def solve(model, tol=1e-3, rule='fw', maxiter=150, w0=None, history=False):
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
        if rule != 'fw' and f-max(best_lower,lower) > tol:
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
