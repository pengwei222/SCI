"""Reference inexact quadratic-model solver for multisensor SCI.

Floating-point outputs are explicitly NOT exact certificates. Use
sci_exact.RationalSCI.certificate for independent exact a posteriori checking.
This implementation prioritizes traceability, not timing performance.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.optimize import minimize
from sci_exact import RationalSCI, psd_factor, quantize_simplex, rational

Array = NDArray[np.float64]


def sym(a: Array) -> Array:
    return (a+a.T)/2


def arr(a) -> Array:
    return np.asarray(a,dtype=float)


def inv_spd(a: Array) -> Array:
    return sym(cho_solve(cho_factor(sym(a),lower=True,check_finite=True),np.eye(len(a))))


def feasible(w: Array) -> Array:
    w=np.maximum(np.asarray(w,dtype=float).reshape(-1),0.0)
    if not np.isfinite(w).all() or w.sum()<=0:
        raise ValueError('Invalid weights.')
    return w/w.sum()


class NumericSCI:
    """Uses exact rank structure from rational input preprocessing.

    It evaluates J_i(t)=J0+t L(Ared+t Cred)^(-1)L.T, including t=0.
    No small positive eigenvalue is reclassified as zero.
    """
    def __init__(self, exact: RationalSCI):
        self.exact=exact; self.d=exact.d; self.s=exact.s; self.n=exact.n
        self.W=arr(exact.W)
        U,D=psd_factor(exact.W)
        self.Wroot=arr(U)*np.sqrt(np.array([float(D[i,i]) for i in range(D.rows)]))
        self.branches=[tuple(arr(x) for x in (q.J0,q.L,q.Ared,q.Cred)) for q in exact.branches]
        self.M0=arr(exact.M0); self.Mi=np.array([arr(x) for x in exact.Mi])
        self.vertices=[arr(v) for v in exact.vertices]

    def info(self,w: Array):
        w=np.asarray(w,dtype=float).reshape(-1)
        if len(w)!=self.s or np.min(w)<-1e-14 or abs(w.sum()-1)>1e-12:
            raise ValueError('Weight outside simplex.')
        J=np.zeros((self.d,self.d)); Ds=[]; Fs=[]
        for t,(J0,L,A,C) in zip(w,self.branches):
            if A.shape[0]==0:
                D=F=np.zeros_like(J0); Ji=J0
            else:
                R=inv_spd(A+t*C)
                Ji=J0+t*L@R@L.T
                D=L@(R-t*R@C@R)@L.T
                F=L@(-2*R@C@R+2*t*R@C@R@C@R)@L.T
            J+=Ji; Ds.append(sym(D)); Fs.append(sym(F))
        return sym(J),Ds,Fs

    def oracle(self,w: Array,mode: str='logdet'):
        J,D,F=self.info(w); P=inv_spd(J)
        grad=np.zeros(self.s); H=np.zeros((self.s,self.s))
        LP=np.linalg.cholesky(P)
        if mode=='logdet':
            LJ=cho_factor(J,lower=True)[0]
            val=-2*np.log(np.diag(LJ)).sum()
            K=np.array([LP.T@Di@LP for Di in D])
            grad=-np.trace(K,axis1=1,axis2=2)
            KF=K.reshape(self.s,-1)
            H=KF@KF.T
            for i in range(self.s):
                H[i,i]-=np.trace(P@F[i])
        elif mode=='trace':
            val=np.trace(self.W@P)
            G=P@self.Wroot
            Y=np.array([LP.T@Di@G for Di in D])
            YF=Y.reshape(self.s,-1)
            H=2*YF@YF.T
            for i in range(self.s):
                grad[i]=-np.trace(G.T@D[i]@G)
                H[i,i]-=np.trace(G.T@F[i]@G)
        else:
            raise ValueError('Unknown objective.')
        return float(val),grad,sym(H),P

    def lift(self,w: Array):
        return sym(self.M0+np.einsum('i,ijk->jk',w,self.Mi))

    def scales(self,c: Array) -> tuple[float,float]:
        M=self.lift(c)
        lo,hi=1.0,1.0
        for V in self.vertices:
            ev=eigh(V,M,eigvals_only=True,check_finite=True)
            lo=min(lo,float(ev[0])); hi=max(hi,float(ev[-1]))
        if not 0<lo<=hi or not np.isfinite(hi):
            raise ArithmeticError('Unable to compute positive lift scales.')
        # Engineering margin only. This is NOT a verified rounding enclosure.
        return lo/(1+1e-9),hi*(1+1e-9)


@dataclass
class QPResult:
    point: Array
    value: float
    dual_lower: float
    dual_gap: float
    iterations: int
    status: str


def simplex_qp(c: Array,g: Array,Q: Array,*,atol: float|None=None,
               relative_decrease: float|None=None,max_iter: int=50000,
               warm_start: bool=True) -> QPResult:
    """q(z)=g.T(z-c)+0.5(z-c).T Q(z-c), with Q PSD.

    Inner acceptance uses the feasible-point simplex dual gap, never just
    an external solver's status. SLSQP supplies a candidate; a conditional
    gradient fallback uses exact quadratic line search. Either atol or the
    relative criterion delta <= rho*(-q) must be supplied.
    """
    c=feasible(c); g=np.asarray(g); Q=sym(np.asarray(Q))
    if (atol is None)==(relative_decrease is None):
        raise ValueError('Specify exactly one inner stopping rule.')
    if atol is not None and atol<=0:
        raise ValueError('atol must be positive.')
    if relative_decrease is not None and relative_decrease<=0:
        raise ValueError('rho must be positive.')
    def q(z):
        v=z-c
        return float(g@v+.5*v@Q@v)
    def jac(z):return g+Q@(z-c)
    z=c.copy()
    if warm_start:
        res=minimize(q,c,jac=jac,method='SLSQP',bounds=[(0,1)]*len(c),
                     constraints={'type':'eq','fun':lambda x:x.sum()-1,
                                  'jac':lambda x:np.ones_like(x)},
                     options={'ftol':1e-13,'maxiter':500,'disp':False})
        cand=feasible(res.x)
        if np.isfinite(q(cand)) and q(cand)<=0:
            z=cand
    for it in range(max_iter+1):
        r=jac(z); j=int(np.argmin(r)); value=q(z)
        raw_gap=float(r@z-r[j]); gap=max(0.0,raw_gap)
        if atol is not None:
            accepted=gap<=atol
        else:
            accepted=value<0 and gap<=relative_decrease*(-value)
            if gap==0 and value==0:
                accepted=True
        if accepted:
            return QPResult(z.copy(),value,value-gap,gap,it,'numerical_gap_met')
        if it==max_iter:
            break
        d=-z.copy(); d[j]+=1
        curvature=float(d@Q@d)
        if curvature < -1e-10*max(1.0,np.linalg.norm(Q,2)):
            raise ArithmeticError('QP matrix has negative curvature.')
        step=1.0 if curvature<=0 else min(1.0,gap/curvature)
        if step<=0 or np.linalg.norm(step*d,np.inf)<1e-17:
            break
        z=feasible(z+step*d)
    raise ArithmeticError(f'Inner gap not reached: gap={gap:.3g}, q={value:.3g}; no certificate issued.')


@dataclass
class SolveResult:
    weights: Array
    lower_anchor: Array
    history: list[dict]
    numerical_gap: float
    converged: bool
    certificate_status: str='floating-point estimate; requires exact verification'


def solve(model: NumericSCI,mode: str='logdet',eps: float=1e-7,
          rho: float=.5,max_outer: int=3000,w0: Array|None=None,
          warm_start: bool=True) -> SolveResult:
    if eps<=0 or rho<=0:
        raise ValueError('Positive tolerances are required.')
    w=np.ones(model.s)/model.s if w0 is None else feasible(w0)
    hist=[]; best_lower=-np.inf; lower=None
    for k in range(max_outer+1):
        f,g,H,P=model.oracle(w,mode); a,b=model.scales(w)
        # No PSD clipping is performed. Ranks are not changed to aid a solve.
        lower=simplex_qp(w,g,H/b,atol=eps/4,warm_start=warm_start)
        first_dual=float(np.min(g)-g@w)
        dual=max(lower.dual_lower,first_dual)
        best_lower=max(best_lower,f+dual)
        gap=max(0.0,f-best_lower)
        row={'k':k,'objective':f,'numerical_gap':gap,
             'first_order_gap':-first_dual,'a':a,'b':b,
             'lower_qp_gap':lower.dual_gap,'weights':w.tolist()}
        hist.append(row)
        if gap<=eps:
            return SolveResult(w,lower.point,hist,gap,True)
        if k==max_outer:break
        up=simplex_qp(w,g,H/a,relative_decrease=rho,warm_start=warm_start)
        fn=model.oracle(up.point,mode)[0]
        if fn>f+up.value+1e-9*max(1,abs(f)):
            raise ArithmeticError('Computed upper model failed direct value check.')
        row.update({'upper_qp_gap':up.dual_gap,'predicted_decrease':-up.value,
                    'next_objective':fn,'upper_inner_steps':up.iterations})
        w=up.point
    return SolveResult(w,lower.point,hist,gap,False)


def verified_solve(exact: RationalSCI,mode: str='logdet',eps='1/1000000',
                   max_attempts: int=5) -> dict:
    """Refine a floating candidate until an exact posterior bound meets eps.

    This is a correctness wrapper, not a fast numerical routine. A failure to
    meet the target is returned explicitly; it never becomes a success flag
    merely because the floating solver declared convergence.
    """
    target=rational(eps)
    if target<=0:raise ValueError('eps must be positive')
    if max_attempts<1:raise ValueError('max_attempts must be at least 1')
    numerical=NumericSCI(exact); attempts=[]; w0=None
    for k in range(max_attempts):
        out=solve(numerical,mode,eps=float(target)/(4**(k+1)),w0=w0)
        w0=out.weights
        c=quantize_simplex(out.weights,bits=40+8*k)
        z=quantize_simplex(out.lower_anchor,bits=40+8*k)
        cert=exact.certificate(c,z,mode,b_proposal=str(out.history[-1]['b']))
        gap=rational(cert['gap']['numerator'])/rational(cert['gap']['denominator'])
        attempts.append({'attempt':k+1,'floating_converged':out.converged,
                         'floating_gap':out.numerical_gap,
                         'verified_gap_upper':cert['gap_upper_decimal']})
        if gap<=target:
            return {'certified':True,'target':str(target),'certificate':cert,
                    'attempts':attempts,'arithmetic':'rational posterior verification'}
    return {'certified':False,'target':str(target),'certificate':cert,
            'attempts':attempts,'reason':'exact posterior target not met'}
