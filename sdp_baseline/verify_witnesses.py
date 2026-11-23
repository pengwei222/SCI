"""Recheck the saved SDP witnesses, independent of every optimizer run and status."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import sympy as sp
from time import perf_counter as clock

def rational(x):
    return sp.Rational(*float(x).as_integer_ratio())

def rmat(a):
    return sp.Matrix([[rational(x) for x in row] for row in a])

def inner(a,b):
    return sum(x*y for x,y in zip(a,b))

def exact_check(A,B,c,K1,K2,wr):
    """Use exact binary input covariances, not the whitened numerical lift."""
    t0=clock()
    A=list(map(rmat,A)); B=list(map(rmat,B))
    d=A[0].rows
    for m in A+B:
        if not m.is_symmetric() or any(m[:i,:i].det()<=0 for i in range(1,d+1)):
            raise ValueError("Binary input is not positive definite")
    IA=[m.inv() for m in A]; IB=[m.inv() for m in B]
    def cv(w,ref=False):
        ww=list(map(rational,w)); ww=[x/sum(ww) for x in ww]
        mats=[(a+t*b).inv() for a,b,t in zip(A,B,ww)]
        J=sum((t*h for t,h in zip(ww,mats)),sp.zeros(d))
        P=J.inv()
        f=sp.trace(P)/d
        if not ref:return f
        g=[-sp.trace(P*h*a*h*P)/d for a,h in zip(A,mats)]
        fw=sum(x*y for x,y in zip(ww,g))-min(g)
        return f,fw
    def lower(K):
        K=rmat(K); top=K[:d,:]
        children=[K[(i+1)*d:(i+2)*d,:] for i in range(len(A))]
        base=sum(inner(top+k,b*(top+k)) for b,k in zip(IB,children))
        worst=max(inner(k,a*k) for k,a in zip(children,IA))
        return (2*sp.trace(top)-base-worst)/d
    fc=cv(c)
    l1,l2=lower(K1),lower(K2)
    best=max(l1,l2)
    # Report floats widened in the conservative direction by one binary64 ulp.
    def outward(x,direction):
        rounded=float(np.nextafter(float(x),np.inf if direction>0 else -np.inf))
        assert direction*(rational(rounded)-x)>=0
        return rounded
    up=lambda x:outward(x,1)
    dn=lambda x:outward(x,-1)
    online=clock()-t0
    fref,fwref=cv(wr,True)
    assert best<=fref and fc>=best
    return dict(dual_gap_ub=up(fc-l1),primal_factor_gap_ub=up(fc-l2),
        sdp_gap_ub=up(fc-best),sdp_fstar_lb=dn(best),candidate_f_ub=up(fc),
        exact_reference_error_lb=dn(max(0,fc-fref)),
        exact_reference_error_ub=up(fc-fref+fwref),
        exact_factor_verification_s=online,
        selected_factor="solver_dual" if l1>=l2 else "primal_recovery")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive",required=True,type=Path)
    p.add_argument("--results",required=True,type=Path)
    args=p.parse_args()
    def rows(path):return list(csv.DictReader(path.open()))
    refs={int(r['index']):r for r in rows(args.archive/'synthetic/data/references.csv') if r['mode']=='trace'}
    cases={int(r['case']):r for r in rows(args.results/'cases.csv')}
    ws=rows(args.results/'witnesses.csv')
    assert set(cases)=={int(w['case']) for w in ws}
    for w in ws:
        i=int(w['case'])
        with np.load(args.archive/'synthetic/data'/f'instance_{i:03d}.npz') as z:
            inp=(z['A'].copy(),z['B'].copy())
        vals=[np.array(json.loads(w[k])) for k in ('c','dual_factor','primal_factor')]
        checked=exact_check(*inp,*vals,np.array(json.loads(refs[i]['wstar'])))
        for key in ('dual_gap_ub','primal_factor_gap_ub','sdp_gap_ub','sdp_fstar_lb','candidate_f_ub',
                    'exact_reference_error_lb','exact_reference_error_ub'):
            assert checked[key]==float(cases[i][key]),(i,key)
        assert checked['selected_factor']==cases[i]['selected_factor']
    print(f"Exactly reproduced all seven directed numerical bounds and selected factor for {len(cases)} saved witnesses.")

if __name__=="__main__":
    main()
