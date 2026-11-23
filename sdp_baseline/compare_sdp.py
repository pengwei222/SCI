"""Cold-start SDP comparison for the archived SPD trace problems. See README.md."""
import os
for var in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","RAYON_NUM_THREADS"):
    os.environ[var]="1"
import argparse, csv, json, platform, random, sys, time, warnings
from collections import defaultdict
from pathlib import Path
from statistics import median
import cvxpy as cp
import clarabel
import numpy as np
import scipy
import scipy.linalg as la
import sympy as sp
from verify_witnesses import exact_check

clock=time.perf_counter

def dump_csv(path, rows):
    with path.open("w",newline="") as out:
        writer=csv.DictWriter(out, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

def weight(x):
    x=np.maximum(np.asarray(x,float),0.)
    return x / x.sum()

def trace_cov(A,B,w):
    d=A.shape[1]
    J=sum(t*la.solve(a+t*b,np.eye(d),assume_a="pos") for a,b,t in zip(A,B,w))
    return np.trace(la.solve(J,np.eye(d),assume_a="pos"))/d

def inverse(a):
    v=la.solve(a,np.eye(len(a)),assume_a="pos")
    return (v+v.T)/2

def setup(A,B):
    s,d=A.shape[:2]
    n=(s+1)*d
    IA=[inverse(a) for a in A]
    IB=[inverse(b) for b in B]
    M0=np.zeros((n,n))
    D=[]
    for i,(a,b) in enumerate(zip(IA,IB)):
        sl=slice((i+1)*d,(i+2)*d)
        M0[:d,:d]+=b
        M0[sl,:d]=M0[:d,sl]=b
        M0[sl,sl]=b
        v=np.zeros_like(M0)
        v[sl,sl]=a
        D.append(v)
    E=np.eye(n)[:,:d]
    L=la.cholesky(M0+sum(D)/s,lower=True)
    C=la.solve_triangular(L,np.eye(n),lower=True)
    V=[C@(M0+v)@C.T for v in D]
    V=[(x+x.T)/2 for x in V]
    U=C@E
    w=cp.Variable(s)
    X=cp.Variable((d,d),symmetric=True)
    m=sum(w[i]*V[i] for i in range(s))
    z=cp.bmat([[m,U],[U.T,X]])
    psd=(z >> 0)
    problem=cp.Problem(cp.Minimize(cp.trace(X)/d),[w>=0,cp.sum(w)==1,psd])
    return problem,psd,w,X,V,C,E,U

def cold_sdp(A,B,c):
    t0=clock()
    problem,constraint,w,X,V,C,E,U=setup(A,B)
    t1=clock()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",category=UserWarning)
        problem.solve(solver="CLARABEL",warm_start=False,verbose=False,
            tol_gap_abs=1e-11,tol_gap_rel=1e-11,tol_feas=1e-11,
            max_iter=500,max_threads=1)
    t2=clock()
    if constraint.dual_value is None or w.value is None:
        raise RuntimeError(f"Solver returned no primal/dual values: {problem.status}")
    d=A.shape[1]; n=len(C)
    Y=constraint.dual_value
    sol=weight(w.value)
    Vw=sum(t*v for t,v in zip(sol,V))
    # If U=C E, the original-coordinate factor is C.T @ K_white.
    dualK=C.T@(-d*Y[:n,n:])
    primalK=C.T@la.solve(Vw,U,assume_a="pos")
    def lower(K):
        kw=la.solve_triangular(C.T,K,lower=False)
        return (2*np.sum(K*E)-max(np.sum(kw*(v@kw)) for v in V))/d
    ldual, lprimal=map(lower,(dualK,primalK))
    fc=trace_cov(A,B,c)
    fsol=trace_cov(A,B,sol)
    t3=clock()
    Vraw=sum(t*v for t,v in zip(w.value,V))
    slack=np.block([[Vraw,U],[U.T,X.value]])
    return dict(model_s=t1-t0,cvx_s=t2-t1,solver_internal_s=problem.solver_stats.solve_time,
        dual_recovery_evaluation_s=t3-t2,sdp_total_s=t3-t0,
        sdp_solver_status=problem.status,sdp_solver_iterations=problem.solver_stats.num_iters,
        raw_primal_psd_min=la.eigvalsh(slack)[0],raw_dual_psd_min=la.eigvalsh(Y)[0],
        raw_dual_bottom_stationarity_inf=la.norm(Y[n:,n:]-np.eye(d)/d,np.inf),
        raw_weight_sum_error=abs(w.value.sum()-1),raw_weight_min=w.value.min(),
        f_candidate=fc,sdp_feasible_f=fsol,dual_lb_fp=ldual,primal_factor_lb_fp=lprimal,
        gap_sdp_fp=fc-max(ldual,lprimal)),dualK,primalK,sol

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--limit",type=int,default=0)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    source=args.archive/"synthetic"
    sys.path.insert(0,str(source/"code"))
    from benchmark_core import from_covariances,certificate
    read=lambda file:list(csv.DictReader((source/"data"/file).open()))
    refs={int(x["index"]):x for x in read("references.csv") if x["mode"]=="trace"}
    records=[r for r in read("certificate_results.csv") if r["mode"]=="trace"
        and float(r["eta"])==.001 and int(r["d"]) in (2,4) and int(r["s"]) in (3,6)]
    assert len(records)==32
    if args.limit:records=records[:args.limit]
    # Warm up library dispatch outside all clocks with an unrelated 1D input.
    dummy=np.array([[[1.]],[[2.]]])
    cold_sdp(dummy,dummy,np.array([.5,.5]))
    for rule in ("cheap","vertex"):
        obj=from_covariances(dummy,dummy)
        if rule=="vertex":obj.build_lift()
        certificate(obj,np.array([.5,.5]),"trace",bkind=rule,tol=1e-13)
    allrows=[]; raw=[]; witness=[]
    for rec in records:
        idx=int(rec["index"])
        with np.load(source/"data"/f"instance_{idx:03d}.npz") as z:
            A,B=z["A"].copy(),z["B"].copy()
        c=weight(json.loads(rec["weights"]))
        ref=np.asarray(json.loads(refs[idx]["wstar"]))
        series=defaultdict(list); status={}
        jobs=[(rule,j) for rule in ("cheap","vertex","sdp") for j in range(args.repeats)]
        random.Random(4890481+idx).shuffle(jobs)
        for rule,rep in jobs:
            if rule=="sdp":
                st,K1,K2,sw=cold_sdp(A,B,c)
                timing={k:st[k] for k in ("model_s","cvx_s","solver_internal_s","dual_recovery_evaluation_s","sdp_total_s")}
                if rep==0:d0,dual,primal,wp=st,K1,K2,sw
                total=st["sdp_total_s"]
            else:
                t0=clock()
                model=from_covariances(A,B)
                if rule=="vertex":model.build_lift()
                cert=certificate(model,c,"trace",bkind=rule,tol=1e-13)
                total=clock()-t0
                timing={rule+"_total_s":total,rule+"_post_setup_s":cert["total_seconds"]}
                if rep==0:
                    status[rule+"_gap"]=cert["gq"]
                    status["fw_gap"]=cert["gfw"]
            for k,v in timing.items():series[k].append(v)
            raw.append(dict(case=idx,method=rule,repeat=rep,total_s=total,
                            status=st["sdp_solver_status"] if rule=="sdp" else cert["qp_status"]))
        exact=exact_check(A,B,c,dual,primal,ref)
        row=dict(case=idx,d=rec["d"],s=rec["s"],condition=rec["cond"],kind=rec["kind"],
            ref_interior=rec["interior"],**d0,**status,**exact)
        for k,vals in series.items():row[k]=median(vals)
        row["fw_to_sdp"]=row["fw_gap"]/row["sdp_gap_ub"]
        row["vertex_to_sdp"]=row["vertex_gap"]/row["sdp_gap_ub"]
        row["cheap_to_sdp"]=row["cheap_gap"]/row["sdp_gap_ub"]
        row['sdp_to_error_upper_denominator']=row["sdp_gap_ub"]/row["exact_reference_error_ub"]
        row['sdp_to_error_lower_denominator']=row["sdp_gap_ub"]/row["exact_reference_error_lb"]
        witness.append(dict(case=idx,c=json.dumps(c.tolist()),solver_weight=json.dumps(wp.tolist()),
            dual_factor=json.dumps(dual.tolist()),primal_factor=json.dumps(primal.tolist())))
        allrows.append(row)
        print(f"case {idx} {row['sdp_solver_status']} SDP/certified-error upper ≈ {row['sdp_to_error_upper_denominator']:.4g}",flush=True)
        dump_csv(args.output/"cases.csv",allrows)
        dump_csv(args.output/"cold_times.csv",raw)
        dump_csv(args.output/"witnesses.csv",witness)
    summaries=[]
    for label,rows in [('all',allrows),('interior',[r for r in allrows if r['ref_interior']=='True']),
        ('boundary',[r for r in allrows if r['ref_interior']=='False'])]:
        if not rows:continue
        s=dict(group=label,n=len(rows))
        for col in ('fw_to_sdp','vertex_to_sdp','cheap_to_sdp','sdp_to_error_upper_denominator',
            'sdp_to_error_lower_denominator','sdp_total_s','model_s','cvx_s','solver_internal_s',
            'dual_recovery_evaluation_s','exact_factor_verification_s','cheap_total_s','vertex_total_s','cheap_post_setup_s',
            'vertex_post_setup_s'):
            s['median_'+col]=median(r[col] for r in rows)
        s['paired_sdp_time_over_cheap']=median(r['sdp_total_s']/r['cheap_total_s'] for r in rows)
        s['paired_sdp_time_over_vertex']=median(r['sdp_total_s']/r['vertex_total_s'] for r in rows)
        s['selected_dual']=sum(r['selected_factor']=='solver_dual' for r in rows)
        summaries.append(s)
    dump_csv(args.output/"summary.csv",summaries)
    (args.output/"environment.json").write_text(json.dumps(dict(python=platform.python_version(),machine=platform.machine(),
        system=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,cvxpy=cp.__version__,
        clarabel=clarabel.__version__,sympy=sp.__version__,repeats=args.repeats),indent=2))
if __name__=="__main__":main()
