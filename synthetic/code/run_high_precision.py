"""Independent 80-digit computation from raw A,B, not the spectral oracle."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import pandas as pd,numpy as np,json,time
import mpmath as mp
mp.mp.dps=80
OUT=Path(__file__).resolve().parents[1]/'data'
refs=pd.read_csv(OUT/'references.csv'); tests=pd.read_csv(OUT/'certificate_results.csv')
rows=[]; start=time.perf_counter()

def mpmatrix(X): return mp.matrix([[mp.mpf(float(a)) for a in row] for row in X])
def tr(X):return mp.fsum(X[j,j] for j in range(X.rows))
for idx in [0,2,28,30,56,62]:
 raw=np.load(OUT/f'instance_{idx:03d}.npz')
 As=[mpmatrix(A) for A in raw['A']]; Bs=[mpmatrix(B) for B in raw['B']]
 s=len(As);d=As[0].rows
 for mode in ['trace','logdet']:
  rr=refs[(refs['index']==idx)&(refs['mode']==mode)].iloc[0]
  wr=np.array(json.loads(rr.wstar));active=np.flatnonzero(wr>1e-7).tolist()
  cache={}
  def oracle(w,hessian=False):
   Ds=[]; Fs=[]; J=mp.zeros(d)
   for A,B,t in zip(As,Bs,w):
    R=(A+t*B)**-1;J+=t*R;DD=R*A*R;Ds.append(DD)
    if hessian:Fs.append(-R*B*DD-DD*B*R)
   P=J**-1
   f=tr(P)/d if mode=='trace' else -mp.log(mp.det(J))/d
   if mode=='trace':gg=mp.matrix([-tr(P*D*P)/d for D in Ds])
   else:gg=mp.matrix([-tr(P*D)/d for D in Ds])
   if not hessian:return f,gg
   H=mp.zeros(s)
   for i in range(s):
    for j in range(s):
     if mode=='trace':
      z=(tr(P*Ds[i]*P*Ds[j]*P)+tr(P*Ds[j]*P*Ds[i]*P))/d
      if i==j:z-=tr(P*Fs[i]*P)/d
     else:
      z=tr(P*Ds[i]*P*Ds[j])/d
      if i==j:z-=tr(P*Fs[i])/d
     H[i,j]=z
   return f,gg,H
  def unpack(y):
   w=[mp.mpf(0)]*s
   for j,x in zip(active[1:],y):w[j]=x
   w[active[0]]=1-mp.fsum(y);return w
  def residual(*y):
   f,g,H=oracle(unpack(y),True)
   return mp.matrix([g[j]-g[active[0]] for j in active[1:]])
  def jac(*y):
   f,g,H=oracle(unpack(y),True);a=active[0]
   return mp.matrix([[H[i,j]-H[i,a]-H[a,j]+H[a,a] for j in active[1:]] for i in active[1:]])
  if len(active)>1:
   yy=mp.findroot(residual,tuple(mp.mpf(float(wr[i])) for i in active[1:]),J=jac,tol=mp.mpf('1e-60'),maxsteps=25)
   wstar=unpack(list(yy))
  else:wstar=unpack([])
  fstar,gstar=oracle(wstar)
  kkt=mp.fsum(x*y for x,y in zip(gstar,wstar))-min(gstar)
  assert min(wstar)>=0 and kkt<mp.mpf('1e-50'),(idx,mode,kkt)
  for eta in [.1,.001,.0001]:
   test=tests[(tests['index']==idx)&(tests['mode']==mode)&(tests.eta==eta)].iloc[0]
   cw=[mp.mpf(float(x)) for x in json.loads(test.weights)];ss=mp.fsum(cw);cw=[x/ss for x in cw]
   ff,_=oracle(cw); gap=ff-fstar
   q=mp.mpf(float(test.gq)); cheap=mp.mpf(float(test.gq_cheap))
   rows.append(dict(index=idx,d=d,s=s,mode=mode,eta=eta,kkt_gap=mp.nstr(kkt,20),
                    gap_80digits=mp.nstr(gap,60),gap_float=float(gap),gq=test.gq,gq_cheap=test.gq_cheap,
                    vertex_bound_verified_sample=bool(gap<=q+mp.mpf('1e-12')),
                    cheap_bound_verified_sample=bool(gap<=cheap+mp.mpf('1e-12')),
                    floating_reference_difference=float(test.true_gap_estimate)-float(gap)))
  print(idx,mode,'KKT',mp.nstr(kkt,8),flush=True)
pd.DataFrame(rows).to_csv(OUT/'high_precision_results.csv',index=False)
print('done',len(rows),'comparisons',time.perf_counter()-start)
