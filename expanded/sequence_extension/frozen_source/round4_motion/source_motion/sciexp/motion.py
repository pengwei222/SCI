"""Causal deterministic command response; no truth input and no fitted test data."""
import numpy as np


class ControlResponse:
    def __init__(self, odometry, calibration):
        self.response = calibration.get('motion_response')
        self.times=[]; self.targets=[]; self.states=[]; self.integrals=[]
        self.change_times=[]
        self.scale=np.asarray(calibration['control_scale'])
        self.bias=np.asarray(calibration['control_bias'])
        if self.response is not None:
            if self.response['family']!='stop_gated_first_order':
                raise ValueError('Unsupported motion response family')
            self.tau=float(self.response['tau_seconds'])
            if self.tau<=0 or self.response.get('delay_seconds',0)!=0:
                raise ValueError('Positive tau and zero explicit delay required')
            coef=np.asarray(self.response['coef'])
            if coef.shape!=(5,2,2):raise ValueError('Expected five robot coefficients')
        for i,od in enumerate(odometry):
            # Removing repeated identical values is exact for a held command.
            od=np.asarray(od);keep=np.r_[True,np.any(np.diff(od[:,1:3],axis=0)!=0,axis=1)]
            od=od[keep];tt=od[:,0];u=od[:,1:3]
            if np.any(np.diff(tt)<=0):raise ValueError('Odometry must be sorted and deduplicated')
            if self.response is None:
                g=u*self.scale[i]+self.bias[i]
                z=g.copy()
            else:
                indicator=np.column_stack([abs(u[:,0])>1e-8,
                    np.where(abs(u[:,1])>1e-8,np.sign(u[:,1]),0.)])
                g=u*coef[i,:,0]+indicator*coef[i,:,1]
                z=np.empty_like(g);z[0]=g[0]
            cs=np.zeros_like(g)
            for j,dt in enumerate(np.diff(tt)):
                if self.response is None:inc=g[j]*dt
                else:
                    one=-np.expm1(-dt/self.tau)
                    inc=g[j]*dt+(z[j]-g[j])*self.tau*one
                    z[j+1]=g[j]+(z[j]-g[j])*(1-one)
                cs[j+1]=cs[j]+inc
            self.times.append(tt);self.targets.append(g)
            self.states.append(z);self.integrals.append(cs)
            self.change_times.extend(tt[1:])
        self.change_times=np.unique(self.change_times)

    def changes_between(self,a,b):
        left,right=np.searchsorted(self.change_times,[a,b],side='right')
        return self.change_times[left:right]

    def integrated(self,t):
        out=[]
        for tt,g,z,cs in zip(self.times,self.targets,self.states,self.integrals):
            if t<tt[0]:raise ValueError('Cannot query before first causal command')
            j=np.searchsorted(tt,t,side='right')-1;dt=t-tt[j]
            value=cs[j]+g[j]*dt
            if self.response is not None:
                value=value+(z[j]-g[j])*self.tau*(-np.expm1(-dt/self.tau))
            out.append(value)
        return np.asarray(out)

    def average(self,a,b):
        if b<=a:raise ValueError('Positive interval required')
        return (self.integrated(b)-self.integrated(a))/(b-a)
