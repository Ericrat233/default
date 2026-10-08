"""Noise MLE incl. censoring, false positive tradeoffs and dB thresholds."""
from dataclasses import dataclass
import numpy as np
from scipy import stats, optimize, special
from .core import minimize_checked, covariance
from .signals import db

@dataclass
class NoiseFit:
    family: str
    params: np.ndarray
    cov: np.ndarray
    log_likelihood: float
    n: int
    def distribution(self): return _distribution(self.family,self.params)
    def threshold(self,pfa=.01):
        if not 0<pfa<1: raise ValueError('0 < PFA < 1 required')
        return float(self.distribution().isf(pfa))
    def pfa(self,threshold): return self.distribution().sf(threshold)
    def summary(self,pfa=.01,reference=1.):
        th=self.threshold(pfa)
        return {'family':self.family,'parameters':self.params.tolist(),'covariance':self.cov.tolist(),
                'parameterization':'normal:mu,log(sd); lognormal:mu_log,log(sd_log); weibull:log(shape),log(scale); exponential:log(scale); LEV:location,log(scale)',
                'log_likelihood':self.log_likelihood,'aic':2*len(self.params)-2*self.log_likelihood,
                'threshold':th,'threshold_db':float(db(th,reference)) if th>0 else None,'pfa':pfa}


def _distribution(family,t):
    if family=='normal': return stats.norm(t[0],np.exp(t[1]))
    if family=='lognormal': return stats.lognorm(np.exp(t[1]),scale=np.exp(t[0]))
    if family=='weibull': return stats.weibull_min(np.exp(t[0]),scale=np.exp(t[1]))
    if family=='exponential': return stats.expon(scale=np.exp(t[0]))
    if family=='lev': return stats.gumbel_r(t[0],np.exp(t[1]))
    raise ValueError('unknown noise family')


def fit_noise(values,family='normal',status=None,bounds=None):
    y=np.asarray(values,float).reshape(-1)
    st=np.full(len(y),'exact') if status is None else np.asarray(status,str)
    if st.shape!=y.shape or not np.isin(st,['exact','left','right']).all(): raise ValueError('noise supports exact/left/right')
    exact=st=='exact'; left=st=='left'; right=st=='right'
    limits=y if bounds is None else np.broadcast_to(np.asarray(bounds,float),y.shape)
    if not np.isfinite(y[exact]).all() or not np.isfinite(limits[~exact]).all(): raise ValueError('finite values/bounds required')
    if exact.sum()<3 or np.std(y[exact])<1e-10: raise ValueError('noise fit requires >=3 exact variable observations')
    z=y[exact]
    if family=='normal': start=[z.mean(),np.log(z.std())]
    elif family=='lognormal':
        if np.any(z<=0) or np.any(limits[~exact]<=0): raise ValueError('positive lognormal values/bounds required')
        start=[np.log(z).mean(),np.log(np.log(z).std())]
    elif family=='weibull':
        if np.any(z<=0) or np.any(limits[~exact]<=0): raise ValueError('positive Weibull values/bounds required')
        start=[0,np.log(z.mean())]
    elif family=='exponential':
        if np.any(z<0) or np.any(limits[~exact]<0): raise ValueError('nonnegative exponential values required')
        start=[np.log(z.mean())]
    elif family=='lev': start=[z.mean()-0.5772*z.std()*np.sqrt(6)/np.pi,np.log(z.std()*np.sqrt(6)/np.pi)]
    else: raise ValueError('unknown family')
    def nll(t):
        if not np.isfinite(t).all() or np.max(np.abs(t if family in ('weibull','exponential') else t[-1:]))>30: return 1e100
        d=_distribution(family,t)
        lp=np.r_[d.logpdf(y[exact]),d.logcdf(limits[left]),d.logsf(limits[right])]
        return -float(lp.sum()) if np.isfinite(lp).all() else 1e100
    theta=minimize_checked(nll,start)
    return NoiseFit(family,theta,covariance(nll,theta),-nll(theta),len(y))


def threshold_tradeoff(signal_fit,noise_fit,thresholds,p=.9,confidence=.95,reference=1.):
    """Both fits MUST use the same response units and acquisition feature."""
    from dataclasses import replace
    rows=[]
    for th in thresholds:
        f=replace(signal_fit,threshold=float(th))
        rows.append([th,float(db(th,reference)) if th>0 else np.nan,float(noise_fit.pfa(th)),f.size(p),f.size_upper(p,confidence)])
    return np.asarray(rows)
