"""Diagnostics that keep censoring and fitted-parameter uncertainty explicit."""
import numpy as np
from scipy import special,stats
from .core import transform,binomial_interval


def randomized_quantile_residuals(fit,a,y,status=None,lower=None,upper=None,covariates=None,seed=1823):
    if fit.kind!='signal': raise ValueError('signal fit required')
    y=np.asarray(y,float); n=len(y); X=fit._design(a,covariates)
    if len(X)!=n: raise ValueError('data lengths differ')
    st=np.full(n,'exact') if status is None else np.asarray(status,str)
    L=np.full(n,-np.inf) if lower is None else np.broadcast_to(np.asarray(lower,float),(n,))
    U=np.full(n,np.inf) if upper is None else np.broadcast_to(np.asarray(upper,float),(n,))
    mu=X@fit.theta[:-1]; sd=np.exp(fit.theta[-1]); low=np.zeros(n); high=np.ones(n); exact=st=='exact'
    low[exact]=special.ndtr((transform(y[exact],fit.y_transform)-mu[exact])/sd); high[exact]=low[exact]
    left=(st=='left')|(st=='interval'); right=(st=='right')|(st=='interval')
    high[left]=special.ndtr((transform(U[left],fit.y_transform)-mu[left])/sd)
    low[right]=special.ndtr((transform(L[right],fit.y_transform)-mu[right])/sd)
    rng=np.random.default_rng(seed); u=low+(high-low)*rng.uniform(size=n)
    r=special.ndtri(np.clip(u,np.finfo(float).eps,1-np.finfo(float).eps)); r[st=='missing']=np.nan
    return r


def grouped_hitmiss_gof(fit,a,hit,bins=8,bootstrap=199,seed=1823):
    """Fixed size-quantile groups; re-fit model in parametric GOF bootstrap.
    Used only for independent baseline observations, not correlated repeats.
    """
    from .core import fit_hitmiss
    if fit.kind!='hitmiss' or fit.covariate_count or fit.repeated_factor!=1: raise ValueError('baseline independent hitmiss only')
    a=np.asarray(a,float); y=np.asarray(hit,float); order=np.argsort(a); groups=np.array_split(order,bins); rng=np.random.default_rng(seed)
    def score(model,y):
        p=model.pod(a); val=0.
        for g in groups:
            expected=p[g].sum(); variance=np.sum(p[g]*(1-p[g])); val+=(y[g].sum()-expected)**2/max(variance,1e-12)
        return val
    observed=score(fit,y); scores=[]; failed=0
    for _ in range(bootstrap):
        sample=rng.binomial(1,fit.pod(a))
        try:
            f=fit_hitmiss(a,sample,x_transform=fit.x_transform,link_name=fit.link_name)
            scores.append(score(f,sample))
        except (ValueError,RuntimeError): failed+=1
    if failed>bootstrap*.1: raise RuntimeError('too many GOF bootstrap failures')
    return {'grouped_pearson':float(observed),'bootstrap_p':float((1+sum(v>=observed for v in scores))/(1+len(scores))),
            'replicates_successful':len(scores),'failures':failed,'note':'model adequacy diagnostic, not proof of valid POD extrapolation'}


def validate_false_alarm(noise_features,threshold,confidence=.95):
    x=np.asarray(noise_features,float)
    if x.ndim!=1 or not np.isfinite(x).all() or len(x)==0: raise ValueError('finite independent held-out background features required')
    hits=int(np.sum(x>threshold)); low,high=binomial_interval(hits,len(x),confidence)
    return {'false_alarms':hits,'trials':len(x),'observed_pfa':hits/len(x),'lower':low,'upper':high,
            'note':'trial unit must match inspection decision; correlated samples require cluster-level validation'}


def plot_diagnostics(fit,a,residuals,path):
    import matplotlib.pyplot as plt
    r=np.asarray(residuals,float); mask=np.isfinite(r); r=r[mask]; a=np.asarray(a)[mask]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].scatter(a,r,s=12); axes[0].axhline(0,color='grey'); axes[0].set_xlabel('Flaw size'); axes[0].set_ylabel('Residual')
    positions=(np.arange(len(r))+.5)/len(r); quantiles=special.ndtri(positions)
    axes[1].plot(quantiles,np.sort(r),'o',ms=3); axes[1].plot(quantiles,quantiles,'--'); axes[1].set_xlabel('Normal quantile'); axes[1].set_ylabel('Ordered residual')
    fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)
