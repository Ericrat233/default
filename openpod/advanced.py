"""Research extensions; independent implementations, not reproductions of papers.
Gaussian random intercepts; conjugate Bayes or multi-chain censored MCMC;
calibrated simulation POD with explicit discrepancy and nuisance integration.
"""
from dataclasses import dataclass
import numpy as np
from scipy import linalg, special, stats
from .core import fit_signal, fit_hitmiss, minimize_checked, covariance, transform, inverse

@dataclass
class RandomInterceptFit:
    theta: np.ndarray
    cov: np.ndarray
    x_transform: str
    threshold: float
    group_count: int
    log_likelihood: float
    def pod(self,a,population=True):
        x=transform(a,self.x_transform); b0,b1,le,lu=self.theta
        sd=np.sqrt(np.exp(2*le)+(np.exp(2*lu) if population else 0))
        return special.ndtr((b0+b1*x-self.threshold)/sd)
    def size(self,p=.9):
        b0,b1,le,lu=self.theta
        if b1<=0: raise ValueError('nonpositive slope')
        q=(self.threshold-b0+special.ndtri(p)*np.sqrt(np.exp(2*le)+np.exp(2*lu)))/b1
        return float(inverse(q,self.x_transform))


def fit_random_intercept(a,y,groups,threshold,x_transform='log'):
    """Gaussian marginal ML, NOT REML. Uncensored raw signal only.
    Population POD integrates random observer/specimen effect; conditional POD
    at random effect zero is a different estimand. Boundary variance rejected.
    """
    a=np.asarray(a,float); y=np.asarray(y,float); groups=np.asarray(groups)
    if a.shape!=y.shape or groups.shape!=a.shape or not np.isfinite(y).all(): raise ValueError('finite matched vectors')
    X=np.column_stack((np.ones(len(a)),transform(a,x_transform))); ids=np.unique(groups)
    if len(ids)<4: raise ValueError('>=4 independent groups required')
    idx=[np.flatnonzero(groups==g) for g in ids]
    if min(map(len,idx))<2: raise ValueError('>=2 records per group')
    b=np.linalg.lstsq(X,y,rcond=None)[0]; sd=np.std(y-X@b)
    if sd<=0: raise ValueError('zero spread')
    def nll(t):
        if not np.isfinite(t).all() or np.max(np.abs(t[2:]))>25: return 1e100
        e=np.exp(2*t[2]); u=np.exp(2*t[3]); res=y-X@t[:2]; val=0.
        for i in idx:
            m=len(i); r=res[i]
            val+=m*np.log(2*np.pi)+(m-1)*np.log(e)+np.log(e+m*u)+r@r/e-u*r.sum()**2/(e*(e+m*u))
        return val/2
    theta=minimize_checked(nll,np.r_[b,np.log(sd*.7),np.log(sd*.7)])
    cov=covariance(nll,theta)
    return RandomInterceptFit(theta,cov,x_transform,threshold,len(ids),-nll(theta))


def bayes_linear(a,y,threshold,x_transform='log',draws=4000,seed=1823,
                 prior_mean=None,prior_precision=.01,alpha=2.,beta=1.):
    """Exact Normal-Inverse-Gamma posterior (uncensored). Proper priors in
    transformed response units; returns credibility, not frequentist confidence.
    beta_coeff | sigma² ~ N(m0,sigma² Lambda0^-1), sigma²~IG(alpha,beta).
    """
    a=np.asarray(a,float); y=np.asarray(y,float)
    if a.shape!=y.shape or not np.isfinite(y).all() or draws<100 or prior_precision<=0 or alpha<=0 or beta<=0: raise ValueError('invalid Bayesian data/prior')
    X=np.column_stack((np.ones(len(a)),transform(a,x_transform))); m0=np.zeros(2) if prior_mean is None else np.asarray(prior_mean,float)
    L0=prior_precision*np.eye(2); Ln=L0+X.T@X
    mn=np.linalg.solve(Ln,L0@m0+X.T@y); an=alpha+len(y)/2
    bn=beta+.5*(y@y+m0@L0@m0-mn@Ln@mn)
    rng=np.random.default_rng(seed); s2=stats.invgamma.rvs(an,scale=bn,size=draws,random_state=rng)
    B=mn+rng.normal(size=(draws,2))@np.linalg.cholesky(np.linalg.inv(Ln)).T*np.sqrt(s2)[:,None]
    valid=B[:,1]>0; q=np.full(draws,np.inf)
    q[valid]=inverse((threshold-B[valid,0]+special.ndtri(.9)*np.sqrt(s2[valid]))/B[valid,1],x_transform)
    return {'beta_draws':B,'sigma_draws':np.sqrt(s2),'a90_draws':q,'a90_credible_upper95':float(np.quantile(q,.95)),
            'nonpositive_slope_fraction':float((~valid).mean()),'posterior_mean':mn,'posterior_precision':Ln}


def _chain_diagnostics(chains):
    # Split-chain Rhat, conservative initial positive autocorrelation ESS.
    m,n,d=chains.shape; half=n//2
    split=np.concatenate((chains[:,:half],chains[:,-half:]),axis=0)
    W=split.var(axis=1,ddof=1).mean(axis=0); B=half*split.mean(axis=1).var(axis=0,ddof=1)
    rhat=np.sqrt(((half-1)/half*W+B/half)/W)
    ess=[]
    for j in range(d):
        rho=[]
        for lag in range(1,min(n//2,1000)):
            values=[]
            for chain in chains[:,:,j]:
                c=chain-chain.mean(); values.append(np.dot(c[:-lag],c[lag:])/np.dot(c,c))
            rr=np.mean(values)
            if rr<=0: break
            rho.append(rr)
        ess.append(m*n/(1+2*sum(rho)))
    return rhat,np.array(ess)


def bayes_censored(fit,draws=1500,burn=1000,chains=4,seed=1823,prior_sd=10.):
    """Random-walk Metropolis using censored likelihood and N(0,sd²) prior
    on each fitted parameter, including log(sigma). Adaptation during burn only.
    Model/prior sensitivity and chain diagnostics are required before inference.
    """
    if fit.kind!='signal' or fit.covariate_count or draws<100 or burn<100 or chains<2 or prior_sd<=0: raise ValueError('invalid MCMC request')
    rng=np.random.default_rng(seed); chol=np.linalg.cholesky(fit.cov/fit.repeated_factor)
    d=len(fit.theta); output=np.empty((chains,draws,d)); acceptance=[]
    def lp(t): return -fit.nll(t)-.5*np.sum((t/prior_sd)**2)
    for c in range(chains):
        t=fit.theta+chol@rng.normal(size=d); v=lp(t); scale=2.38/np.sqrt(d); accepted=0; window=0
        for i in range(burn+draws):
            new=t+scale*(chol@rng.normal(size=d)); value=lp(new)
            accept=np.log(rng.uniform())<value-v
            if accept: t=new; v=value; window+=1
            if i<burn and (i+1)%50==0:
                scale*=np.exp(np.clip(window/50-.25,-.2,.2)); window=0
            if i>=burn: output[c,i-burn]=t; accepted+=accept
        acceptance.append(accepted/draws)
    rhat,ess=_chain_diagnostics(output); flat=output.reshape(-1,d); pos=flat[:,1]>0
    q=np.full(len(flat),np.inf)
    q[pos]=inverse((transform(fit.threshold,fit.y_transform)-flat[pos,0]+special.ndtri(.9)*np.exp(flat[pos,-1]))/flat[pos,1],fit.x_transform)
    return {'chains':output,'rhat':rhat,'ess':ess,'acceptance':np.array(acceptance),
            'a90_credible_upper95':float(np.quantile(q,.95)) if not fit.covariate_count else None,
            'converged_diagnostic':bool(np.all(rhat<1.01) and np.all(ess>400)),
            'note':'Rhat/ESS alone do not establish posterior validity'}


@dataclass
class SimulationCalibration:
    coefficients: np.ndarray
    covariance: np.ndarray
    discrepancy_sd: float
    n: int
    def predict(self,simulation): return self.coefficients[0]+self.coefficients[1]*np.asarray(simulation)


def calibrate_simulation(simulation,experiment):
    """Paired transfer function with experimental residual discrepancy.
    Paired calibration data must be independent of validation specimens.
    """
    x=np.asarray(simulation,float); y=np.asarray(experiment,float)
    if x.ndim!=1 or x.shape!=y.shape or len(x)<5 or not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError('>=5 finite calibration pairs required')
    X=np.column_stack((np.ones(len(x)),x))
    if np.linalg.matrix_rank(X)<2: raise ValueError('constant simulation input')
    b=np.linalg.lstsq(X,y,rcond=None)[0]; sd=np.sqrt(np.sum((y-X@b)**2)/(len(x)-2))
    return SimulationCalibration(b,sd**2*np.linalg.inv(X.T@X),float(sd),len(x))


def mapod_monte_carlo(simulator,a_grid,nuisance,threshold,calibration=None,seed=1823,
                      calibration_draws=0):
    """simulator(a,nuisance)-> responses (len(a),n_mc), model assisted POD.
    Integrates specified nuisance population; adds calibration discrepancy noise.
    Optional calibration parameter draws produce an uncertainty sensitivity band,
    NOT a confidence band for an unvalidated physical simulator.
    """
    a=np.asarray(a_grid,float); c=np.asarray(nuisance,float)
    if a.ndim!=1 or c.ndim!=2 or len(c)<20: raise ValueError('size grid and >=20 nuisance samples required')
    z=np.asarray(simulator(a,c),float)
    if z.shape!=(len(a),len(c)) or not np.isfinite(z).all(): raise ValueError('simulator shape/nonfinite output')
    rng=np.random.default_rng(seed)
    def evaluate(coef=None):
        if calibration is None: y=z
        else:
            b=calibration.coefficients if coef is None else coef
            y=b[0]+b[1]*z+rng.normal(0,calibration.discrepancy_sd,z.shape)
        hits=np.sum(y>threshold,axis=1); p=hits/len(c)
        lo=np.where(hits==0,0,stats.beta.ppf(.025,hits,len(c)-hits+1))
        hi=np.where(hits==len(c),1,stats.beta.ppf(.975,hits+1,len(c)-hits))
        return p,lo,hi
    pod,lo,hi=evaluate(); result={'a':a,'pod':pod,'mc_lower':lo,'mc_upper':hi,'n_mc':len(c),'uncertainty_note':'binomial Monte Carlo error only; nuisance distribution and simulator validity are assumed'}
    if calibration is not None and calibration_draws>0:
        B=rng.multivariate_normal(calibration.coefficients,calibration.covariance,size=calibration_draws)
        curves=np.array([evaluate(b)[0] for b in B]); result['calibration_band']=np.quantile(curves,[.025,.975],axis=0)
    return result


class GaussianSurrogate:
    """Small-data fixed-hyperparameter RBF GP with explicit observation nugget.
    Input scaling determined on training data; no claims of automatic optimality.
    """
    def __init__(self,length_scale=1.,nugget=1e-4):
        if length_scale<=0 or nugget<=0: raise ValueError('positive GP hyperparameters')
        self.length_scale=length_scale; self.nugget=nugget
    def fit(self,X,y):
        X=np.asarray(X,float); y=np.asarray(y,float)
        if X.ndim!=2 or y.shape!=(len(X),) or not np.isfinite(X).all() or not np.isfinite(y).all(): raise ValueError('finite GP training inputs')
        self.center=X.mean(0); self.scale=X.std(0); self.scale[self.scale==0]=1
        self.X=(X-self.center)/self.scale; self.mean=y.mean(); self.sd=y.std()
        if self.sd==0: raise ValueError('GP output has no variation')
        K=self._kernel(self.X,self.X)+self.nugget*np.eye(len(X))
        self.chol=linalg.cho_factor(K); self.alpha=linalg.cho_solve(self.chol,(y-self.mean)/self.sd)
        return self
    def _kernel(self,X,Y):
        D=np.sum((X[:,None,:]-Y[None,:,:])**2,axis=2)
        return np.exp(-D/(2*self.length_scale**2))
    def predict(self,X):
        X=(np.asarray(X,float)-self.center)/self.scale; K=self._kernel(X,self.X)
        mean=self.mean+self.sd*K@self.alpha
        v=1-np.sum(K*linalg.cho_solve(self.chol,K.T).T,axis=1)
        return mean,self.sd*np.sqrt(np.maximum(v,0))


def marginal_pod(fit,a_grid,nuisance):
    c=np.asarray(nuisance,float); a=np.asarray(a_grid,float)
    if c.ndim!=2 or c.shape[1]!=fit.covariate_count or len(c)==0: raise ValueError('nuisance shape')
    return np.array([fit.pod(np.full(len(c),v),c).mean() for v in a])


def cluster_bootstrap(a,y,groups,fit_kwargs,kind='signal',replicates=200,seed=1823):
    a=np.asarray(a); y=np.asarray(y); groups=np.asarray(groups)
    if a.shape!=y.shape or groups.shape!=a.shape: raise ValueError('matched group vectors')
    ids=np.unique(groups)
    if len(ids)<4 or replicates<20: raise ValueError('bootstrap requires >=4 independent groups and >=20 replicates')
    rng=np.random.default_rng(seed); sizes=[]; failures=0
    for _ in range(replicates):
        idx=np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(ids,len(ids),replace=True)])
        kw=dict(fit_kwargs)
        for key in ['status','lower','upper','covariates']:
            if key in kw and kw[key] is not None and np.ndim(kw[key])>0: kw[key]=np.asarray(kw[key])[idx]
        try:
            fit=(fit_signal if kind=='signal' else fit_hitmiss)(a[idx],y[idx],**kw)
            sizes.append(fit.size())
        except (ValueError,RuntimeError): failures+=1
    if failures>replicates*.2: raise RuntimeError('>20% bootstrap fits failed; do not report a nominal bound')
    return {'a90_samples':np.array(sizes),'a90_bootstrap_upper95':float(np.quantile(sizes,.95)),
            'successful':len(sizes),'failed':failures,'method':'cluster percentile bootstrap'}
