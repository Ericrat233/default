"""MLE POD. Parameter covariance uses observed information, log(sigma) convention.
No upstream mh1823 source is incorporated. Raw likelihoods live in transformed
response units (constant Jacobian omitted); compare AIC only within same transform.
"""
from dataclasses import dataclass, field
import warnings
import numpy as np
from scipy import optimize, special, stats


def transform(x, kind):
    x = np.asarray(x, float)
    if kind == 'identity': return x
    if kind != 'log': raise ValueError('transform must be identity or log')
    if np.any(x <= 0): raise ValueError('log transform requires positive values')
    return np.log(x)


def inverse(x, kind):
    with np.errstate(over='ignore'): return np.exp(x) if kind == 'log' else x


def link(p, name):
    p = np.asarray(p, float)
    if np.any((p <= 0) | (p >= 1)): raise ValueError('0 < POD < 1 required')
    if name == 'logit': return special.logit(p)
    if name == 'probit': return special.ndtri(p)
    if name == 'cloglog': return np.log(-np.log1p(-p))
    if name == 'loglog': return -np.log(-np.log(p))
    raise ValueError('unknown link')


def link_logs(eta, name):
    eta = np.asarray(eta, float)
    if name == 'logit': return -np.logaddexp(0, -eta), -np.logaddexp(0, eta)
    if name == 'probit': return special.log_ndtr(eta), special.log_ndtr(-eta)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        if name == 'cloglog':
            e = np.exp(np.minimum(eta, 700))
            return np.where(eta < -35, eta, np.log(-np.expm1(-e))), -e
        if name == 'loglog':
            lp, lq = link_logs(-eta, 'cloglog')
            return lq, lp
    raise ValueError('unknown link')


def log_interval(lo, hi):
    """Stable log[Phi(hi)-Phi(lo)], using survival tails for positive intervals."""
    lo, hi = np.broadcast_arrays(lo, hi)
    a = np.where(lo > 0, special.log_ndtr(-lo), special.log_ndtr(hi))
    b = np.where(lo > 0, special.log_ndtr(-hi), special.log_ndtr(lo))
    with np.errstate(divide='ignore', invalid='ignore'):
        return a + np.log(-np.expm1(np.minimum(b-a, 0)))


def hessian(fn, theta):
    theta = np.asarray(theta, float)
    step = 2e-4 * np.maximum(1, np.abs(theta))
    n = len(theta); H = np.empty((n, n)); f = fn(theta)
    for i in range(n):
        ei = np.eye(n)[i]*step[i]
        H[i,i] = (fn(theta+ei)-2*f+fn(theta-ei))/step[i]**2
        for j in range(i):
            ej = np.eye(n)[j]*step[j]
            H[i,j] = H[j,i] = (fn(theta+ei+ej)-fn(theta+ei-ej)-fn(theta-ei+ej)+fn(theta-ei-ej))/(4*step[i]*step[j])
    return (H+H.T)/2


def covariance(fn, theta):
    H = hessian(fn, theta)
    if not np.isfinite(H).all() or np.min(np.linalg.eigvalsh(H)) <= 0:
        raise RuntimeError('non-positive observed information: unidentified or boundary MLE')
    return np.linalg.inv(H)


def minimize_checked(fn, start):
    r = optimize.minimize(fn, start, method='BFGS', options={'gtol':1e-7, 'maxiter':2000})
    if not r.success:
        s = optimize.minimize(fn, r.x, method='Nelder-Mead', options={'xatol':1e-9, 'fatol':1e-9, 'maxiter':8000})
        if s.success and np.isfinite(s.fun): r = s
        else: raise RuntimeError('MLE did not converge: '+r.message)
    if not np.isfinite(r.fun): raise RuntimeError('nonfinite maximum likelihood')
    return r.x


@dataclass
class PODFit:
    kind: str
    theta: np.ndarray
    cov: np.ndarray
    nll: object = field(repr=False)
    x_transform: str = 'log'
    y_transform: str = 'identity'
    threshold: float = 0.
    link_name: str = 'probit'
    n: int = 0
    diagnostics: dict = field(default_factory=dict)
    covariate_count: int = 0
    repeated_factor: float = 1.

    def _design(self, a, covariates=None):
        a = np.atleast_1d(np.asarray(a, float))
        if not np.isfinite(a).all(): raise ValueError('nonfinite sizes')
        X = np.column_stack((np.ones(len(a)), transform(a, self.x_transform)))
        if self.covariate_count:
            c = np.asarray(covariates, float)
            if c.shape != (len(a), self.covariate_count) or not np.isfinite(c).all():
                raise ValueError('covariates require finite (n,k) shape')
            X = np.column_stack((X,c))
        return X

    def pod(self, a, covariates=None, threshold=None):
        X = self._design(a, covariates)
        if self.kind == 'hitmiss': return np.exp(link_logs(X@self.theta, self.link_name)[0])
        c = transform(self.threshold if threshold is None else threshold, self.y_transform)
        return special.ndtr((X@self.theta[:-1]-c)/np.exp(self.theta[-1]))

    def quantile_x(self, p=.9, covariates=None, threshold=None):
        if self.theta[1] <= 0: raise ValueError('nonpositive size slope: increasing POD limit undefined')
        offset = self.theta[0]
        if self.covariate_count:
            c = np.asarray(covariates,float)
            if c.shape != (self.covariate_count,): raise ValueError('quantile covariates require k entries')
            offset += c @ (self.theta[2:] if self.kind=='hitmiss' else self.theta[2:-1])
        target = link(p, self.link_name) if self.kind=='hitmiss' else transform(self.threshold if threshold is None else threshold,self.y_transform)+special.ndtri(p)*np.exp(self.theta[-1])
        return float((target-offset)/self.theta[1])

    def size(self, p=.9, covariates=None, threshold=None):
        return float(inverse(self.quantile_x(p,covariates,threshold),self.x_transform))

    def size_upper(self, p=.9, confidence=.95, method=None, lr_df=1, covariates=None):
        """Default mh-compatible: signal horizontal delta-Wald; hitmiss LR df=1.
        lr_df=2 gives joint 2-parameter region for simple hit/miss; profile_one_sided
        uses z(confidence)^2. These confidence definitions are intentionally distinct.
        """
        if not .5 < confidence < 1: raise ValueError('confidence must lie in (.5,1)')
        if self.covariate_count and covariates is None: raise ValueError('set covariates')
        q = self.quantile_x(p, covariates)
        method = method or ('wald' if self.kind=='signal' else 'lr')
        if method == 'wald':
            g = np.zeros_like(self.theta); g[0]=-1/self.theta[1]; g[1]=-q/self.theta[1]
            if self.covariate_count:
                g[2:2+self.covariate_count]=-np.asarray(covariates)/self.theta[1]
            if self.kind == 'signal': g[-1]=special.ndtri(p)*np.exp(self.theta[-1])/self.theta[1]
            upper = q+special.ndtri(confidence)*np.sqrt(g@self.cov@g)
        elif method in ('lr','profile_one_sided'):
            if self.covariate_count: raise ValueError('LR size implemented for two-variable baseline only; use Wald or bootstrap with covariates')
            cutoff = special.ndtri(confidence)**2 if method=='profile_one_sided' else stats.chi2.ppf(confidence,lr_df)
            base = self.nll(self.theta); b = self.theta[1]
            def prof(qx):
                def obj(v):
                    if self.kind=='hitmiss': t = np.array([link(p,self.link_name)-v[0]*qx, v[0]])
                    else:
                        t = np.array([transform(self.threshold,self.y_transform)+special.ndtri(p)*np.exp(v[1])-v[0]*qx, v[0], v[1]])
                    return self.nll(t)
                v = [b] if self.kind=='hitmiss' else [b,self.theta[-1]]
                r = optimize.minimize(obj,v,method='BFGS')
                if not r.success:
                    r = optimize.minimize(obj,r.x,method='Nelder-Mead',options={'maxiter':3000,'xatol':1e-8})
                if not r.success: raise RuntimeError('profile likelihood failed')
                return 2*(r.fun-base)/self.repeated_factor-cutoff
            hi=q+max(1,abs(q)*.1)
            for _ in range(40):
                if prof(hi)>0: break
                hi=q+2*(hi-q)
            else: return np.inf
            upper=optimize.brentq(prof,q,hi,xtol=1e-9)
        else: raise ValueError('method must be wald/lr/profile_one_sided')
        return float(inverse(upper,self.x_transform))

    def lr_band(self,a,confidence=.95,df=1):
        if self.kind!='hitmiss' or self.covariate_count: raise ValueError('baseline hitmiss only')
        X=self._design(a); level=self.nll(self.theta)+self.repeated_factor*stats.chi2.ppf(confidence,df)/2
        low=[]; high=[]
        for row in X:
            ends=[]
            for sign in (1,-1):
                r=optimize.minimize(lambda t:sign*(row@t),self.theta,method='SLSQP',
                    constraints={'type':'ineq','fun':lambda t:level-self.nll(t)},options={'ftol':1e-9,'maxiter':500})
                if not r.success: raise RuntimeError('LR envelope optimization failed')
                ends.append(float(np.exp(link_logs(row@r.x,self.link_name)[0])))
            low.append(ends[0]); high.append(ends[1])
        return np.array(low),np.array(high)

    def wald_quantile_band(self, probabilities, confidence=.95):
        """Horizontal bounds parameterized by POD; avoids conflating x- and y-Wald."""
        q=np.asarray(probabilities)
        lower=np.array([self.size_upper(p,confidence,'wald') for p in q])
        centers=np.array([self.size(p) for p in q])
        # Symmetric on transformed size, not on raw logarithmic size.
        upper=inverse(2*transform(centers,self.x_transform)-transform(lower,self.x_transform),self.x_transform)
        return upper,centers,lower

    def summary(self, p=.9, confidence=.95):
        result={'kind':self.kind,'parameters':self.theta.tolist(),'parameterization':'intercept,slope,covariates[,log(sigma)]',
                'covariance':self.cov.tolist(),'x_transform':self.x_transform,'y_transform':self.y_transform,
                'threshold':self.threshold,'link':self.link_name,'n':self.n,'log_likelihood':-float(self.nll(self.theta)),
                'aic':2*len(self.theta)+2*float(self.nll(self.theta)),'diagnostics':self.diagnostics,
                'repeated_factor':self.repeated_factor,'critical_POD':p,'confidence':confidence,
                'confidence_method':'horizontal_delta_wald' if self.kind=='signal' else 'LR_envelope_df1'}
        if self.covariate_count==0 and self.theta[1]>0:
            b=self.theta; slope=b[1]
            c=transform(self.threshold,self.y_transform) if self.kind=='signal' else 0.
            mu=(c-b[0])/slope
            scale=(np.exp(b[-1]) if self.kind=='signal' else 1.)/slope
            J=np.zeros((2,len(b))); J[0,0]=-1/slope; J[0,1]=-mu/slope; J[1,1]=-scale/slope
            if self.kind=='signal': J[1,-1]=scale
            result['POD_location_scale']={'mu_transformed':float(mu),'scale_transformed':float(scale),'covariance':(J@self.cov@J.T).tolist(),
                'note':'mu equals transformed a50 only for symmetric links'}
            result.update(a50=self.size(.5),a90=self.size(.9),a_p=self.size(p),a_p_conf=self.size_upper(p,confidence))
        return result


def _design(a,x_transform,covariates):
    a=np.asarray(a,float).reshape(-1)
    if not np.isfinite(a).all(): raise ValueError('sizes must be finite; missing sizes require a separate model')
    X=np.column_stack((np.ones(len(a)),transform(a,x_transform)))
    if covariates is not None:
        c=np.asarray(covariates,float)
        if c.ndim!=2 or c.shape[0]!=len(a) or not np.isfinite(c).all(): raise ValueError('finite covariates (n,k) required')
        X=np.column_stack((X,c))
    if np.linalg.matrix_rank(X)<X.shape[1]: raise ValueError('rank-deficient design')
    return a,X


def fit_signal(a,y,threshold,*,x_transform='log',y_transform='identity',status=None,
               lower=None,upper=None,covariates=None,missing='error',repeated_factor=1):
    """status exact/left/right/interval/missing; lower/upper on raw response scale.
    For left use upper, for right use lower. Censored y may be NaN.
    Missing response excluded only with explicit missing='mar' assumption.
    """
    if not np.isfinite(threshold): raise ValueError('finite threshold required')
    a,X=_design(a,x_transform,covariates); y=np.asarray(y,float).reshape(-1); n=len(a)
    if len(y)!=n: raise ValueError('size/response length mismatch')
    if repeated_factor<1: raise ValueError('repeated_factor >=1 required')
    st=np.where(np.isnan(y),'missing','exact') if status is None else np.asarray(status,str)
    if st.shape!=(n,) or not np.isin(st,['exact','left','right','interval','missing']).all(): raise ValueError('invalid censor status')
    miss=st=='missing'
    if miss.any() and missing!='mar': raise ValueError('missing responses: explicitly set missing="mar" or use a missingness model')
    lo=np.full(n,-np.inf) if lower is None else np.broadcast_to(np.asarray(lower,float),(n,)).copy()
    hi=np.full(n,np.inf) if upper is None else np.broadcast_to(np.asarray(upper,float),(n,)).copy()
    exact=st=='exact'; left=st=='left'; right=st=='right'; interval=st=='interval'
    if not np.isfinite(y[exact]).all(): raise ValueError('exact response must be finite')
    if not np.isfinite(hi[left|interval]).all() or not np.isfinite(lo[right|interval]).all(): raise ValueError('finite censor bounds required')
    if np.any(lo[interval]>=hi[interval]): raise ValueError('interval lower must be < upper')
    yt=np.zeros(n); yt[exact]=transform(y[exact],y_transform)
    lt=np.full(n,-np.inf); ut=np.full(n,np.inf)
    lt[right|interval]=transform(lo[right|interval],y_transform); ut[left|interval]=transform(hi[left|interval],y_transform)
    keep=~miss; X=X[keep]; st=st[keep]; yt=yt[keep]; lt=lt[keep]; ut=ut[keep]
    if len(X)<=X.shape[1]+1 or np.linalg.matrix_rank(X)<X.shape[1]: raise ValueError('insufficient identified data')
    exact=st=='exact'; left=st=='left'; right=st=='right'; interval=st=='interval'
    if exact.sum()<X.shape[1]: raise ValueError('at least k exact records required for stable initialization')
    initial=np.linalg.lstsq(X[exact],yt[exact],rcond=None)[0]
    s=np.std(yt[exact]-X[exact]@initial)
    if s<1e-10: raise ValueError('zero residual spread; no regular Gaussian MLE')
    def nll(t):
        if not np.isfinite(t).all() or abs(t[-1])>30: return 1e100
        sig=np.exp(t[-1]); mu=X@t[:-1]; terms=np.zeros(len(X))
        terms[exact]=stats.norm.logpdf(yt[exact],mu[exact],sig)
        terms[left]=special.log_ndtr((ut[left]-mu[left])/sig)
        terms[right]=special.log_ndtr((mu[right]-lt[right])/sig)
        terms[interval]=log_interval((lt[interval]-mu[interval])/sig,(ut[interval]-mu[interval])/sig)
        return -float(terms.sum()) if np.isfinite(terms).all() else 1e100
    t=minimize_checked(nll,np.r_[initial,np.log(s)]); cov=covariance(nll,t)*repeated_factor
    residual=yt[exact]-X[exact]@t[:-1]
    diag={'n_exact':int(exact.sum()),'n_left':int(left.sum()),'n_right':int(right.sum()),'n_interval':int(interval.sum()),
          'n_missing_excluded':int(miss.sum()),'missing_assumption':missing,'rmse_exact':float(np.sqrt(np.mean(residual**2))),
          'residuals_exact':residual.tolist(),'residual_normality_note':'exact-only diagnostics selection-biased under censoring'}
    if len(residual)>=8: diag['normaltest_exact_p']=float(stats.normaltest(residual).pvalue)
    if len(residual)>=4:
        z=X[exact]; r2=residual**2; b=np.linalg.lstsq(z,r2,rcond=None)[0]
        denom=np.sum((r2-r2.mean())**2)
        diag['breusch_pagan_exact_p']=float(stats.chi2.sf(len(r2)*(1-np.sum((r2-z@b)**2)/denom),z.shape[1]-1)) if denom>0 else None
    f=PODFit('signal',t,cov,nll,x_transform,y_transform,float(threshold),'probit',len(X),diag,X.shape[1]-2,repeated_factor)
    transform(threshold,y_transform)
    if t[1]<=0: warnings.warn('nonpositive slope: POD detection limit undefined')
    return f


def fit_hitmiss(a,hit,*,x_transform='log',link_name='logit',covariates=None,repeated_factor=1,covariance_method='fisher'):
    a,X=_design(a,x_transform,covariates); y=np.asarray(hit,float).reshape(-1)
    if y.shape!=(len(a),) or not np.isin(y,[0,1]).all() or len(np.unique(y))!=2: raise ValueError('binary data with both hits and misses required')
    if repeated_factor<1: raise ValueError('repeated_factor >=1 required')
    link(.5,link_name)
    # Center/scale the size internally to avoid numerical failure for mm versus mil.
    center=X[:,1:].mean(0); scale=X[:,1:].std(0); Z=np.column_stack((X[:,0],(X[:,1:]-center)/scale))
    def objective(b):
        lp,lq=link_logs(Z@b,link_name)
        return -float(np.where(y==1,lp,lq).sum())
    b=minimize_checked(objective,np.zeros(X.shape[1]))
    if np.linalg.norm(b)>100 or np.max(np.abs(Z@b))>200: raise ValueError('separation or near separation: unregularized MLE not reliable')
    def nll(t):
        lp,lq=link_logs(X@t,link_name)
        v=-np.where(y==1,lp,lq).sum()
        return float(v) if np.isfinite(v) else 1e100
    t=np.r_[b[0]-np.sum(b[1:]*center/scale),b[1:]/scale]
    if covariance_method=='observed': cov=covariance(nll,t)*repeated_factor
    elif covariance_method=='fisher':
        eta=X@t; lp,lq=link_logs(eta,link_name)
        if link_name=='logit': ld=lp+lq
        elif link_name=='probit': ld=-.5*eta**2-.5*np.log(2*np.pi)
        elif link_name=='cloglog': ld=eta-np.exp(np.minimum(eta,700))
        else: ld=-eta-np.exp(np.minimum(-eta,700))
        w=np.exp(2*ld-lp-lq); info=X.T@(X*w[:,None])
        if np.min(np.linalg.eigvalsh(info))<=0: raise RuntimeError('singular Fisher information')
        cov=np.linalg.inv(info)*repeated_factor
    else: raise ValueError('covariance_method fisher/observed')
    prob=np.exp(link_logs(X@t,link_name)[0]); ll=-nll(t)
    sat=0.; dev=2*(sat-ll)
    diag={'covariance_method':covariance_method,'deviance':float(dev),'brier_score':float(np.mean((prob-y)**2)),
          'pearson_chi2':float(np.sum((y-prob)**2/np.maximum(prob*(1-prob),1e-15))),
          'goodness_of_fit_note':'individual Bernoulli deviance is not a calibrated chi-square GOF test; use replicated sizes/grouped assessment'}
    if t[1]<=0: warnings.warn('nonpositive size slope')
    return PODFit('hitmiss',t,cov,nll,x_transform,'identity',0.,link_name,len(y),diag,X.shape[1]-2,repeated_factor)


def binomial_interval(hits,total,confidence=.95,one_sided=False):
    if total<=0 or hits<0 or hits>total: raise ValueError('0 <= hits <= total required')
    alpha=(1-confidence) if one_sided else (1-confidence)/2
    return (0. if hits==0 else float(stats.beta.ppf(alpha,hits,total-hits+1)),
            1. if hits==total else float(stats.beta.ppf(1-alpha,hits+1,total-hits)))


def compare_nested(reduced,full):
    if reduced.n!=full.n or reduced.kind!=full.kind or reduced.x_transform!=full.x_transform or reduced.y_transform!=full.y_transform:
        raise ValueError('compare nested models on same data, likelihood and transform')
    df=len(full.theta)-len(reduced.theta)
    if df<=0: raise ValueError('full model must have more parameters')
    lr=2*(reduced.nll(reduced.theta)-full.nll(full.theta))
    if lr < -1e-6: raise ValueError('non-nested models or optimization inconsistency')
    return {'LR':float(lr),'df':df,'p_value':float(stats.chi2.sf(max(0,lr),df))}
