import numpy as np
import pytest
from scipy import special,stats
from openpod import fit_signal,fit_hitmiss,binomial_interval
from openpod.core import log_interval,link,link_logs
from openpod.noise import fit_noise
from openpod.advanced import calibrate_simulation,mapod_monte_carlo,GaussianSurrogate,bayes_linear,fit_random_intercept


def test_signal_closed_form_and_wald():
    rng=np.random.default_rng(12); a=np.exp(np.linspace(-1,2,140)); y=.5+1.8*np.log(a)+rng.normal(0,.4,len(a))
    fit=fit_signal(a,y,2.); X=np.column_stack((np.ones(len(a)),np.log(a))); b=np.linalg.lstsq(X,y,rcond=None)[0]
    s=np.sqrt(np.mean((y-X@b)**2)); np.testing.assert_allclose(fit.theta,np.r_[b,np.log(s)],atol=2e-6)
    expected=np.zeros((3,3)); expected[:2,:2]=s*s*np.linalg.inv(X.T@X); expected[2,2]=1/(2*len(a))
    np.testing.assert_allclose(fit.cov,expected,rtol=2e-5,atol=1e-7)
    q=(2-b[0]+special.ndtri(.9)*s)/b[1]; g=np.array([-1/b[1],-q/b[1],special.ndtri(.9)*s/b[1]])
    assert fit.size_upper()==pytest.approx(np.exp(q+special.ndtri(.95)*np.sqrt(g@expected@g)),rel=2e-5)

@pytest.mark.parametrize('name',['logit','probit','cloglog','loglog'])
def test_binary_statsmodels(name):
    import statsmodels.api as sm
    rng=np.random.default_rng(20); a=np.exp(np.linspace(-1,2,200)); x=np.log(a); p=np.exp(link_logs(-2+2.8*x,name)[0]); y=rng.binomial(1,p)
    f=fit_hitmiss(a,y,link_name=name); X=sm.add_constant(x)
    family={'logit':sm.families.links.Logit(),'probit':sm.families.links.Probit(),'cloglog':sm.families.links.CLogLog(),'loglog':sm.families.links.LogLog()}[name]
    r=sm.GLM(y,X,family=sm.families.Binomial(link=family)).fit()
    np.testing.assert_allclose(f.theta,r.params,rtol=2e-4,atol=2e-5)
    # Default Fisher covariance aligns with native GLM; observed information is optional.
    np.testing.assert_allclose(f.cov,r.cov_params(),rtol=3e-4,atol=1e-6)
    assert abs(f.nll(f.theta)+r.llf)<1e-5
    assert f.size_upper()>f.size()


def test_stable_extreme_interval():
    assert np.isfinite(log_interval(np.array([40.,-42]),np.array([41.,-41]))).all()
    assert log_interval(np.array([1.]),np.array([1.1]))[0]==pytest.approx(np.log(stats.norm.cdf(1.1)-stats.norm.cdf(1.)),abs=1e-12)


def test_censored_recovery_and_missing():
    rng=np.random.default_rng(321); a=np.exp(rng.uniform(-1,2,900)); latent=.5+1.8*np.log(a)+rng.normal(0,.5,len(a))
    y=latent.copy(); st=np.full(len(a),'exact',dtype='<U8'); lo=np.full(len(a),-np.inf); hi=np.full(len(a),np.inf)
    left=y<0; right=y>3.5; st[left]='left'; st[right]='right'; hi[left]=0; lo[right]=3.5; y[left|right]=np.nan
    idx=np.flatnonzero(st=='exact')[::25]; st[idx]='interval'; lo[idx]=latent[idx]-.05; hi[idx]=latent[idx]+.05; y[idx]=np.nan
    st[5]='missing'; y[5]=np.nan
    with pytest.raises(ValueError): fit_signal(a,y,2,status=st,lower=lo,upper=hi)
    f=fit_signal(a,y,2,status=st,lower=lo,upper=hi,missing='mar')
    np.testing.assert_allclose(f.theta[:2],[.5,1.8],atol=.07); assert np.exp(f.theta[-1])==pytest.approx(.5,abs=.05)
    assert f.size_upper(method='profile_one_sided')>f.size()


def test_repeated_factor():
    rng=np.random.default_rng(6); a=np.exp(np.linspace(-1,2,100)); y=np.log(a)+rng.normal(0,.4,100)
    f=fit_signal(a,y,.8); g=fit_signal(a,y,.8,repeated_factor=4)
    np.testing.assert_allclose(g.cov,4*f.cov); assert g.size_upper()>f.size_upper()


def test_bad_data():
    with pytest.raises(ValueError): fit_signal([0,1,2],[1,2,3],2)
    with pytest.raises(ValueError): fit_hitmiss([1,2,3],[1,1,1])
    with pytest.raises(ValueError): fit_hitmiss([1,1,1,1],[0,0,1,1])
    with pytest.raises(ValueError): fit_signal(np.arange(1,8),np.arange(1,8),2,x_transform='identity')

@pytest.mark.parametrize('family',['normal','lognormal','weibull','exponential','lev'])
def test_noise_threshold(family):
    rng=np.random.default_rng(1)
    d={'normal':stats.norm(3,.5),'lognormal':stats.lognorm(.3,scale=2),'weibull':stats.weibull_min(2,scale=2),'exponential':stats.expon(scale=2),'lev':stats.gumbel_r(2,.5)}[family]
    f=fit_noise(d.rvs(500,random_state=rng),family); assert f.pfa(f.threshold(.01))==pytest.approx(.01,rel=1e-7)


def test_noise_left_censor():
    rng=np.random.default_rng(2); latent=rng.normal(3,.5,1000); st=np.where(latent<2.7,'left','exact'); y=latent.copy(); y[st=='left']=np.nan
    f=fit_noise(y,'normal',st,np.full(len(y),2.7)); np.testing.assert_allclose(f.params,[3,np.log(.5)],atol=.07)


def test_exact_binomial_29_of_29():
    lo,hi=binomial_interval(29,29,one_sided=True); assert lo==pytest.approx(.05**(1/29)); assert hi==1


def test_nuisance_model():
    rng=np.random.default_rng(3); a=np.exp(rng.uniform(-1,2,300)); c=rng.normal(size=(300,1)); y=.5+1.8*np.log(a)+.6*c[:,0]+rng.normal(0,.4,300)
    f=fit_signal(a,y,2,covariates=c); assert f.theta[2]==pytest.approx(.6,abs=.08)
    assert f.size_upper(covariates=[0])>f.size(covariates=[0])


def test_mapod_calibration_and_gp():
    rng=np.random.default_rng(9); x=np.arange(40)/10; cal=calibrate_simulation(x,.3+1.2*x+rng.normal(0,.1,40))
    c=rng.normal(size=(1500,1)); sim=lambda a,c: a[:,None]+c[:,0]
    result=mapod_monte_carlo(sim,np.array([0,1,2]),c,1,calibration=cal)
    assert np.all(np.diff(result['pod'])>0)
    gp=GaussianSurrogate().fit(x[:,None],np.sin(x)); pred,sd=gp.predict(x[::2,None]); np.testing.assert_allclose(pred,np.sin(x[::2]),atol=.03); assert np.all(sd>=0)


def test_random_intercept_and_bayes():
    rng=np.random.default_rng(44); groups=np.repeat(np.arange(20),12); a=np.tile(np.exp(np.linspace(-1,2,12)),20)
    y=.5+1.8*np.log(a)+np.repeat(rng.normal(0,.35,20),12)+rng.normal(0,.3,len(a))
    f=fit_random_intercept(a,y,groups,2); assert np.exp(f.theta[2])==pytest.approx(.3,abs=.07); assert f.size()>0
    b=bayes_linear(a,y,2,draws=1000); assert b['a90_credible_upper95']>0

@pytest.mark.parametrize('xt,yt',[('identity','identity'),('log','identity'),('identity','log'),('log','log')])
def test_all_signal_transform_combinations(xt,yt):
    rng=np.random.default_rng(71); a=np.linspace(.5,3,100); x=np.log(a) if xt=='log' else a
    response=.8+1.3*x+rng.normal(0,.2,100); y=np.exp(response) if yt=='log' else response
    threshold=np.exp(2) if yt=='log' else 2
    f=fit_signal(a,y,threshold,x_transform=xt,y_transform=yt)
    assert f.size_upper()>f.size(); assert f.pod([f.size()])[0]==pytest.approx(.9,abs=1e-10)


def test_diagnostics_and_gof():
    from openpod.diagnostics import randomized_quantile_residuals,grouped_hitmiss_gof,validate_false_alarm
    rng=np.random.default_rng(22); a=np.exp(np.linspace(-1,2,100)); y=.5+1.8*np.log(a)+rng.normal(0,.4,100)
    f=fit_signal(a,y,2); r=randomized_quantile_residuals(f,a,y)
    np.testing.assert_allclose(r,(y-f.theta[0]-f.theta[1]*np.log(a))/np.exp(f.theta[-1]),atol=1e-6)
    hit=rng.binomial(1,special.expit(-2+2.8*np.log(a))); h=fit_hitmiss(a,hit)
    d=grouped_hitmiss_gof(h,a,hit,bootstrap=20); assert 0<d['bootstrap_p']<=1
    p=validate_false_alarm(np.arange(100),95); assert p['false_alarms']==4


def test_legacy_grid_matches_R_example3_logit():
    from pathlib import Path
    from openpod.compat import read_wide_csv,legacy_size_upper
    root=Path(__file__).parents[1]; a,Y,_=read_wide_csv(root/'data/reference/example3.csv',1,[3])
    f=fit_hitmiss(a,Y[:,0],x_transform='identity'); upper=legacy_size_upper(f,np.linspace(a.min(),a.max()*2,101))
    assert upper==pytest.approx(.197340414521742,abs=2e-7)
    np.testing.assert_allclose(f.cov,np.loadtxt(root/'validation/example3_logit_R_cov.csv',delimiter=','),rtol=3e-4)


def test_noise_units_and_threshold_validation():
    rng=np.random.default_rng(73); y=rng.normal(1500,100,500); f=fit_noise(y,'normal')
    assert f.params[0]==pytest.approx(1500,abs=15)
    assert f.threshold(.01)>1500
    with pytest.raises(ValueError): fit_signal(np.arange(1,9),np.arange(1,9)**2,np.nan)
