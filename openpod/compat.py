"""mh1823-style wide CSV adapters and optional historical 51x51 LR contour.
The latter is intentionally grid dependent. Prefer exact core profile/envelope
for new analyses; use this only to investigate old tool plotting differences.
"""
import csv
import numpy as np
from .core import fit_signal,fit_hitmiss,link,transform,inverse


def read_wide_csv(path,size_column,response_columns):
    with open(path,encoding='utf-8-sig',newline='') as f:
        table=list(csv.reader(f))
    labels=table[0]; rows=table[1:]
    def index(v): return labels.index(v) if isinstance(v,str) else v
    si=index(size_column); ri=[index(c) for c in response_columns]
    def number(v): return np.nan if v in ('','NA','NaN','nan') else float(v)
    a=np.array([number(r[si]) for r in rows]); Y=np.array([[number(r[i]) for i in ri] for r in rows])
    return a,Y,labels


def fit_wide_signal(a,Y,threshold,left=-np.inf,right=np.inf,**kwargs):
    """Historical repeated adjustment factor=m for a balanced wide matrix.
    This is a compatibility correction, not a random effects estimator.
    """
    Y=np.asarray(Y,float); m=Y.shape[1]; aa=np.tile(a,m); y=Y.reshape(-1,order='F')
    st=np.full(len(y),'exact',dtype='<U8'); st[y<=left]='left'; st[y>=right]='right'; st[np.isnan(y)]='missing'
    return fit_signal(aa,y,threshold,status=st,lower=np.full(len(y),right),upper=np.full(len(y),left),repeated_factor=m,**kwargs)


def legacy_lr_band(fit,a_grid,confidence=.95,simultaneous=False):
    """Independent implementation of historical rotated grid + linear contour.
    95% uses rounded log-LR drop 1.9207; joint region drop 2.996. Input a_grid
    should have 101 equally spaced TRANSFORMED sizes to mimic plotting exactly.
    """
    if fit.kind!='hitmiss' or fit.covariate_count: raise ValueError('baseline binary fit required')
    if confidence not in (.9,.95): raise ValueError('historical 90/95 contours only')
    cov=fit.cov; t=fit.theta; angle=np.arctan(cov[0,1]/cov[1,1]); c=np.cos(angle); s=np.sin(angle)
    R=np.array([[c,s],[-s,c]]); df=2
    from scipy import stats
    distance=np.sqrt(stats.chi2.ppf(.999 if simultaneous else .99,df))
    yy=np.linspace(-np.sqrt(cov[1,1])*distance,np.sqrt(cov[1,1])*distance,101)
    root=np.sqrt(np.abs(np.linalg.det(cov)/cov[1,1]**2*(cov[1,1]*distance**2-yy**2))); root[[0,-1]]=0
    center=cov[0,1]/cov[1,1]*yy
    ellipse=np.r_[np.column_stack((center-root,yy)),np.column_stack((center[::-1]+root[::-1],yy[::-1]))]@R
    bx=np.linspace(ellipse[:,0].min(),ellipse[:,0].max(),51); by=np.linspace(ellipse[:,1].min(),ellipse[:,1].max(),51)
    gx,gy=np.meshgrid(bx,by); pairs=np.column_stack((gx.ravel(),gy.ravel()))@R.T+t
    ll=np.array([-fit.nll(p)/fit.repeated_factor for p in pairs]).reshape(gx.shape)
    drop=1.9207 if confidence==.95 else 1.353
    if simultaneous: drop=stats.chi2.ppf(stats.chi2.cdf(2*drop,1),2)/2
    level=np.round(-fit.nll(t)/fit.repeated_factor-drop,4)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(); con=ax.contour(bx,by,ll,levels=[level]); vertices=np.concatenate(con.allsegs[0]); plt.close(fig)
    beta=vertices@R.T+t; eta=beta[:,0,None]+beta[:,1,None]*transform(a_grid,fit.x_transform)[None,:]
    from .core import link_logs
    return np.exp(link_logs(eta.min(0),fit.link_name)[0]),np.exp(link_logs(eta.max(0),fit.link_name)[0]),beta


def legacy_size_upper(fit,a_grid,p=.9,confidence=.95,simultaneous=False):
    lower,upper,beta=legacy_lr_band(fit,a_grid,confidence,simultaneous)
    x=transform(a_grid,fit.x_transform)
    if not lower.min()<=p<=lower.max(): return np.nan
    # Reference interpolates eta, not POD.
    eta=(beta[:,0,None]+beta[:,1,None]*x[None,:]).min(0); q=np.interp(link(p,fit.link_name),eta,x)
    return float(inverse(q,fit.x_transform))
