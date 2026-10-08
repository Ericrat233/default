"""Auditable trace pipeline and spatial image/volume preprocessing.
Time sample axis is explicit; a C-scan amplitude image has no time axis.
CFAR formula assumes independent exponential power cells, not correlated RF.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy import signal, ndimage, optimize, special

@dataclass
class TraceConfig:
    axis: int = -1
    baseline_order: int = 1
    baseline_samples: int = 50
    sg_window: int = 9
    sg_order: int = 3
    band_hz: object = None
    sample_rate_hz: float = 1.
    background: str = 'none'  # none/median_trace/reference
    align: bool = False
    max_shift: int = 20
    gate: object = None
    normalize: str = 'none'  # none/reference_rms/peak
    cfar_training: int = 16  # per side
    cfar_guard: int = 4
    cfar_pfa: float = .01
    cfar_method: str = 'ca'


def snr_db(signal_values, noise_values, amplitude=False):
    s=np.asarray(signal_values,float); n=np.asarray(noise_values,float)
    if not s.size or not n.size or not np.isfinite(s).all() or not np.isfinite(n).all(): raise ValueError('finite nonempty SNR windows required')
    if amplitude:
        num=np.max(np.abs(s)); den=np.sqrt(np.mean(n*n)); multiplier=20
    else:
        num=np.mean(s*s); den=np.mean(n*n); multiplier=10
    if den<=0: raise ValueError('zero noise power')
    with np.errstate(divide='ignore'): return float(multiplier*np.log10(num/den))


def db(values,reference=1.,power=False):
    if reference<=0: raise ValueError('positive reference required')
    x=np.asarray(values,float)
    if np.any(x<0): raise ValueError('dB input must be magnitude or power')
    with np.errstate(divide='ignore'): return (10 if power else 20)*np.log10(x/reference)


def shift_zero(x,shift):
    """Integer positive delay, negative advance, without circular wraparound."""
    out=np.zeros_like(x)
    if abs(shift)>=len(x): return out
    if shift>0: out[shift:]=x[:-shift]
    elif shift<0: out[:shift]=x[-shift:]
    else: out[:]=x
    return out


def cfar(power,training=16,guard=4,pfa=.01,method='ca',rank=None):
    x=np.asarray(power,float)
    if x.ndim!=1 or np.any(x<0) or not np.isfinite(x).all(): raise ValueError('CFAR requires finite nonnegative 1D power')
    if training<1 or guard<0 or not 0<pfa<1: raise ValueError('invalid CFAR parameters')
    N=2*training; rank=int(np.ceil(.75*N)) if rank is None else int(rank)
    if method=='ca': alpha=N*(pfa**(-1/N)-1)
    elif method=='os':
        if not 1<=rank<=N: raise ValueError('OS rank out of range')
        def f(alpha):
            return special.gammaln(N+1)-special.gammaln(N-rank+1)+special.gammaln(N-rank+alpha+1)-special.gammaln(N+alpha+1)-np.log(pfa)
        hi=1.
        while f(hi)>0: hi*=2
        alpha=optimize.brentq(f,0,hi)
    else: raise ValueError('CFAR method ca/os')
    threshold=np.full_like(x,np.nan); margin=training+guard
    for i in range(margin,len(x)-margin):
        cells=np.r_[x[i-margin:i-guard],x[i+guard+1:i+margin+1]]
        noise=cells.mean() if method=='ca' else np.partition(cells,rank-1)[rank-1]
        threshold[i]=alpha*noise
    return {'threshold':threshold,'detected':x>threshold,'alpha':float(alpha),'valid':np.isfinite(threshold),
            'assumption':'independent exponential power; verify empirical PFA on held-out background'}


def cfar_nd(power,training=3,guard=1,pfa=.01):
    """Rectangular CA-CFAR on 2D C-scan or 3D power volume, valid interior only."""
    x=np.asarray(power,float)
    if x.ndim not in (2,3) or not np.isfinite(x).all() or np.any(x<0): raise ValueError('2D/3D nonnegative power required')
    if training<1 or guard<0 or not 0<pfa<1: raise ValueError('invalid CFAR parameters')
    m=training+guard; k=np.ones((2*m+1,)*x.ndim)
    k[(slice(training,training+2*guard+1),)*x.ndim]=0
    N=int(k.sum()); alpha=N*(pfa**(-1/N)-1)
    th=alpha*ndimage.convolve(x,k,mode='constant',cval=0)/N
    valid=np.zeros(x.shape,bool)
    if all(d>2*m for d in x.shape): valid[(slice(m,-m),)*x.ndim]=True
    th[~valid]=np.nan
    return {'threshold':th,'detected':x>th,'valid':valid,'alpha':float(alpha)}


def process_traces(data,config=None,reference=None,baseline_mask=None,calibration_coeff=None):
    cfg=config or TraceConfig(); raw=np.asarray(data,float)
    if raw.ndim<1 or not np.isfinite(raw).all(): raise ValueError('finite RF trace array required')
    if cfg.sample_rate_hz<=0 or cfg.max_shift<0: raise ValueError('invalid sampling/shift')
    moved=np.moveaxis(raw,cfg.axis,-1); shape=moved.shape; n=shape[-1]; rows=moved.reshape(-1,n).copy()
    if cfg.baseline_samples<cfg.baseline_order+2: raise ValueError('baseline fit needs more samples than coefficients')
    mask=np.arange(n)<min(cfg.baseline_samples,n) if baseline_mask is None else np.asarray(baseline_mask,bool)
    if mask.shape!=(n,) or mask.sum()<cfg.baseline_order+2: raise ValueError('valid baseline mask required')
    t=np.linspace(-1,1,n); baseline=np.empty_like(rows)
    for i,row in enumerate(rows): baseline[i]=np.polyval(np.polyfit(t[mask],row[mask],cfg.baseline_order),t)
    rows-=baseline
    if cfg.sg_window:
        if cfg.sg_window%2!=1 or cfg.sg_window<=cfg.sg_order or cfg.sg_window>n: raise ValueError('odd SG window > order and <= trace length required')
        rows=signal.savgol_filter(rows,cfg.sg_window,cfg.sg_order,axis=-1)
    if cfg.band_hz is not None:
        low,high=cfg.band_hz
        if not 0<low<high<cfg.sample_rate_hz/2: raise ValueError('band must lie below Nyquist')
        sos=signal.butter(4,[low,high],btype='bandpass',fs=cfg.sample_rate_hz,output='sos')
        rows=signal.sosfiltfilt(sos,rows,axis=-1)
    ref=None if reference is None else np.asarray(reference,float)
    if ref is not None and (ref.shape!=(n,) or not np.isfinite(ref).all()): raise ValueError('reference must have n samples')
    if cfg.background=='median_trace':
        if len(rows)<3: raise ValueError('median_trace background requires at least three traces')
        rows-=np.median(rows,axis=0)
    elif cfg.background=='reference':
        if ref is None: raise ValueError('background reference required')
        rows-=ref
    elif cfg.background!='none': raise ValueError('invalid background mode')
    if calibration_coeff is not None: rows=np.polyval(calibration_coeff,rows)
    gate=(0,n) if cfg.gate is None else tuple(cfg.gate)
    if not 0<=gate[0]<gate[1]<=n: raise ValueError('gate out of bounds')
    shifts=np.zeros(len(rows),int)
    if cfg.align:
        if ref is None: raise ValueError('alignment needs external processed RF reference')
        r=ref[gate[0]:gate[1]]; lags=signal.correlation_lags(len(r),len(r))
        keep=np.abs(lags)<=cfg.max_shift
        for i in range(len(rows)):
            c=signal.correlate(rows[i,gate[0]:gate[1]],r,method='fft')
            lag=int(lags[keep][np.argmax(c[keep])]); shifts[i]=-lag; rows[i]=shift_zero(rows[i],-lag)
    factors=np.ones(len(rows))
    if cfg.normalize=='peak': factors=np.max(np.abs(rows[:,gate[0]:gate[1]]),axis=1)
    elif cfg.normalize=='reference_rms':
        if ref is None: raise ValueError('normalization reference required')
        factors[:]=np.sqrt(np.mean(ref**2))
    elif cfg.normalize!='none': raise ValueError('invalid normalization')
    if np.any(factors<=0): raise ValueError('normalization divisor must be positive')
    rows/=factors[:,None]
    envelope=np.abs(signal.hilbert(rows,axis=-1)); th=np.empty_like(rows); detections=np.zeros(rows.shape,bool); packets=[]
    for i in range(len(rows)):
        cf=cfar(envelope[i]**2,cfg.cfar_training,cfg.cfar_guard,cfg.cfar_pfa,cfg.cfar_method)
        th[i]=np.sqrt(cf['threshold']); detections[i]=cf['detected']
        labels,count=ndimage.label(detections[i])
        current=[]
        for label in range(1,count+1):
            idx=np.flatnonzero(labels==label); peak=int(idx[np.argmax(envelope[i,idx])])
            if gate[0]<=peak<gate[1]: current.append({'start':int(idx[0]),'stop':int(idx[-1])+1,'peak':peak,'amplitude':float(envelope[i,peak])})
        packets.append(current)
    def restore(x): return np.moveaxis(x.reshape(shape),-1,cfg.axis)
    return {'raw':raw,'processed':restore(rows),'baseline':restore(baseline),'envelope':restore(envelope),
            'cfar_threshold':restore(th),'detections':restore(detections),'packets':packets,
            'baseline_mask':mask,'shifts_samples':shifts.reshape(shape[:-1]),'normalization_factors':factors.reshape(shape[:-1]),'config':asdict(cfg)}


def process_field(data,median_size=3,background_sigma=6.,cfar_training=3,cfar_guard=1,pfa=.01):
    """Scalar C-scan / simulation cloud / volume; local background, no Hilbert."""
    x=np.asarray(data,float)
    if x.ndim not in (2,3) or not np.isfinite(x).all(): raise ValueError('finite scalar 2D/3D field required')
    smooth=ndimage.median_filter(x,size=median_size)
    background=ndimage.gaussian_filter(smooth,background_sigma)
    y=smooth-background; cf=cfar_nd(y**2,cfar_training,cfar_guard,pfa)
    labels,n=ndimage.label(cf['detected']); objects=[]
    for k in range(1,n+1):
        idx=np.argwhere(labels==k); vals=y[labels==k]
        objects.append({'label':k,'voxels':len(idx),'centroid':idx.mean(0).tolist(),'peak_abs':float(np.max(np.abs(vals)))})
    return {'raw':x,'processed':y,'background':background,'cfar':cf,'labels':labels,'objects':objects}


def calibrate(measured,standard,degree=1):
    x=np.asarray(measured,float); y=np.asarray(standard,float)
    if x.ndim!=1 or x.shape!=y.shape or len(x)<=degree+1 or not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError('finite paired calibration standards, n > degree+1')
    coeff,cov=np.polyfit(x,y,degree,cov=True)
    return {'coefficients':coeff,'covariance':cov,'rmse':float(np.sqrt(np.mean((np.polyval(coeff,x)-y)**2)))}


def extract_features(result,gate):
    cfg=result['config']; env=np.moveaxis(result['envelope'],cfg['axis'],-1)[...,gate[0]:gate[1]]
    return {'peak':np.max(env,axis=-1),'energy':np.sum(env**2,axis=-1)/cfg['sample_rate_hz'],
            'tof_seconds':(np.argmax(env,axis=-1)+gate[0])/cfg['sample_rate_hz']}


def plot_trace_comparison(result,path,trace_index=0):
    import matplotlib.pyplot as plt
    cfg=result['config']; n=np.moveaxis(result['raw'],cfg['axis'],-1).shape[-1]; t=np.arange(n)/cfg['sample_rate_hz']
    fig,ax=plt.subplots(2,1,figsize=(10,6),sharex=True)
    for key in ('raw','processed'):
        v=np.moveaxis(result[key],cfg['axis'],-1).reshape(-1,n)[trace_index]; ax[0].plot(t,v,label=key)
    for key in ('envelope','cfar_threshold'):
        v=np.moveaxis(result[key],cfg['axis'],-1).reshape(-1,n)[trace_index]; ax[1].plot(t,v,label=key)
    ax[0].set_ylabel('RF amplitude'); ax[1].set_ylabel('Envelope'); ax[1].set_xlabel('Time (s)')
    for a in ax: a.legend(); a.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)
