"""Reproducible synthetic data, not a validated physical ultrasound simulator."""
from pathlib import Path
import csv
import numpy as np
from scipy import special
from .core import fit_signal,fit_hitmiss
from .signals import TraceConfig,process_traces,process_field,plot_trace_comparison,extract_features,snr_db
from .noise import fit_noise,threshold_tradeoff
from .advanced import calibrate_simulation,mapod_monte_carlo,fit_random_intercept,bayes_linear,bayes_censored,GaussianSurrogate,marginal_pod
from .visualize import plot_pod,plot_field,plot_bscan
from .cli import save_json


def write_csv(path,header,rows):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)


def run_demo(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(1823)
    a=np.exp(np.linspace(-1,2,240)); x=np.log(a); latent=.5+1.8*x+rng.normal(0,.5,len(a)); threshold=2.
    status=np.full(len(a),'exact',dtype='<U8'); lower=np.full(len(a),-np.inf); upper=np.full(len(a),np.inf); y=latent.copy()
    left=latent<0; right=latent>3.7; status[left]='left'; upper[left]=0; status[right]='right'; lower[right]=3.7; y[left|right]=np.nan
    idx=np.arange(20,220,19); idx=idx[status[idx]=='exact']; status[idx]='interval'; lower[idx]=latent[idx]-.1; upper[idx]=latent[idx]+.1; y[idx]=np.nan
    miss=np.array([30,61,98,141,178,220]); status[miss]='missing'; y[miss]=np.nan
    write_csv(output/'censored.csv',['a','y','status','lower','upper'],zip(a,y,status,lower,upper))
    np.savetxt(output/'censored_numeric.csv',np.column_stack((a,y,np.array([['exact','left','right','interval','missing'].index(s) for s in status]),lower,upper)),delimiter=',',header='a,y,status_code,lower,upper',comments='')
    f=fit_signal(a,y,threshold,status=status,lower=lower,upper=upper,missing='mar'); save_json(f.summary(),output/'signal_fit.json'); plot_pod(f,a,output/'pod_signal.png')
    hit=(rng.uniform(size=len(a))<special.expit(-2+2.8*x)).astype(int); write_csv(output/'hitmiss.csv',['a','y'],zip(a,hit))
    binary={}
    for name in ['logit','probit','cloglog','loglog']:
        h=fit_hitmiss(a,hit,link_name=name); binary[name]=h.summary()
    save_json(binary,output/'hitmiss_links.json'); plot_pod(fit_hitmiss(a,hit),a,output/'pod_hitmiss.png')
    noise=rng.lognormal(-1,.3,500); nf=fit_noise(noise,'lognormal'); save_json(nf.summary(),output/'noise_fit.json')
    trade=threshold_tradeoff(f,nf,np.linspace(.4,3.5,35)); np.savetxt(output/'threshold_tradeoff.csv',trade,delimiter=',',header='threshold,threshold_db,pfa,a90,a90_95',comments='')
    n=512; t=np.arange(n); fs=100e6
    reference=np.sin(2*np.pi*.05*t)*np.exp(-.5*((t-250)/16)**2)
    delays=rng.integers(-10,11,48); amp=np.linspace(.3,2,48)
    from .signals import shift_zero
    clean=np.array([amp[i]*shift_zero(reference,int(delays[i])) for i in range(48)])
    raw=clean+.25+.001*t+rng.normal(0,.1,clean.shape)
    cfg=TraceConfig(sample_rate_hz=fs,baseline_samples=100,sg_window=9,align=True,max_shift=14,gate=(150,350),cfar_training=24,cfar_guard=14)
    processed=process_traces(raw,cfg,reference=reference,baseline_mask=(t<100)|(t>=400))
    np.save(output/'bscan_raw.npy',raw); np.save(output/'ascan_raw.npy',raw[0]); np.save(output/'reference.npy',reference)
    np.savetxt(output/'trace_reference.csv',np.column_stack((raw[0],reference)),delimiter=',',header='raw,reference',comments='')
    np.savez_compressed(output/'bscan_processed.npz',processed=processed['processed'],envelope=processed['envelope'],shifts=processed['shifts_samples'])
    plot_trace_comparison(processed,output/'ascan_comparison.png',trace_index=20); plot_bscan(raw,processed['processed'],output/'bscan_comparison.png')
    features=extract_features(processed,(150,350)); write_csv(output/'features.csv',['a','peak','energy','tof_seconds'],zip(amp,features['peak'],features['energy'],features['tof_seconds']))
    # C-scan/cloud amplitude plus spatially coherent background, 3D scalar volume.
    xx,yy=np.meshgrid(np.linspace(-1,1,48),np.linspace(-1,1,48)); field=2*np.exp(-((xx-.2)**2+(yy+.1)**2)/.03)+.2*xx+.15*yy+rng.normal(0,.1,xx.shape)
    c=process_field(field); np.save(output/'cscan_raw.npy',field); plot_field(c,output/'cscan_comparison.png')
    zz=np.linspace(-1,1,24); volume=field[:,:,None]*np.exp(-zz[None,None,:]**2/.1)+rng.normal(0,.02,(48,48,24)); np.save(output/'volume_raw.npy',volume)
    v=process_field(volume,background_sigma=4); plot_field(v,output/'volume_comparison.png')
    groups=np.repeat(np.arange(16),15); ag=np.tile(np.exp(np.linspace(-1,2,15)),16); yg=.5+1.8*np.log(ag)+np.repeat(rng.normal(0,.35,16),15)+rng.normal(0,.3,len(ag))
    write_csv(output/'hierarchical.csv',['a','y','group'],zip(ag,yg,groups)); gf=fit_random_intercept(ag,yg,groups,threshold)
    bf=bayes_linear(a,latent,threshold,draws=4000); mc=bayes_censored(f,draws=1200,burn=700)
    paired=np.linspace(0,5,40); real=.2+1.1*paired+rng.normal(0,.2,40); cal=calibrate_simulation(paired,real)
    nuisance=rng.normal(size=(5000,2)); grid=np.exp(np.linspace(-1,2,70))
    simulator=lambda aa,c: .3+1.6*np.log(aa)[:,None]+.3*c[None,:,0]+.2*c[None,:,1]
    mp=mapod_monte_carlo(simulator,grid,nuisance,threshold,cal,calibration_draws=100)
    np.savetxt(output/'mapod.csv',np.column_stack((grid,mp['pod'],mp['mc_lower'],mp['mc_upper'])),delimiter=',',header='a,pod,mc_lower,mc_upper',comments='')
    training=rng.uniform(-2,2,(80,2)); response=np.sin(training[:,0])+training[:,1]**2; gp=GaussianSurrogate().fit(training,response)
    held=rng.uniform(-1.5,1.5,(30,2)); pred,sd=gp.predict(held); gp_rmse=np.sqrt(np.mean((pred-(np.sin(held[:,0])+held[:,1]**2))**2))
    save_json({'hierarchical':{'theta':gf.theta,'cov':gf.cov,'a90_population':gf.size()},
        'bayesian_exact':{'a90_credible_upper95':bf['a90_credible_upper95']},
        'bayesian_censored':{k:mc[k] for k in ['rhat','ess','acceptance','a90_credible_upper95','converged_diagnostic']},
        'calibration':{'coefficients':cal.coefficients,'discrepancy_sd':cal.discrepancy_sd},'gp_holdout_rmse':gp_rmse,
        'alignment':{'shift_mae_samples':float(np.mean(abs(processed['shifts_samples']+delays)))},
        'data_note':'all generated datasets synthetic; parameter recovery tests do not qualify an NDE system'},output/'advanced_results.json')
