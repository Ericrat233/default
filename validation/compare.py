"""Run after reference_R.R and MATLAB/Octave selftest."""
from pathlib import Path
import csv,json
import numpy as np
from openpod import fit_signal,fit_hitmiss
from openpod.compat import read_wide_csv,fit_wide_signal,legacy_size_upper
from openpod.cli import save_json
from openpod.signals import process_traces,TraceConfig
root=Path(__file__).resolve().parents[1]; v=root/'validation'; data=root/'data'; models={}
a,Y,_=read_wide_csv(data/'reference/example1.csv',1,[2]); models['example1']=fit_signal(a,Y[:,0],200)
a,Y,_=read_wide_csv(data/'reference/example2.csv',1,[2,3,4,5]); models['example2']=fit_wide_signal(a,Y,250,left=200,right=600)
D=np.genfromtxt(data/'generated/censored_numeric.csv',delimiter=',',skip_header=1); st=np.array(['exact','left','right','interval','missing'])[D[:,2].astype(int)]
models['generated_signal']=fit_signal(D[:,0],D[:,1],2,status=st,lower=D[:,3],upper=D[:,4],missing='mar')
a,Y,_=read_wide_csv(data/'reference/example3.csv',1,[3]); binary_standard={}
for name in ['logit','probit','cloglog','loglog']:
    f=fit_hitmiss(a,Y[:,0],x_transform='identity',link_name=name); models['example3_'+name]=f
    binary_standard[name]=f
B=np.genfromtxt(data/'generated/hitmiss.csv',delimiter=',',skip_header=1)
for name in ['logit','probit','cloglog','loglog']: models['generated_'+name]=fit_hitmiss(B[:,0],B[:,1],link_name=name)
references={r['case']:r for r in csv.DictReader((v/'reference_R_results.csv').open())}
rows=[]; summaries={}; cov_errors={}
for name,f in models.items():
    s=f.summary(); summaries[name]=s; ref=references[name]; refparam=np.array([float(ref['b0']),float(ref['b1'])]+([float(ref['log_sigma'])] if f.kind=='signal' else []))
    covref=np.loadtxt(v/(name+'_R_cov.csv'),delimiter=','); cov_errors[name]=float(np.max(np.abs(f.cov-covref)/np.sqrt(np.diag(covref)[:,None]*np.diag(covref)[None,:])))
    legacy=np.nan
    if name.startswith('example3_'): legacy=legacy_size_upper(f,np.linspace(a.min(),a.max()*2,101))
    rows.append([name,float(np.max(abs(f.theta-refparam))),s['a90'],float(ref['a90']),s['a_p_conf'],float(ref['a90_95']),legacy,
                 (s['a_p_conf']/float(ref['a90_95'])-1)*100, float(cov_errors[name])])
np.savetxt(v/'comparison.csv',np.asarray(rows,object),delimiter=',',fmt='%s',header='case,max_parameter_abs_error,python_a90,R_a90,python_exact_a90_95,R_a90_95,python_legacy_a90_95,exact_bound_difference_percent,max_covariance_normalized_error',comments='')
save_json(summaries,v/'python_fits.json'); save_json({'rows':rows,'covariance_relative_errors':cov_errors},v/'comparison.json')
# MATLAB/Octave parallel implementation comparison.
ms=np.loadtxt(v/'matlab/signal_parameters.csv',delimiter=','); ml=np.loadtxt(v/'matlab/signal_limits.csv',delimiter=','); f=models['generated_signal']
bh=np.loadtxt(v/'matlab/hitmiss.csv',delimiter=','); errors={}
errors['signal_parameter_max_abs']=float(np.max(abs(ms-f.theta)))
errors['signal_cov_max_abs']=float(np.max(abs(np.loadtxt(v/'matlab/signal_covariance.csv',delimiter=',')-f.cov)))
errors['signal_limit_max_relative']=float(np.max(abs(ml/np.array([f.size(.5),f.size(.9),f.size_upper()])-1)))
for i,name in enumerate(['logit','probit','cloglog','loglog']):
    f=models['generated_'+name]; py=np.r_[f.theta,f.size(.5),f.size(.9),f.size_upper()]
    errors['binary_'+name+'_max_abs']=float(np.max(abs(bh[i]-py)))
rawref=np.loadtxt(data/'generated/trace_reference.csv',delimiter=',',skiprows=1)
r=process_traces(rawref[:,0],TraceConfig(baseline_samples=100,sg_window=9,sample_rate_hz=100e6,align=True,gate=(150,350),max_shift=14,cfar_training=24,cfar_guard=14),reference=rawref[:,1],baseline_mask=(np.arange(len(rawref))<100)|(np.arange(len(rawref))>=400))
tc=np.loadtxt(v/'matlab/trace.csv',delimiter=',')
for i,key in enumerate(['processed','envelope','cfar_threshold']): errors['trace_'+key+'_max_abs']=float(np.nanmax(abs(tc[:,i]-r[key])))
if (v/'matlab/legacy_bounds.csv').exists():
    legacy_mat=np.loadtxt(v/'matlab/legacy_bounds.csv',delimiter=',')
    legacy_py=np.array([float(row[6]) for row in rows if row[0].startswith('example3_')])
    errors['legacy_bound_matlab_python_max_abs']=float(np.max(abs(legacy_py-legacy_mat)))
    legacy_r=np.array([float(row[5]) for row in rows if row[0].startswith('example3_')])
    errors['legacy_bound_matlab_R_max_abs']=float(np.max(abs(legacy_r-legacy_mat)))
save_json(errors,v/'python_matlab_comparison.json'); print(json.dumps(errors,indent=2))
assert errors['signal_parameter_max_abs']<1e-5
assert errors['signal_limit_max_relative']<1e-5
assert all(errors['binary_'+name+'_max_abs']<1e-4 for name in ['logit','probit','cloglog','loglog'])
assert errors['trace_processed_max_abs']<1e-8
assert all(float(row[1])<1e-3 for row in rows)
