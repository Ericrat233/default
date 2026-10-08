"""Small fixed-seed simulation illustration; not regulatory coverage qualification."""
from pathlib import Path
import numpy as np
from scipy import special
from openpod.core import fit_signal,fit_hitmiss
from openpod.cli import save_json
root=Path(__file__).resolve().parents[1]; rng=np.random.default_rng(20261008)
trials=100; true_signal=np.exp((2-.5+special.ndtri(.9)*.5)/1.8); true_binary=np.exp((2+special.logit(.9))/2.8)
records=[]; signal_success=0; binary_success=0; signal_covered=0; binary_covered=0
for i in range(trials):
    a=np.exp(rng.uniform(-1,2,120)); response=.5+1.8*np.log(a)+rng.normal(0,.5,len(a)); st=np.full(len(a),'exact',dtype='<U8'); lo=np.full(len(a),-np.inf); hi=np.full(len(a),np.inf)
    st[response<0]='left'; hi[response<0]=0; st[response>3.7]='right'; lo[response>3.7]=3.7
    try:
        f=fit_signal(a,response,2,status=st,lower=lo,upper=hi); upper=f.size_upper(); signal_success+=1; signal_covered+=upper>=true_signal
    except (ValueError,RuntimeError): upper=np.nan
    hit=rng.binomial(1,special.expit(-2+2.8*np.log(a)))
    try:
        b=fit_hitmiss(a,hit); bound=b.size_upper(); binary_success+=1; binary_covered+=bound>=true_binary
    except (ValueError,RuntimeError): bound=np.nan
    records.append([i,upper,bound])
np.savetxt(root/'validation/coverage_trials.csv',records,delimiter=',',header='trial,signal_wald_upper,binary_lr_df1_upper',comments='')
save_json({'trials':trials,'signal_successful':signal_success,'signal_upper_coverage':signal_covered/signal_success,
           'binary_successful':binary_success,'binary_upper_coverage':binary_covered/binary_success,
           'signal_true_a90':true_signal,'binary_true_a90':true_binary,
           'signal_monte_carlo_se':np.sqrt((signal_covered/signal_success)*(1-signal_covered/signal_success)/signal_success),
           'binary_monte_carlo_se':np.sqrt((binary_covered/binary_success)*(1-binary_covered/binary_success)/binary_success),
           'note':'100 trials only. Signal target upper confidence .95; historical binary df1 LR95 region has approximately .975 one-sided quantile coverage under regularity.'},root/'validation/coverage_summary.json')
