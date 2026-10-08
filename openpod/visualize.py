import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_pod(fit,a,path):
    x=np.linspace(min(a),max(a)*1.5,100) if fit.x_transform=='identity' else np.geomspace(min(a),max(a)*1.5,100)
    fig,ax=plt.subplots(figsize=(8,5)); ax.plot(x,fit.pod(x),label='POD MLE')
    if fit.kind=='hitmiss':
        lo,hi=fit.lr_band(x); ax.fill_between(x,lo,hi,alpha=.2,label='LR envelope, df=1, 95%')
    else:
        p=np.linspace(.001,.999,300); lower,middle,upper=fit.wald_quantile_band(p)
        ax.plot(lower,p,'--',label='Horizontal Wald 95%'); ax.plot(upper,p,'--')
    for name,v in [('a90',fit.size()),('a90/95',fit.size_upper())]:
        if np.isfinite(v): ax.axvline(v,ls=':',label=f'{name}={v:.5g}')
    ax.axhline(.9,color='grey',lw=.7); ax.set_ylim(0,1); ax.set_xlim(min(x),max(x)); ax.set_xlabel('Flaw size (input units)'); ax.set_ylabel('Detection probability')
    if fit.x_transform=='log': ax.set_xscale('log')
    ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)


def plot_field(result,path):
    raw=result['raw']; processed=result['processed']
    if raw.ndim==3:
        raw=np.max(raw,axis=-1); processed=np.max(processed,axis=-1)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    vmax=max(abs(raw).max(),abs(processed).max())
    for ax,v,title in zip(axes,[raw,processed],['Raw scalar field','Processed field']):
        im=ax.imshow(v,cmap='viridis',vmin=-vmax,vmax=vmax); ax.set_title(title); fig.colorbar(im,ax=ax)
    fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)


def plot_bscan(raw,processed,path):
    fig,axes=plt.subplots(1,2,figsize=(10,5)); vmax=max(abs(raw).max(),abs(processed).max())
    for ax,v,title in zip(axes,[raw,processed],['Raw B-scan','Processed B-scan']):
        ax.imshow(v,aspect='auto',cmap='RdBu_r',vmin=-vmax,vmax=vmax); ax.set_title(title); ax.set_xlabel('Time sample'); ax.set_ylabel('Trace')
    fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)
