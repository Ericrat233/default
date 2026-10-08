"""CSV fit / NPY preprocess / bundled demo command line."""
import argparse, csv, json
from pathlib import Path
import numpy as np
from .core import fit_signal,fit_hitmiss
from .signals import TraceConfig,process_traces,process_field,plot_trace_comparison


def load_csv(path):
    with open(path,encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))


def save_json(value,path):
    def convert(o):
        if isinstance(o,np.ndarray): return o.tolist()
        if isinstance(o,np.generic): return o.item()
        raise TypeError(type(o).__name__)
    # Infinity/NaN replaced by null; keep JSON interoperable across languages.
    def clean(v):
        if isinstance(v,dict): return {k:clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)): return [clean(x) for x in v]
        if isinstance(v,np.ndarray): return clean(v.tolist())
        if isinstance(v,(float,np.floating)) and not np.isfinite(v): return None
        return v
    Path(path).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,default=convert),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description='OpenPOD: auditable NDE reliability')
    sub=ap.add_subparsers(dest='command',required=True)
    fit=sub.add_parser('fit'); fit.add_argument('csv'); fit.add_argument('--kind',choices=['signal','hitmiss'],default='signal')
    fit.add_argument('--threshold',type=float,default=1); fit.add_argument('--x-transform',choices=['log','identity'],default='log')
    fit.add_argument('--y-transform',choices=['log','identity'],default='identity'); fit.add_argument('--link',default='logit')
    fit.add_argument('--missing',choices=['error','mar'],default='error'); fit.add_argument('--repeated-factor',type=float,default=1)
    fit.add_argument('--out',default='fit.json'); fit.add_argument('--plot',default='pod.png')
    prep=sub.add_parser('preprocess'); prep.add_argument('npy'); prep.add_argument('--mode',choices=['traces','field'],default='traces')
    prep.add_argument('--config'); prep.add_argument('--out',default='processed.npz'); prep.add_argument('--plot',default='preprocessing.png')
    demo=sub.add_parser('demo'); demo.add_argument('--out',default='demo_output')
    args=ap.parse_args()
    if args.command=='fit':
        records=load_csv(args.csv)
        number=lambda r,k:float(r.get(k,'nan') or 'nan')
        a=np.array([number(r,'a') for r in records]); y=np.array([number(r,'y') for r in records])
        if args.kind=='hitmiss': f=fit_hitmiss(a,y,x_transform=args.x_transform,link_name=args.link,repeated_factor=args.repeated_factor)
        else:
            kw={}
            if 'status' in records[0]:
                kw['status']=[r['status'] for r in records]
                kw['lower']=[number(r,'lower') for r in records]; kw['upper']=[number(r,'upper') for r in records]
            f=fit_signal(a,y,args.threshold,x_transform=args.x_transform,y_transform=args.y_transform,missing=args.missing,repeated_factor=args.repeated_factor,**kw)
        save_json(f.summary(),args.out)
        from .visualize import plot_pod
        plot_pod(f,a,args.plot)
        print(args.out,args.plot)
    elif args.command=='preprocess':
        data=np.load(args.npy,allow_pickle=False)
        config={} if not args.config else json.loads(Path(args.config).read_text())
        if args.mode=='traces':
            result=process_traces(data,TraceConfig(**config)); plot_trace_comparison(result,args.plot)
            np.savez_compressed(args.out,**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
        else:
            result=process_field(data,**config)
            np.savez_compressed(args.out,raw=data,processed=result['processed'],background=result['background'],detections=result['cfar']['detected'])
            from .visualize import plot_field
            plot_field(result,args.plot)
        print(args.out,args.plot)
    else:
        from .demo import run_demo
        run_demo(Path(args.out)); print(args.out)

if __name__=='__main__': main()
