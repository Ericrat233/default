% Set working folder to repository root; tested with Octave 9.4, MATLAB pending.
addpath('matlab'); result=op_selftest();
D=dlmread('data/generated/censored_numeric.csv',',',1,0);
opts=struct('threshold',2,'status',D(:,3),'lower',D(:,4),'upper',D(:,5),'missing','mar');
model=op_fit('signal',D(:,1),D(:,2),opts);
op_plot(model,D(:,1),'validation/matlab/pod.png');
R=dlmread('data/generated/trace_reference.csv',',',1,0);
out=op_preprocess(R(:,1),struct('baseline_samples',100,'sg_window',9));
op_plot_preprocess(out,'validation/matlab/preprocessing.png');
