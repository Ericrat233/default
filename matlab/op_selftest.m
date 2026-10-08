function result=op_selftest(data_directory,output_directory)
% Run on MATLAB R2014b-R2020a or GNU Octave; requires no add-on toolbox.
if nargin<1, data_directory=fullfile(fileparts(mfilename('fullpath')),'..','data','generated'); end
if nargin<2, output_directory=fullfile(fileparts(mfilename('fullpath')),'..','validation','matlab'); end
if ~exist(output_directory,'dir'), mkdir(output_directory); end
D=dlmread(fullfile(data_directory,'censored_numeric.csv'),',',1,0);
opts=struct('threshold',2,'status',D(:,3),'lower',D(:,4),'upper',D(:,5),'missing','mar');
s=op_fit('signal',D(:,1),D(:,2),opts); assert(all(isfinite(s.cov(:))) && s.a90_95>s.a90);
dlmwrite(fullfile(output_directory,'signal_parameters.csv'),s.theta','precision','%.16g'); dlmwrite(fullfile(output_directory,'signal_covariance.csv'),s.cov,'precision','%.16g');
dlmwrite(fullfile(output_directory,'signal_limits.csv'),[s.a50,s.a90,s.a90_95],'precision','%.16g');
B=dlmread(fullfile(data_directory,'hitmiss.csv'),',',1,0); links={'logit','probit','cloglog','loglog'}; values=zeros(4,5);
for i=1:4
 m=op_fit('hitmiss',B(:,1),B(:,2),struct('link',links{i})); values(i,:)=[m.theta',m.a50,m.a90,m.a90_95];
end
dlmwrite(fullfile(output_directory,'hitmiss.csv'),values,'precision','%.16g');
ref=dlmread(fullfile(data_directory,'trace_reference.csv'),',',1,0);
r=op_preprocess(ref(:,1),struct('baseline_samples',100,'baseline_mask',((1:512)<=100)|((1:512)>=401),'sg_window',9,'sample_rate_hz',100e6,'align',true,'gate',[151,350],'max_shift',14,'cfar_training',24,'cfar_guard',14),ref(:,2));
dlmwrite(fullfile(output_directory,'trace.csv'),[r.processed(:),r.envelope(:),r.cfar_threshold(:)],'precision','%.16g');
assert(isequal(op_shift([1;2;3],1),[0;1;2])); assert(abs(op_snr(2*ones(10,1),ones(10,1))-6.020599913)<1e-8);
c=op_cfar(ones(200,1),16,4,.01,'os'); assert(all(~c.detected));
n=op_noise((1:100)'/100+1,'normal'); assert(abs(n.pfa_at(n.threshold)-.01)<1e-8);
cal=op_calibrate((1:6)',2*(1:6)'+1); assert(max(abs(cal.coefficients-[2,1]))<1e-10);
f=op_field(reshape(sin(1:400),20,20)); assert(isequal(size(f.processed),[20,20]));
field3=op_field(reshape(sin(1:2000),20,20,5)); assert(isequal(size(field3.processed),[20,20,5]));
cp=op_binomial(29,29,.95,true); assert(abs(cp(1)-.05^(1/29))<1e-9);
[lower,upper]=op_lr_band(m,[1;2;3]); assert(all(lower<upper));
G=dlmread(fullfile(data_directory,'hierarchical.csv'),',',1,0); ri=op_random_intercept(G(:,1),G(:,2),G(:,3),2);
dlmwrite(fullfile(output_directory,'random_intercept.csv'),[ri.theta',ri.a90],'precision','%.16g');
bayes=op_bayes_linear(G(:,1),G(:,2),2,1200); assert(bayes.a90_credible_upper95>0);
mcmc=op_bayes_censored(s,600,500,4); assert(all(isfinite(mcmc.rhat)));
cal=op_simulation_calibrate((1:20)',.2+1.1*(1:20)'+sin((1:20)')*.1);
mp=op_mapod(@(a,c)bsxfun(@plus,a,c(:,1)'),[1;2;3],randn(200,1),2,cal); assert(all(diff(mp.pod)>=0));
gp=op_gp((1:20)'/10,sin((1:20)'/10),(1:20)'/10); assert(max(abs(gp.mean-sin((1:20)'/10)))<.03);
S=dlmread(fullfile(data_directory,'..','reference','example3.csv'),',',1,0);
history=zeros(4,1);
for i=1:4
 m=op_fit('hitmiss',S(:,2),S(:,4),struct('link',links{i},'x_transform','identity'));
 legacy=op_legacy_band(m,linspace(min(S(:,2)),max(S(:,2))*2,101)'); history(i)=legacy.a_p_conf;
end
dlmwrite(fullfile(output_directory,'legacy_bounds.csv'),history,'precision','%.16g');
result=struct('checks_passed',true,'signal',s,'hitmiss_values',values,'random_intercept',ri,'mcmc_rhat',mcmc.rhat,'runtime_version',version);
save(fullfile(output_directory,'results.mat'),'result'); disp('OpenPOD MATLAB interface selftest PASS');
end
