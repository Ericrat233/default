function out=op_preprocess(data,opts,reference)
% RF processing. Default time dimension LAST; first sample index is 1.
% No signal toolbox: local polynomial SG, FFT analytic envelope, gated alignment.
if nargin<2, opts=struct(); end
if nargin<3, reference=[]; end
if any(~isfinite(data(:))), error('op:signal','Finite RF required'); end
dim=getopt(opts,'time_dim',ndims(data));
if isvector(data), data=data(:); dim=1; end
order=[setdiff(1:ndims(data),dim),dim]; x=permute(data,order); shape=size(x); n=shape(end); rows=reshape(x,[],n);
bo=getopt(opts,'baseline_order',1); bn=getopt(opts,'baseline_samples',50); mask=getopt(opts,'baseline_mask',(1:n)<=min(bn,n)); mask=logical(mask(:)');
if numel(mask)~=n || sum(mask)<bo+2, error('op:baseline','Need baseline samples'); end
t=linspace(-1,1,n); baseline=zeros(size(rows));
for i=1:size(rows,1), baseline(i,:)=polyval(polyfit(t(mask),rows(i,mask),bo),t); end
rows=rows-baseline;
w=getopt(opts,'sg_window',9); degree=getopt(opts,'sg_order',3);
if w>0
 if mod(w,2)~=1 || w>n || w<=degree, error('op:SG','Odd SG window > degree and <= n'); end
 smooth=zeros(size(rows)); m=(w-1)/2;
 for j=1:n
  start=max(1,min(j-m,n-w+1)); idx=start:start+w-1; local=idx-j;
  A=zeros(w,degree+1); for k=0:degree, A(:,k+1)=local(:).^k; end
  P=pinv(A); smooth(:,j)=rows(:,idx)*P(1,:)';
 end
 rows=smooth;
end
fs=getopt(opts,'sample_rate_hz',1); if fs<=0, error('op:fs','Positive sample rate'); end
band=getopt(opts,'band_hz',[]);
if ~isempty(band)
 if numel(band)~=2 || band(1)<=0 || band(2)<=band(1) || band(2)>=fs/2, error('op:band','Band below Nyquist'); end
 freq=(0:n-1)*fs/n; freq=min(freq,fs-freq); pass=freq>=band(1) & freq<=band(2);
 rows=real(ifft(bsxfun(@times,fft(rows,[],2),pass),[],2));
end
bg=getopt(opts,'background','none'); reference=reference(:)';
if ~isempty(reference) && (numel(reference)~=n || any(~isfinite(reference))), error('op:reference','Reference length/values'); end
if strcmp(bg,'median_trace')
 if size(rows,1)<3, error('op:background','Median background needs >=3 traces'); end
 rows=bsxfun(@minus,rows,median(rows,1));
elseif strcmp(bg,'reference')
 if isempty(reference), error('op:reference','Need background reference'); end
 rows=bsxfun(@minus,rows,reference);
elseif ~strcmp(bg,'none'), error('op:background','Invalid background'); end
coef=getopt(opts,'calibration_coeff',[]); if ~isempty(coef), rows=polyval(coef,rows); end
gate=getopt(opts,'gate',[1,n]); maxshift=getopt(opts,'max_shift',20); shifts=zeros(size(rows,1),1);
if numel(gate)~=2 || gate(1)<1 || gate(2)>n || gate(1)>gate(2), error('op:gate','Gate inclusive [first,last]'); end
if getopt(opts,'align',false)
 if isempty(reference), error('op:reference','Alignment reference required'); end
 ref=reference(gate(1):gate(2));
 for i=1:size(rows,1)
  best=-Inf; lagbest=0;
  for lag=-maxshift:maxshift
   shifted=op_shift(rows(i,gate(1):gate(2))',-lag); val=shifted'*ref';
   if val>best, best=val; lagbest=lag; end
  end
  shifts(i)=-lagbest; rows(i,:)=op_shift(rows(i,:)',-lagbest)';
 end
end
mode=getopt(opts,'normalize','none'); factors=ones(size(rows,1),1);
if strcmp(mode,'peak'), factors=max(abs(rows(:,gate(1):gate(2))),[],2);
elseif strcmp(mode,'reference_rms')
 if isempty(reference), error('op:reference','Normalization reference required'); end
 factors(:)=sqrt(mean(reference.^2));
elseif ~strcmp(mode,'none'), error('op:normalize','Invalid normalization'); end
if any(factors<=0), error('op:normalize','Zero divisor'); end
rows=bsxfun(@rdivide,rows,factors);
h=zeros(1,n); h(1)=1;
if mod(n,2)==0, h(2:n/2)=2; h(n/2+1)=1; else, h(2:(n+1)/2)=2; end
envelope=abs(ifft(bsxfun(@times,fft(rows,[],2),h),[],2));
threshold=zeros(size(rows)); detected=false(size(rows)); packets=cell(size(rows,1),1);
for i=1:size(rows,1)
 cf=op_cfar(envelope(i,:).^2,getopt(opts,'cfar_training',16),getopt(opts,'cfar_guard',4),getopt(opts,'cfar_pfa',.01),getopt(opts,'cfar_method','ca'));
 threshold(i,:)=sqrt(cf.threshold)'; detected(i,:)=cf.detected';
 edges=diff([false,detected(i,:),false]); starts=find(edges==1); stops=find(edges==-1)-1; pack=[];
 for k=1:numel(starts)
  idx=starts(k):stops(k); [amp,where]=max(envelope(i,idx)); peak=idx(where);
  if peak>=gate(1) && peak<=gate(2), pack=[pack; starts(k),stops(k),peak,amp]; end
 end
 packets{i}=pack;
end
out=struct('raw',data,'processed',ipermute(reshape(rows,shape),order),'baseline',ipermute(reshape(baseline,shape),order),'envelope',ipermute(reshape(envelope,shape),order),'cfar_threshold',ipermute(reshape(threshold,shape),order),'detections',ipermute(reshape(detected,shape),order),'baseline_mask',mask,'shifts_samples',shifts,'normalization_factors',factors,'packets',{packets},'options',opts);
end
function v=getopt(s,name,default)
if isfield(s,name), v=s.(name); else, v=default; end
end
