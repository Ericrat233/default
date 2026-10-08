function out=op_field(data,median_size,sigma,training,guard,pfa)
% Scalar 2D/3D C-scan/cloud/volume; local median + separable Gaussian.
if nargin<2, median_size=3; end
if nargin<3, sigma=6; end
if nargin<4, training=3; end
if nargin<5, guard=1; end
if nargin<6, pfa=.01; end
d=ndims(data); shape=size(data);
if d>3 || d<2 || any(~isfinite(data(:))) || mod(median_size,2)~=1 || sigma<=0 || training<1 || guard<0 || pfa<=0 || pfa>=1, error('op:field','Invalid 2D/3D field parameters'); end
m=(median_size-1)/2; smooth=zeros(shape);
for i=1:numel(data)
 if d==2, [i1,i2]=ind2sub(shape,i); i3=1; else, [i1,i2,i3]=ind2sub(shape,i); end
 a=max(1,i1-m):min(shape(1),i1+m); b=max(1,i2-m):min(shape(2),i2+m);
 if d==2, v=data(a,b); else, c=max(1,i3-m):min(shape(3),i3+m); v=data(a,b,c); end
 smooth(i)=median(v(:));
end
bg=smooth; radius=ceil(3*sigma); g=exp(-((-radius:radius).^2)/(2*sigma^2)); g=g/sum(g);
for dim=1:d
 dims=ones(1,d); dims(dim)=numel(g); kernel=reshape(g,dims);
 normweight=convn(ones(shape),kernel,'same'); bg=convn(bg,kernel,'same')./normweight;
end
processed=smooth-bg; power=processed.^2; margin=training+guard;
k=ones(repmat(2*margin+1,1,d)); region=training+1:training+2*guard+1;
if d==2, k(region,region)=0; else, k(region,region,region)=0; end
N=sum(k(:)); alpha=N*(pfa^(-1/N)-1); threshold=alpha*convn(power,k,'same')/N;
valid=false(shape);
if all(shape>2*margin)
 if d==2, valid(margin+1:end-margin,margin+1:end-margin)=true;
 else, valid(margin+1:end-margin,margin+1:end-margin,margin+1:end-margin)=true; end
end
threshold(~valid)=NaN;
out=struct('raw',data,'processed',processed,'background',bg,'threshold',threshold,'detected',power>threshold,'valid',valid,'alpha',alpha);
end
