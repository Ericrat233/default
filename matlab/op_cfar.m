function out=op_cfar(power,training,guard,pfa,method,rank_value)
if nargin<2, training=16; end
if nargin<3, guard=4; end
if nargin<4, pfa=.01; end
if nargin<5, method='ca'; end
power=power(:); N=2*training;
if nargin<6, rank_value=ceil(.75*N); end
if training<1 || guard<0 || pfa<=0 || pfa>=1 || any(power<0) || any(~isfinite(power)), error('op:cfar','Invalid CFAR data/parameters'); end
if strcmp(method,'ca'), alpha=N*(pfa^(-1/N)-1);
elseif strcmp(method,'os')
 if rank_value<1 || rank_value>N, error('op:cfar','OS rank'); end
 fn=@(a)gammaln(N+1)-gammaln(N-rank_value+1)+gammaln(N-rank_value+a+1)-gammaln(N+a+1)-log(pfa);
 hi=1; while fn(hi)>0, hi=2*hi; end
 alpha=fzero(fn,[0,hi]);
else, error('op:cfar','Use ca/os'); end
threshold=NaN(size(power)); margin=training+guard;
for i=margin+1:numel(power)-margin
 cells=[power(i-margin:i-guard-1);power(i+guard+1:i+margin)];
 if strcmp(method,'ca'), noise=mean(cells); else, cells=sort(cells); noise=cells(rank_value); end
 threshold(i)=alpha*noise;
end
out=struct('threshold',threshold,'detected',power>threshold,'valid',isfinite(threshold),'alpha',alpha);
end
