function model=op_random_intercept(a,y,groups,threshold,logx)
% Gaussian marginal ML random intercept; uncensored only.
if nargin<5, logx=true; end
a=a(:); y=y(:); groups=groups(:); ids=unique(groups);
if numel(a)~=numel(y) || numel(a)~=numel(groups) || numel(ids)<4 || any(~isfinite(y)), error('op:group','>=4 independent groups and finite records'); end
x=a; if logx, if any(a<=0), error('op:log','Positive sizes'); end; x=log(a); end
X=[ones(numel(a),1),x]; idx=cell(numel(ids),1);
for i=1:numel(ids), idx{i}=find(groups==ids(i)); if numel(idx{i})<2, error('op:group','>=2/group'); end; end
b=X\y; sd=std(y-X*b,1); fn=@(t)group_nll(t,X,y,idx);
[t,f,flag]=fminsearch(fn,[b;log(sd*.7);log(sd*.7)],optimset('Display','off','MaxIter',10000,'MaxFunEvals',30000,'TolX',1e-9,'TolFun',1e-9));
if flag<=0, error('op:group','Marginal MLE failed'); end
V=op_numeric_cov(fn,t); total=sqrt(exp(2*t(3))+exp(2*t(4)));
q=(threshold-t(1)+op_link(.9,'probit')*total)/t(2); a90=q; if logx, a90=exp(q); end
model=struct('theta',t,'cov',V,'log_likelihood',-f,'group_count',numel(ids),'a90',a90,'threshold',threshold,'logx',logx);
model.pod=@(a)population_pod(a,t,threshold,logx);
end
function f=group_nll(t,X,y,idx)
if any(~isfinite(t)) || max(abs(t(3:4)))>25, f=1e100; return; end
e=exp(2*t(3)); u=exp(2*t(4)); r=y-X*t(1:2); f=0;
for i=1:numel(idx)
 v=r(idx{i}); m=numel(v);
 f=f+.5*(m*log(2*pi)+(m-1)*log(e)+log(e+m*u)+v'*v/e-u*sum(v)^2/(e*(e+m*u)));
end
end
function p=population_pod(a,t,threshold,logx)
x=a; if logx, x=log(a); end
z=(t(1)+t(2)*x-threshold)/sqrt(exp(2*t(3))+exp(2*t(4))); p=.5*erfc(-z/sqrt(2));
end
