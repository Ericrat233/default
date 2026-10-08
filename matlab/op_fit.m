function model = op_fit(kind,a,y,opts)
% OP_FIT Independent POD MLE, base MATLAB, R2014b syntax.
% kind='signal' or 'hitmiss'; opts fields documented in README.
% Status: 0 exact,1 left,2 right,3 interval,4 missing; raw bounds.
if nargin<4, opts=struct(); end
a=a(:); y=y(:); n=numel(a);
if numel(y)~=n || any(~isfinite(a)), error('op:data','Invalid size/response'); end
xt=getopt(opts,'x_transform','log'); yt=getopt(opts,'y_transform','identity');
link=getopt(opts,'link','logit'); threshold=getopt(opts,'threshold',0);
repeat=getopt(opts,'repeated_factor',1);
if repeat<1, error('op:data','repeated_factor >=1'); end
x=trans(a,xt); X=[ones(n,1),x];
C=getopt(opts,'covariates',[]);
if ~isempty(C)
 if size(C,1)~=n || any(~isfinite(C(:))), error('op:data','Covariates'); end
 X=[X,C];
end
if rank(X)<size(X,2), error('op:data','Rank deficient design'); end
k=size(X,2); excluded=0;
if strcmp(kind,'signal')
 if ~isfinite(threshold), error('op:threshold','Finite threshold required'); end
 st=getopt(opts,'status',zeros(n,1)); st=st(:);
 if numel(st)~=n || any(~ismember(st,0:4)), error('op:data','Status must be 0..4'); end
 if ~isfield(opts,'status'), st(isnan(y))=4; end
 missing=(st==4); excluded=sum(missing);
 if excluded>0 && ~strcmp(getopt(opts,'missing','error'),'mar'), error('op:missing','Explicit missing=mar required'); end
 lo=getopt(opts,'lower',-Inf(n,1)); hi=getopt(opts,'upper',Inf(n,1));
 if isscalar(lo), lo=repmat(lo,n,1); end
 if isscalar(hi), hi=repmat(hi,n,1); end
 lo=lo(:); hi=hi(:);
 if numel(lo)~=n || numel(hi)~=n, error('op:data','Bounds length'); end
 exact=st==0; left=st==1; right=st==2; interval=st==3;
 if any(~isfinite(y(exact))) || any(~isfinite(hi(left|interval))) || any(~isfinite(lo(right|interval))) || any(lo(interval)>=hi(interval))
  error('op:data','Invalid censor bounds or exact response');
 end
 v=zeros(n,1); v(exact)=trans(y(exact),yt);
 L=-Inf(n,1); U=Inf(n,1); L(right|interval)=trans(lo(right|interval),yt); U(left|interval)=trans(hi(left|interval),yt);
 keep=~missing; X=X(keep,:); v=v(keep); L=L(keep); U=U(keep); st=st(keep);
 exact=st==0;
 if sum(exact)<k || size(X,1)<=k+1 || rank(X)<k, error('op:data','Insufficient identified records'); end
 b=X(exact,:)\v(exact); residual=v(exact)-X(exact,:)*b; sd=sqrt(mean((residual-mean(residual)).^2));
 if sd<1e-10, error('op:data','Zero residual spread'); end
 start=[b;log(sd)]; fn=@(t)signal_nll(t,X,v,L,U,st);
 theta=solve(fn,start); V=info_cov(fn,theta)*repeat;
 diagnostics=struct('n_exact',sum(st==0),'n_left',sum(st==1),'n_right',sum(st==2),'n_interval',sum(st==3),'n_missing_excluded',excluded);
 diagnostics.residuals_exact=v(exact)-X(exact,:)*theta(1:end-1);
 diagnostics.rmse_exact=sqrt(mean(diagnostics.residuals_exact.^2));
 trans(threshold,yt);
elseif strcmp(kind,'hitmiss')
 if any(~ismember(y,[0,1])) || numel(unique(y))~=2, error('op:data','Both hits and misses required'); end
 op_link(.5,link);
 center=mean(X(:,2:end),1); scale=std(X(:,2:end),1,1);
 Z=[X(:,1),bsxfun(@rdivide,bsxfun(@minus,X(:,2:end),center),scale)];
 bf=@(b)binary_nll(b,Z,y,link); b=solve(bf,zeros(k,1));
 if norm(b)>100 || max(abs(Z*b))>200, error('op:separation','Separated/near-separated data'); end
 theta=[b(1)-sum(b(2:end)'.*center./scale); b(2:end)./scale'];
 fn=@(t)binary_nll(t,X,y,link);
 covariance_method=getopt(opts,'covariance_method','fisher');
 if strcmp(covariance_method,'observed'), V=info_cov(fn,theta)*repeat;
 elseif strcmp(covariance_method,'fisher')
  eta=X*theta; [lp,lq]=op_linklogs(eta,link);
  if strcmp(link,'logit'), ld=lp+lq;
  elseif strcmp(link,'probit'), ld=-.5*eta.^2-.5*log(2*pi);
  elseif strcmp(link,'cloglog'), ld=eta-exp(min(eta,700));
  else, ld=-eta-exp(min(-eta,700)); end
  w=exp(2*ld-lp-lq); information=X'*bsxfun(@times,X,w);
  if min(eig(information))<=0, error('op:information','Singular Fisher information'); end
  V=(information\eye(k))*repeat;
 else, error('op:covariance','Use fisher/observed'); end
 [lp,unused]=op_linklogs(X*theta,link); p=exp(lp);
 diagnostics=struct('covariance_method',covariance_method,'deviance',2*fn(theta),'brier_score',mean((p-y).^2),'pearson_chi2',sum((p-y).^2./max(p.*(1-p),1e-15)));
else
 error('op:kind','Kind signal/hitmiss required');
end
model=struct('kind',kind,'theta',theta,'cov',V,'nll',fn,'x_transform',xt,'y_transform',yt,'threshold',threshold,'link',link,'n',size(X,1),'covariate_count',k-2,'repeated_factor',repeat,'diagnostics',diagnostics);
model.log_likelihood=-fn(theta); model.aic=2*numel(theta)+2*fn(theta);
if theta(2)>0 && k==2
 c=threshold; if strcmp(yt,'log'), c=log(c); end
 if strcmp(kind,'hitmiss'), c=0; scale=1/theta(2); else, scale=exp(theta(end))/theta(2); end
 mu=(c-theta(1))/theta(2); J=zeros(2,numel(theta)); J(1,1)=-1/theta(2); J(1,2)=-mu/theta(2); J(2,2)=-scale/theta(2);
 if strcmp(kind,'signal'), J(2,end)=scale; end
 model.pod_mu=mu; model.pod_scale=scale; model.pod_cov=J*V*J';
 model.a50=op_size(model,.5); model.a90=op_size(model,.9);
 model.a90_95=op_size_upper(model,.9,.95);
else
 model.a50=NaN; model.a90=NaN; model.a90_95=NaN;
end
end

function value=signal_nll(t,X,y,L,U,st)
if any(~isfinite(t)) || abs(t(end))>30, value=1e100; return; end
s=exp(t(end)); mu=X*t(1:end-1); v=zeros(size(y));
j=st==0; z=(y(j)-mu(j))/s; v(j)=-.5*z.^2-log(s)-.5*log(2*pi);
j=st==1; v(j)=op_logphi((U(j)-mu(j))/s);
j=st==2; v(j)=op_logphi((mu(j)-L(j))/s);
j=st==3;
if any(j)
 lo=(L(j)-mu(j))/s; hi=(U(j)-mu(j))/s;
 aa=op_logphi(hi); bb=op_logphi(lo); tail=lo>0;
 aa(tail)=op_logphi(-lo(tail)); bb(tail)=op_logphi(-hi(tail));
 v(j)=aa+log(-expm1(min(bb-aa,0)));
end
value=-sum(v); if ~isfinite(value), value=1e100; end
end
function value=binary_nll(t,X,y,link)
[lp,lq]=op_linklogs(X*t,link); v=lq; v(y==1)=lp(y==1); value=-sum(v);
if ~isfinite(value), value=1e100; end
end
function theta=solve(fn,start)
options=optimset('Display','off','MaxIter',12000,'MaxFunEvals',30000,'TolX',1e-9,'TolFun',1e-9);
[theta,value,flag]=fminsearch(fn,start,options);
if flag<=0 || ~isfinite(value) || value>=1e99, error('op:MLE','Optimization failed'); end
end
function V=info_cov(fn,t)
n=numel(t); H=zeros(n); h=2e-4*max(1,abs(t)); f=fn(t);
for i=1:n
 ei=zeros(n,1); ei(i)=h(i); H(i,i)=(fn(t+ei)-2*f+fn(t-ei))/h(i)^2;
 for j=1:i-1
  ej=zeros(n,1); ej(j)=h(j);
  H(i,j)=(fn(t+ei+ej)-fn(t+ei-ej)-fn(t-ei+ej)+fn(t-ei-ej))/(4*h(i)*h(j)); H(j,i)=H(i,j);
 end
end
if any(~isfinite(H(:))) || min(eig(H))<=0, error('op:information','MLE information not positive definite'); end
V=H\eye(n);
end
function v=getopt(s,name,default)
if isfield(s,name), v=s.(name); else, v=default; end
end
function y=trans(x,kind)
if strcmp(kind,'log')
 if any(x(:)<=0), error('op:log','Positive log input required'); end
 y=log(x);
elseif strcmp(kind,'identity'), y=x;
else, error('op:transform','Use log or identity'); end
end
