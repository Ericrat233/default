function model=op_noise(y,family,status,bounds,pfa,reference)
% Noise MLE exact/left/right only; status 0/1/2. Bounds on raw amplitude.
y=y(:); n=numel(y);
if nargin<2, family='normal'; end
if nargin<3, status=zeros(n,1); end
if nargin<4, bounds=y; end
if nargin<5, pfa=.01; end
if nargin<6, reference=1; end
status=status(:); bounds=bounds(:); j=status==0; z=y(j);
if numel(status)~=n || numel(bounds)~=n || any(~ismember(status,0:2)) || any(~isfinite(z)) || any(~isfinite(bounds(~j))) || numel(z)<3 || std(z)<1e-10
 error('op:noise','Invalid noise data');
end
if pfa<=0 || pfa>=1 || reference<=0, error('op:noise','Invalid PFA/reference'); end
sd=sqrt(mean((z-mean(z)).^2));
if strcmp(family,'normal'), start=[mean(z);log(sd)];
elseif strcmp(family,'lognormal')
 if any(z<=0) || any(bounds(~j)<=0), error('op:noise','Positive lognormal inputs'); end
 start=[mean(log(z));log(std(log(z),1))];
elseif strcmp(family,'weibull')
 if any(z<=0) || any(bounds(~j)<=0), error('op:noise','Positive Weibull inputs'); end
 start=[0;log(mean(z))];
elseif strcmp(family,'exponential')
 if any(z<0) || any(bounds(~j)<0), error('op:noise','Nonnegative exponential inputs'); end
 start=log(mean(z));
elseif strcmp(family,'lev'), start=[mean(z)-.5772*sd*sqrt(6)/pi;log(sd*sqrt(6)/pi)];
else, error('op:noise','Unknown family'); end
fn=@(t)nll(t,y,status,bounds,family);
[t,f,flag]=fminsearch(fn,start,optimset('Display','off','MaxIter',10000,'MaxFunEvals',25000,'TolX',1e-9,'TolFun',1e-9));
if flag<=0, error('op:noise','Noise MLE failed'); end
V=op_numeric_cov(fn,t);
if strcmp(family,'normal'), th=t(1)+exp(t(2))*op_link(1-pfa,'probit');
elseif strcmp(family,'lognormal'), th=exp(t(1)+exp(t(2))*op_link(1-pfa,'probit'));
elseif strcmp(family,'weibull'), th=exp(t(2))*(-log(pfa))^(1/exp(t(1)));
elseif strcmp(family,'exponential'), th=-exp(t(1))*log(pfa);
else, th=t(1)-exp(t(2))*log(-log1p(-pfa)); end
model=struct('family',family,'theta',t,'cov',V,'log_likelihood',-f,'aic',2*numel(t)+2*f,'threshold',th,'pfa',pfa,'threshold_db',NaN);
if th>0, model.threshold_db=20*log10(th/reference); end
model.pfa_at=@(x)survival(x,t,family);
end
function v=nll(t,y,st,bounds,family)
if strcmp(family,'weibull') || strcmp(family,'exponential'), lp=t; else, lp=t(end); end
if any(~isfinite(t)) || max(abs(lp))>30, v=1e100; return; end
v=zeros(size(y)); j=st==0; v(j)=logpdf(y(j),t,family);
j=st==1; v(j)=logcdf(bounds(j),t,family);
j=st==2; v(j)=logsf(bounds(j),t,family);
v=-sum(v); if ~isfinite(v), v=1e100; end
end
function v=logpdf(y,t,f)
if strcmp(f,'normal'), s=exp(t(2)); v=-.5*((y-t(1))/s).^2-log(s)-.5*log(2*pi);
elseif strcmp(f,'lognormal'), s=exp(t(2)); v=-.5*((log(y)-t(1))/s).^2-log(s)-log(y)-.5*log(2*pi);
elseif strcmp(f,'weibull'), k=exp(t(1)); s=exp(t(2)); v=log(k)-log(s)+(k-1)*log(y/s)-(y/s).^k;
elseif strcmp(f,'exponential'), s=exp(t(1)); v=-log(s)-y/s;
else, s=exp(t(2)); z=(y-t(1))/s; v=-log(s)-z-exp(-z); end
end
function v=logcdf(y,t,f)
if strcmp(f,'normal'), v=op_logphi((y-t(1))/exp(t(2)));
elseif strcmp(f,'lognormal'), v=op_logphi((log(y)-t(1))/exp(t(2)));
elseif strcmp(f,'weibull'), v=log(-expm1(-(y/exp(t(2))).^exp(t(1))));
elseif strcmp(f,'exponential'), v=log(-expm1(-y/exp(t(1))));
else, v=-exp(-(y-t(1))/exp(t(2))); end
end
function v=logsf(y,t,f)
if strcmp(f,'normal'), v=op_logphi((t(1)-y)/exp(t(2)));
elseif strcmp(f,'lognormal'), v=op_logphi((t(1)-log(y))/exp(t(2)));
elseif strcmp(f,'weibull'), v=-(y/exp(t(2))).^exp(t(1));
elseif strcmp(f,'exponential'), v=-y/exp(t(1));
else, v=log(-expm1(-exp(-(y-t(1))/exp(t(2))))); end
end
function p=survival(x,t,f)
p=exp(logsf(x,t,f));
if strcmp(f,'weibull') || strcmp(f,'lognormal') || strcmp(f,'exponential'), p(x<0)=1; end
end
