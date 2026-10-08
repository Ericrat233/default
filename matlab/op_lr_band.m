function [lower,upper]=op_lr_band(model,a,confidence,df)
% Exact profile envelope at each size. Baseline hit/miss only.
if nargin<3, confidence=.95; end
if nargin<4, df=1; end
if ~strcmp(model.kind,'hitmiss') || model.covariate_count>0, error('op:band','Baseline hit/miss required'); end
if df==1, cutoff=op_link((1+confidence)/2,'probit')^2;
elseif df==2, cutoff=-2*log(1-confidence);
else, error('op:df','df 1/2 only'); end
a=a(:); x=a;
if strcmp(model.x_transform,'log'), if any(a<=0), error('op:log','Positive sizes'); end; x=log(a); end
lower=zeros(size(a)); upper=lower; base=model.nll(model.theta);
for i=1:numel(x)
 row=[1,x(i)]; eta=row*model.theta; span=max(1,sqrt(row*model.cov*row')); fun=@(v)profile(v,x(i),model,base,cutoff);
 low=eta-span; high=eta+span;
 for j=1:40, if fun(low)>0, break; end; low=eta-2*(eta-low); end
 for j=1:40, if fun(high)>0, break; end; high=eta+2*(high-eta); end
 elo=fzero(fun,[low,eta]); ehi=fzero(fun,[eta,high]);
 [lp,unused]=op_linklogs(elo,model.link); lower(i)=exp(lp);
 [lp,unused]=op_linklogs(ehi,model.link); upper(i)=exp(lp);
end
end
function f=profile(eta,x,m,base,cutoff)
fn=@(b1)m.nll([eta-b1*x;b1]); [unused,v,flag]=fminsearch(fn,m.theta(2),optimset('Display','off','MaxIter',3000,'TolX',1e-9,'TolFun',1e-9));
if flag<=0, error('op:profile','LR profile failed'); end
f=2*(v-base)/m.repeated_factor-cutoff;
end
