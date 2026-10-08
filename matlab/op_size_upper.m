function a=op_size_upper(model,p,confidence,method,df,covariates)
if nargin<2, p=.9; end
if nargin<3, confidence=.95; end
if nargin<4 || isempty(method)
 if strcmp(model.kind,'signal'), method='wald'; else, method='lr'; end
end
if nargin<5, df=1; end
if nargin<6, covariates=[]; end
if confidence<=.5 || confidence>=1, error('op:confidence','Require .5<confidence<1'); end
[unused,q]=op_size(model,p,covariates); b=model.theta;
if strcmp(method,'wald')
 g=zeros(size(b)); g(1)=-1/b(2); g(2)=-q/b(2);
 if model.covariate_count>0, g(3:2+model.covariate_count)=-covariates(:)/b(2); end
 if strcmp(model.kind,'signal'), g(end)=op_link(p,'probit')*exp(b(end))/b(2); end
 upper=q+op_link(confidence,'probit')*sqrt(g'*model.cov*g);
elseif strcmp(method,'lr') || strcmp(method,'profile_one_sided')
 if model.covariate_count>0, error('op:LR','LR with nuisance covariates: use Wald/bootstrap'); end
 if strcmp(method,'profile_one_sided'), cutoff=op_link(confidence,'probit')^2;
 else
  if df==1, cutoff=op_link((1+confidence)/2,'probit')^2;
  elseif df==2, cutoff=-2*log(1-confidence);
  else, error('op:df','LR df 1 or 2 supported'); end
 end
 base=model.nll(b); fun=@(qx)profile(qx,model,p,base,cutoff);
 hi=q+max(1,.1*abs(q)); found=false;
 for i=1:40
  if fun(hi)>0, found=true; break; end
  hi=q+2*(hi-q);
 end
 if ~found, a=Inf; return; end
 upper=fzero(fun,[q,hi]);
else, error('op:method','Unknown method'); end
a=upper; if strcmp(model.x_transform,'log'), a=exp(upper); end
end
function value=profile(q,model,p,base,cutoff)
b=model.theta;
if strcmp(model.kind,'hitmiss'), start=b(2); else, start=[b(2);b(end)]; end
fn=@(v)profile_nll(v,q,model,p);
[unused,f,flag]=fminsearch(fn,start,optimset('Display','off','MaxIter',5000,'MaxFunEvals',12000,'TolX',1e-9,'TolFun',1e-9));
if flag<=0, error('op:profile','Profile optimization failed'); end
value=2*(f-base)/model.repeated_factor-cutoff;
end
function value=profile_nll(v,q,model,p)
if strcmp(model.kind,'hitmiss'), t=[op_link(p,model.link)-v(1)*q;v(1)];
else
 c=model.threshold; if strcmp(model.y_transform,'log'), c=log(c); end
 t=[c+op_link(p,'probit')*exp(v(2))-v(1)*q;v(1);v(2)];
end
value=model.nll(t);
end
