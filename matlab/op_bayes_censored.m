function result=op_bayes_censored(model,draws,burn,nchains,seed,prior_sd)
% Multi-chain random-walk Metropolis, proper N(0,prior_sd^2) parameter prior.
if nargin<2, draws=1500; end
if nargin<3, burn=1000; end
if nargin<4, nchains=4; end
if nargin<5, seed=1823; end
if nargin<6, prior_sd=10; end
if ~strcmp(model.kind,'signal') || model.covariate_count>0 || draws<100 || burn<100 || nchains<2, error('op:bayes','Baseline signal MCMC, >=2 chains'); end
rng(seed); b=model.theta; d=numel(b); L=chol(model.cov/model.repeated_factor,'lower'); output=zeros(draws,d,nchains); acceptance=zeros(nchains,1);
for c=1:nchains
 t=b+L*randn(d,1); v=logpost(t,model,prior_sd); scale=2.38/sqrt(d); accepted=0; window=0;
 for i=1:burn+draws
  candidate=t+scale*L*randn(d,1); nv=logpost(candidate,model,prior_sd); accept=log(rand)<nv-v;
  if accept, t=candidate; v=nv; window=window+1; end
  if i<=burn && mod(i,50)==0, scale=scale*exp(max(-.2,min(.2,window/50-.25))); window=0; end
  if i>burn, output(i-burn,:,c)=t'; accepted=accepted+accept; end
 end
 acceptance(c)=accepted/draws;
end
h=floor(draws/2); split=cat(3,output(1:h,:,:),output(end-h+1:end,:,:)); W=squeeze(mean(var(split,0,1),3)); means=squeeze(mean(split,1)); B=h*var(means,0,2); rhat=sqrt(((h-1)/h*W+B/h)./W);
ess=zeros(d,1);
for j=1:d
 sumrho=0;
 for lag=1:min(floor(draws/2),1000)-1
  rho=0;
  for c=1:nchains
   z=output(:,j,c)-mean(output(:,j,c)); rho=rho+(z(1:end-lag)'*z(1+lag:end))/(z'*z)/nchains;
  end
  if rho<=0, break; end
  sumrho=sumrho+rho;
 end
 ess(j)=nchains*draws/(1+2*sumrho);
end
flat=reshape(permute(output,[1,3,2]),[],d); pos=flat(:,2)>0; q=Inf(size(flat,1),1); threshold=model.threshold;
if strcmp(model.y_transform,'log'), threshold=log(threshold); end
q(pos)=(threshold-flat(pos,1)+op_link(.9,'probit')*exp(flat(pos,end)))./flat(pos,2);
if strcmp(model.x_transform,'log'), q=exp(q); end
q=sort(q); at=1+(numel(q)-1)*.95; upper=q(floor(at))+(at-floor(at))*(q(ceil(at))-q(floor(at)));
result=struct('chains',output,'rhat',rhat,'ess',ess,'acceptance',acceptance,'a90_credible_upper95',upper,'converged_diagnostic',all(rhat<1.01)&all(ess>400));
end
function value=logpost(t,model,sd)
value=-model.nll(t)-.5*sum((t/sd).^2);
end
