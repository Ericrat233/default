function result=op_bayes_linear(a,y,threshold,draws,seed,logx)
% Exact normal inverse gamma posterior; credible != confidence bound.
% Prior beta|s2 ~ N(0,100*s2 I), s2~IG(2,1); adjust source for sensitivity.
if nargin<4, draws=4000; end
if nargin<5, seed=1823; end
if nargin<6, logx=true; end
rng(seed); a=a(:); y=y(:);
if numel(a)~=numel(y) || any(~isfinite(y)), error('op:bayes','Invalid data'); end
x=a; if logx, if any(a<=0), error('op:log','Positive sizes'); end; x=log(a); end
X=[ones(numel(a),1),x]; L0=.01*eye(2); Ln=L0+X'*X; mn=Ln\(X'*y);
an=2+numel(y)/2; bn=1+.5*(y'*y-mn'*Ln*mn);
s2=bn./randg(an,draws,1); B=bsxfun(@plus,bsxfun(@times,randn(draws,2)*chol(Ln\eye(2)),sqrt(s2)),mn');
pos=B(:,2)>0; q=Inf(draws,1); q(pos)=(threshold-B(pos,1)+op_link(.9,'probit')*sqrt(s2(pos)))./B(pos,2);
if logx, q=exp(q); end
result=struct('beta_draws',B,'sigma_draws',sqrt(s2),'a90_draws',q,'a90_credible_upper95',percentile(q,.95),'nonpositive_slope_fraction',mean(~pos));
end
function value=percentile(x,p)
x=sort(x); r=1+(numel(x)-1)*p; lo=floor(r); hi=ceil(r); if lo==hi, value=x(lo); else, value=x(lo)+(r-lo)*(x(hi)-x(lo)); end
end
