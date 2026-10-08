function result=op_gp(X,y,Xnew,lengthscale,nugget)
% Fixed-hyperparameter scaled RBF Gaussian process; latent predictive variance.
if nargin<4, lengthscale=1; end
if nargin<5, nugget=1e-4; end
y=y(:);
if size(X,1)~=numel(y) || any(~isfinite(X(:))) || any(~isfinite(y)) || lengthscale<=0 || nugget<=0, error('op:gp','Invalid GP input'); end
center=mean(X,1); scale=std(X,1,1); scale(scale==0)=1;
Z=bsxfun(@rdivide,bsxfun(@minus,X,center),scale); Znew=bsxfun(@rdivide,bsxfun(@minus,Xnew,center),scale);
m=mean(y); sd=std(y,1); if sd<=0, error('op:gp','Constant GP response'); end
K=kernel(Z,Z,lengthscale)+nugget*eye(numel(y)); L=chol(K,'lower'); alpha=L'\(L\((y-m)/sd)); Kn=kernel(Znew,Z,lengthscale);
pred=m+sd*Kn*alpha; v=L\Kn'; variance=max(0,1-sum(v.^2,1)');
result=struct('mean',pred,'sd',sd*sqrt(variance),'length_scale',lengthscale,'nugget',nugget);
end
function K=kernel(X,Y,ls)
D=bsxfun(@plus,sum(X.^2,2),sum(Y.^2,2)')-2*X*Y'; K=exp(-max(D,0)/(2*ls^2));
end
