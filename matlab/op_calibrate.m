function result=op_calibrate(measured,standard,degree)
if nargin<3, degree=1; end
x=measured(:); y=standard(:); n=numel(x);
if numel(y)~=n || n<=degree+1 || any(~isfinite(x)) || any(~isfinite(y)), error('op:calibration','Finite paired standards n>degree+1'); end
X=zeros(n,degree+1); for j=0:degree, X(:,degree+1-j)=x.^j; end
if rank(X)<degree+1, error('op:calibration','Rank deficient standards'); end
b=X\y; residual=y-X*b; s2=sum(residual.^2)/(n-degree-1);
result=struct('coefficients',b','covariance',s2*((X'*X)\eye(degree+1)),'rmse',sqrt(mean(residual.^2)));
end
