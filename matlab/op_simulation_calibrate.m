function result=op_simulation_calibrate(simulated,experimental)
x=simulated(:); y=experimental(:); n=numel(x);
if n<5 || numel(y)~=n || any(~isfinite(x)) || any(~isfinite(y)), error('op:calibration','>=5 calibration pairs'); end
X=[ones(n,1),x]; if rank(X)<2, error('op:calibration','Constant simulation'); end
b=X\y; sd=sqrt(sum((y-X*b).^2)/(n-2));
result=struct('coefficients',b,'covariance',sd^2*((X'*X)\eye(2)),'discrepancy_sd',sd,'n',n);
end
