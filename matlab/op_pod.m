function p=op_pod(model,a,covariates)
a=a(:); if nargin<3, covariates=[]; end
if any(~isfinite(a)), error('op:data','Finite size required'); end
if strcmp(model.x_transform,'log')
 if any(a<=0), error('op:log','Positive sizes required'); end
 x=log(a);
else, x=a; end
X=[ones(numel(a),1),x];
if model.covariate_count>0
 if size(covariates,1)~=numel(a) || size(covariates,2)~=model.covariate_count || any(~isfinite(covariates(:))), error('op:covariates','Covariates required'); end
 X=[X,covariates];
end
if strcmp(model.kind,'hitmiss')
 [lp,unused]=op_linklogs(X*model.theta,model.link); p=exp(lp);
else
 c=model.threshold; if strcmp(model.y_transform,'log'), c=log(c); end
 z=(X*model.theta(1:end-1)-c)/exp(model.theta(end)); p=.5*erfc(-z/sqrt(2));
end
end
