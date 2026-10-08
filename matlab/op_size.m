function [a,q]=op_size(model,p,covariates)
if nargin<2, p=.9; end
if nargin<3, covariates=[]; end
if model.theta(2)<=0, error('op:slope','Nonpositive slope; limit undefined'); end
b=model.theta; offset=b(1);
if model.covariate_count>0
 if numel(covariates)~=model.covariate_count, error('op:covariates','Set covariates'); end
 offset=offset+covariates(:)'*b(3:2+model.covariate_count);
end
if strcmp(model.kind,'hitmiss'), target=op_link(p,model.link);
else
 c=model.threshold; if strcmp(model.y_transform,'log'), c=log(c); end
 target=c+op_link(p,'probit')*exp(b(end));
end
q=(target-offset)/b(2); a=q;
if strcmp(model.x_transform,'log'), a=exp(q); end
end
