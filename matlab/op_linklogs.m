function [lp,lq]=op_linklogs(eta,name)
if strcmp(name,'logit')
 lp=-max(-eta,0)-log1p(exp(-abs(eta))); lq=-max(eta,0)-log1p(exp(-abs(eta)));
elseif strcmp(name,'probit')
 lp=op_logphi(eta); lq=op_logphi(-eta);
elseif strcmp(name,'cloglog')
 e=exp(min(eta,700)); lp=log(-expm1(-e)); lp(eta<-35)=eta(eta<-35); lq=-e;
elseif strcmp(name,'loglog')
 [lq,lp]=op_linklogs(-eta,'cloglog');
else, error('op:link','Unknown link'); end
end
