function z=op_link(p,name)
if any(p(:)<=0 | p(:)>=1), error('op:POD','0<POD<1'); end
if strcmp(name,'logit'), z=log(p./(1-p));
elseif strcmp(name,'probit'), z=-sqrt(2)*erfcinv(2*p);
elseif strcmp(name,'cloglog'), z=log(-log1p(-p));
elseif strcmp(name,'loglog'), z=-log(-log(p));
else, error('op:link','Unknown link'); end
end
