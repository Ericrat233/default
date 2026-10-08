function V=op_numeric_cov(fn,t)
t=t(:); n=numel(t); H=zeros(n); h=2e-4*max(1,abs(t)); f=fn(t);
for i=1:n
 ei=zeros(n,1); ei(i)=h(i); H(i,i)=(fn(t+ei)-2*f+fn(t-ei))/h(i)^2;
 for j=1:i-1
  ej=zeros(n,1); ej(j)=h(j); H(i,j)=(fn(t+ei+ej)-fn(t+ei-ej)-fn(t-ei+ej)+fn(t-ei-ej))/(4*h(i)*h(j)); H(j,i)=H(i,j);
 end
end
if any(~isfinite(H(:))) || min(eig(H))<=0, error('op:information','Nonpositive observed information'); end
V=H\eye(n);
end
