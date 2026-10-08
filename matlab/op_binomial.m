function interval=op_binomial(hits,total,confidence,one_sided)
if nargin<3, confidence=.95; end
if nargin<4, one_sided=false; end
if total<=0 || hits<0 || hits>total, error('op:binomial','0 <= hits <= total'); end
alpha=(1-confidence)/2; if one_sided, alpha=1-confidence; end
lo=0; hi=1;
if hits>0, lo=betaincinv(alpha,hits,total-hits+1); end
if hits<total, hi=betaincinv(1-alpha,hits+1,total-hits); end
interval=[lo,hi];
end
