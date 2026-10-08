function y=op_logphi(x)
% Stable log normal CDF, including deep negative tails, without Stats Toolbox.
y=zeros(size(x)); neg=x<0;
y(neg)=log(.5)+log(erfcx(-x(neg)/sqrt(2)))-x(neg).^2/2;
y(~neg)=log1p(-.5*erfc(x(~neg)/sqrt(2)));
end
