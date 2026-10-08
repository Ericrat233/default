function out=op_shift(x,delay)
% Positive integer delay, no circular wrapping.
x=x(:); out=zeros(size(x)); n=numel(x);
if delay~=round(delay), error('op:shift','Integer delay required'); end
if abs(delay)>=n, return; end
if delay>0, out(delay+1:end)=x(1:end-delay);
elseif delay<0, out(1:end+delay)=x(1-delay:end);
else, out=x; end
end
