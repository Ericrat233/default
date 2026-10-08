function value=op_snr(signal_values,noise_values,amplitude)
if nargin<3, amplitude=false; end
s=signal_values(:); n=noise_values(:);
if isempty(s) || isempty(n) || any(~isfinite(s)) || any(~isfinite(n)) || mean(n.^2)<=0, error('op:SNR','Finite windows, positive noise power required'); end
if amplitude, value=20*log10(max(abs(s))/sqrt(mean(n.^2)));
else, value=10*log10(mean(s.^2)/mean(n.^2)); end
end
