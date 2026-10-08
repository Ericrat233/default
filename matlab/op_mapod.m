function result=op_mapod(simulator,a,nuisance,threshold,calibration,seed)
% Model-assisted POD. Simulator returns [numel(a),size(nuisance,1)].
% MC band is numerical sampling uncertainty only.
if nargin<5, calibration=[]; end
if nargin<6, seed=1823; end
rng(seed); a=a(:); n=size(nuisance,1); z=simulator(a,nuisance);
if n<20 || ~isequal(size(z),[numel(a),n]) || any(~isfinite(z(:))), error('op:mapod','Invalid simulation matrix/nuisance samples'); end
if isempty(calibration), y=z;
else, y=calibration.coefficients(1)+calibration.coefficients(2)*z+calibration.discrepancy_sd*randn(size(z)); end
hits=sum(y>threshold,2); p=hits/n; lo=zeros(size(p)); hi=ones(size(p));
for i=1:numel(p)
 if hits(i)>0, lo(i)=betaincinv(.025,hits(i),n-hits(i)+1); end
 if hits(i)<n, hi(i)=betaincinv(.975,hits(i)+1,n-hits(i)); end
end
result=struct('a',a,'pod',p,'mc_lower',lo,'mc_upper',hi,'n_mc',n);
end
