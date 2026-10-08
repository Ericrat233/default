function result=op_bootstrap(a,y,groups,opts,kind,replicates,seed)
% Resample independent clusters, preserve all rows per selected group.
if nargin<5, kind='signal'; end
if nargin<6, replicates=200; end
if nargin<7, seed=1823; end
rng(seed); a=a(:); y=y(:); groups=groups(:); ids=unique(groups); values=[]; failures=0;
if numel(ids)<4 || replicates<20, error('op:bootstrap','>=4 groups and >=20 replicates'); end
if numel(y)~=numel(a) || numel(groups)~=numel(a), error('op:bootstrap','Matched vectors'); end
for i=1:replicates
 chosen=ids(randi(numel(ids),numel(ids),1)); idx=[];
 for j=1:numel(chosen), idx=[idx;find(groups==chosen(j))]; end
 local=opts; fields={'status','lower','upper','covariates'};
 for j=1:numel(fields)
  name=fields{j}; if isfield(local,name) && ~isscalar(local.(name)), v=local.(name); local.(name)=v(idx,:); end
 end
 try
  m=op_fit(kind,a(idx),y(idx),local); values=[values;m.a90];
 catch
  failures=failures+1;
 end
end
if failures>replicates*.2, error('op:bootstrap','>20 percent fits failed'); end
values=sort(values); at=1+(numel(values)-1)*.95; upper=values(floor(at))+(at-floor(at))*(values(ceil(at))-values(floor(at)));
result=struct('a90_samples',values,'a90_bootstrap_upper95',upper,'successful',numel(values),'failed',failures);
end
