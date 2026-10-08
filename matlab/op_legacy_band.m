function result=op_legacy_band(model,a,p,confidence,simultaneous)
% Historical mh1823 51x51 rotated contour + 101 plotting-size interpolation.
% Independently implemented, grid dependent; use op_lr_band for new analyses.
if nargin<3, p=.9; end
if nargin<4, confidence=.95; end
if nargin<5, simultaneous=false; end
if ~strcmp(model.kind,'hitmiss') || model.covariate_count>0, error('op:legacy','Baseline binary only'); end
V=model.cov; b=model.theta; theta=atan(V(1,2)/V(2,2)); R=[cos(theta),sin(theta);-sin(theta),cos(theta)];
prob=.99; if simultaneous, prob=.999; end
dist=sqrt(-2*log(1-prob)); yy=linspace(-sqrt(V(2,2))*dist,sqrt(V(2,2))*dist,101)';
root=sqrt(abs(det(V)/V(2,2)^2*(V(2,2)*dist^2-yy.^2))); root([1,end])=0; center=V(1,2)/V(2,2)*yy;
ellipse=[[center-root,yy];flipud([center+root,yy])]*R;
bx=linspace(min(ellipse(:,1)),max(ellipse(:,1)),51); by=linspace(min(ellipse(:,2)),max(ellipse(:,2)),51); z=zeros(51);
for i=1:51
 for j=1:51
  t=([bx(j),by(i)]*R')'+b; z(i,j)=-model.nll(t)/model.repeated_factor;
 end
end
if confidence==.95, drop=1.9207; elseif confidence==.9, drop=1.353; else, error('op:legacy','90/95 percent only'); end
if simultaneous
 % chi-square 2 quantile of CDF(2*drop,1); erf is chi-square1 CDF.
 prob=erf(sqrt(drop)); drop=-log(1-prob);
end
level=round((-model.nll(b)/model.repeated_factor-drop)*1e4)/1e4;
segments=contourc(bx,by,z,[level,level]); vertices=[]; index=1;
while index<size(segments,2)
 n=segments(2,index); vertices=[vertices;segments(:,index+1:index+n)']; index=index+n+1;
end
beta=bsxfun(@plus,vertices*R',b'); a=a(:); x=a;
if strcmp(model.x_transform,'log'), x=log(a); end
eta=bsxfun(@plus,beta(:,1),beta(:,2)*x'); elo=min(eta,[],1)'; ehi=max(eta,[],1)';
[lp,unused]=op_linklogs(elo,model.link); lower=exp(lp);
[lp,unused]=op_linklogs(ehi,model.link); upper=exp(lp);
q=interp1(elo,x,op_link(p,model.link)); value=q; if strcmp(model.x_transform,'log'), value=exp(q); end
result=struct('a',a,'lower',lower,'upper',upper,'beta',beta,'a_p_conf',value,'grid_points',51);
end
