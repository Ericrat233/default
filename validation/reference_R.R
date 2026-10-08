# Run with: Rscript reference_R.R /path/to/mh1823-source/R /path/to/OpenPOD
# User must supply legitimately obtained reference source; not included in deliverable.
# Source-selected 7.3.7 routines used here, NOT an installed 7.3.0 GUI run.
args=commandArgs(trailingOnly=TRUE); ref=args[1]; root=args[2]
library(survival)
for(name in c('censored.regression','compute.a.POD.etc','estimate.model.glm','loglog.glm','GLM.hit.miss.log.likelihood.fn','ellipse.fn','rotate','unrotate.untranslate.pairs','find.interrogation.space','interrogate.criterion.contour','generate.all.plausible.POD.curves.0.0')) {
 source(file.path(ref,paste0(name,'.R')),encoding='UTF-8')
}
DIAGNOSTIC.PRINT=FALSE; LOG.X=TRUE; LOG.Y=FALSE; REPEATED=FALSE
critical.POD=.9; critical.conf=.95; results=list()
add=function(label,t,cov,limits,ll) {
 results[[label]] <<- c(t,rep(NA,3-length(t)),limits,ll)
 write.table(cov,file.path(root,'validation',paste0(label,'_R_cov.csv')),sep=',',row.names=FALSE,col.names=FALSE)
}
D=read.csv(file.path(root,'data/reference/example1.csv'),fileEncoding='UTF-8-BOM')
a.hat.columns=3; a.hat.vs.a.censored=censored.regression(data.frame(size=log(D[,2]),a.hat=D[,3]),-Inf,Inf)
add('example1',c(coef(a.hat.vs.a.censored),log(a.hat.vs.a.censored$scale)),a.hat.vs.a.censored$var,compute.a.POD.etc(200),as.numeric(logLik(a.hat.vs.a.censored)))
D=read.csv(file.path(root,'data/reference/example2.csv'),fileEncoding='UTF-8-BOM'); REPEATED=TRUE; a.hat.columns=3:6
sizes=rep(D[,2],4); responses=as.vector(as.matrix(D[,3:6]));
a.hat.vs.a.censored=censored.regression(data.frame(size=log(sizes),a.hat=responses),200,600)
add('example2',c(coef(a.hat.vs.a.censored),log(a.hat.vs.a.censored$scale)),a.hat.vs.a.censored$var*4,compute.a.POD.etc(250),as.numeric(logLik(a.hat.vs.a.censored)))
# Generated mixed left/right/interval and missing MAR response.
D=read.csv(file.path(root,'data/generated/censored_numeric.csv')); D=D[D$status_code!=4,]
L=D$lower; U=D$upper; L[D$status_code==1]=NA; U[D$status_code==2]=NA
L[D$status_code==0]=D$y[D$status_code==0]; U[D$status_code==0]=D$y[D$status_code==0]
REPEATED=FALSE; a.hat.vs.a.censored=survreg(Surv(L,U,type='interval2')~log(D$a),dist='gaussian')
add('generated_signal',c(coef(a.hat.vs.a.censored),log(a.hat.vs.a.censored$scale)),a.hat.vs.a.censored$var,compute.a.POD.etc(2),as.numeric(logLik(a.hat.vs.a.censored)))
# Exercise the ACTUAL 51x51 rotated reference contour/interpolation routine.
SIMULTANEOUS=FALSE; INTERROGATION.PLOTS=FALSE; file.label='OpenPOD validation'; LOG.X=FALSE
D=read.csv(file.path(root,'data/reference/example3.csv'),fileEncoding='UTF-8-BOM'); size.vs.hit.miss.data=data.frame(size=D[,2],hit.miss=D[,4])
for(link in c('logit','probit','cloglog','loglog')) {
 invlink.fn=switch(link,logit=function(z)plogis(z),probit=function(z)pnorm(z),cloglog=function(z)-expm1(-exp(z)),loglog=function(z)exp(-exp(-z)))
 link.fn=switch(link,logit=function(p)qlogis(p),probit=function(p)qnorm(p),cloglog=function(p)log(-log1p(-p)),loglog=function(p)-log(-log(p)))
 x.min=min(D[,2]); x.max=max(D[,2])*2
 generate.all.plausible.POD.curves.0.0(PLOT.0.0=FALSE,PLOT.GLM=FALSE,PLOT.NOTE=FALSE)
 b=coef(model.glm); limits=c((link.fn(.5)-b[1])/b[2],(link.fn(.9)-b[1])/b[2],a.90.95)
 add(paste0('example3_',link),b,VCV,limits,GLM.hit.miss.log.likelihood.fn(b,data.frame(size=D[,2],hit.miss=D[,4])))
}
# Independent native R GLM against all synthetic binary links.
LOG.X=TRUE; D=read.csv(file.path(root,'data/generated/hitmiss.csv')); size.vs.hit.miss.data=data.frame(size=D$a,hit.miss=D$y)
for(link in c('logit','probit','cloglog','loglog')) {
 estimate.model.glm(); b=coef(model.glm)
 link.fn=switch(link,logit=qlogis,probit=qnorm,cloglog=function(p)log(-log1p(-p)),loglog=function(p)-log(-log(p)))
 target=link.fn(.9); q=(target-b[1])/b[2]
 invlink.fn=switch(link,logit=function(z)plogis(z),probit=function(z)pnorm(z),cloglog=function(z)-expm1(-exp(z)),loglog=function(z)exp(-exp(-z)))
 data=data.frame(size=log(D$a),hit.miss=D$y); base=GLM.hit.miss.log.likelihood.fn(b,data)
 profile=function(qx) {
  fn=function(b1)-GLM.hit.miss.log.likelihood.fn(c(target-b1*qx,b1),data)
  r=optim(b[2],fn,method='BFGS',control=list(reltol=1e-12))
  2*(r$value+base)-qchisq(.95,1)
 }
 hi=q+1; while(profile(hi)<0)hi=q+2*(hi-q)
 bound=uniroot(profile,c(q,hi),tol=1e-10)$root
 limits=exp(c((link.fn(.5)-b[1])/b[2],q,bound))
 add(paste0('generated_',link),b,summary(model.glm)$cov.scaled,limits,base)
}
out=do.call(rbind,results); colnames(out)=c('b0','b1','log_sigma','a50','a90','a90_95','log_likelihood')
write.csv(data.frame(case=rownames(out),out),file.path(root,'validation/reference_R_results.csv'),row.names=FALSE)
writeLines(c(R.version.string,paste('survival',packageVersion('survival')),paste('reference',ref)),file.path(root,'validation/R_runtime.txt'))
