DIR="/home/praveen/Paurush/Dream8/GEP/"
setwd(DIR)
load(paste(DIR, "testInput.rda", sep=""))
dim(datex_train2)
comp=cbind(datex_train2, datex_leader2)
d=dist(t(comp))
hc=hclust(d=d, method="complete", members=NULL)
k=5
groups=cutree(hc, k=k)
plot(hc, hang=-1, labels=groups)
rect.hclust(hc, k=k, which=NULL, x=NULL, border=2, cluster=NULL)
ccel=list()
learn=app=cnvT=cnvLe=y=list()
for(j in 1:k){
	cat(length(which(groups==j)))
	cat("\n")
	ccel[[j]]=names(which(groups==j))
	l=intersect(ccel[[j]], colnames(datex_train2))
	a=intersect(ccel[[j]], colnames(datex_leader2))
	learn[[j]]=datex_train2[,l]
	app[[j]]=datex_leader2[,a]
	cnvT[[j]]=cnv_train2[,l]
	cnvLe[[j]]=cnv_leader[,a]
	y[[j]]=essen_train2[,l]
	cat(length(learn))
	cat(length(app))
}

myres=list()
for(s in 1:length(learn)){
	myres[[s]]=runLearning(EXtrain=learn[[s]], yvar=y[[s]], use.cnv=FALSE, EXtest=app[[s]], cnv.train=cnvT[[s]], cnv.test=cnvLe[[s]], cross=FALSE, K=3,  featureSelection="top", svrParams="autoOpt", addKnowl=FALSE, knowProb=onco, fn=2000, outGCT=FALSE, file="prediction.gct")
}

