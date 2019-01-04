###########################################################
### AUTHOR: PAURUSH PRAVEEN
### PURPOSE: GENE ESSENTIALITY PREDICTION BASED ON DATA
### E-mail: praveen@cosbi.eu
### THE SVR METHOD TO COMPUTE GENE ESSENTIALITY 
### ALLOWS OPTIONAL BOOTSTRAP
### BASED ON -ONE GENE ONE MODEL- PRINCIPLE
### FACILITATES KNOWLEDGE BASED FEATURE SELECTION
### WITH PARAMETER OPTIMIZATION FUNCTIONS BUILT-IN
### SEE Cherkassy et.al 2004, Vapnik et.al 1998
###	Collobert et.al 2001, Drucker et.al 1996
### Requires: kernlab, e1071, ggplot2 (optional)
###########################################################
library(kernlab)
library(e1071)
#library(ggplot2)
runSVR=function(D, ess, cnv.train=cnvTrain, cnv.test=cnvTest,KFcross=TRUE, K=10, fn=fn, featureSelection=c("top", "significant", "anova"), use.cnv=FALSE, kfeat=NULL, addKnowl, kernel="anovadot", svmtype="eps-svr", svrParams="autoOpt", params=list(eps=0.01, nu=0.01, C=1), lim){
	#fn=length(fea)
	if(!use.cnv){
		myData=t(D)
	}
	if(use.cnv){
		myData=cbind(t(D), t(cnv.train))
	}
	cat(paste("length of Knowledge features= ", length(kfeat), "\n"))
	cat("RUNNING SVR ALGORITHM \n")
	#rownames(myData)=NULL
	#sdv=apply(myData[,1:12951], 2, var)
	#sdv=rank(sdv)
	#fea=names(sort(sdv, decreasing = TRUE)[1:fn])
	#colnames(myData)=paste("X",colnames(myData), sep="")
	#dd=myData[1:66,]#[,fea]
	#dd=cbind(dd, Essentiality=as.numeric(ess))
	#dd=data.frame(dd, check.names=FALSE)
	cat("###### OPTIMIZING HYPER-PARAMETERS ######\n")	# Get optimal SVM parameters for the data if not assigned manually -autoOpt- mode
	#C=2^c(-2:12)
	if(svrParams=="autoOpt"){
		if(svmtype=="eps-svr"){ 
			epsilon=max(3*sd(ess)*(sqrt(log(length(ess))/length(ess))),0.1)
		}
		if(svmtype=="nu-svr"){
			Nu=sigest(Essemtiality~., data=data.frame(myData, Essentiality=ess))
			Nu=Nu[3]-Nu[1]
		}else{
			Nu=params$nu
		}
		
		C=max(abs(mean(ess)+sd(ess)), abs(mean(ess)-sd(ess)))
	}
	if(svrParams=="forced"){
		C=params$C
		epsilon=params$eps
		Nu=params$nu
	}
	cat("###### HYPER-PARAMETERS OPTIMIZED ######\n")
	#epsilon=c(rnorm(n=10, mean = epsilon, sd = 0.05), abs(rnorm(n=10, mean=0.000000001, sd=0.00000005)))
	#hyper=list(nu,C,epsilon)
	#x=y=c()
	#es=f=list()
	if(featureSelection!="none"){
		fea=selectFeatures(myData, ess,fn=fn, seltype=featureSelection)
		if(addKnowl==TRUE & length(kfeat) > 0){
			fea=unique(union(fea, kfeat))
		}
	}
	if(featureSelection=="knowl" & length(kfeat) > 0 & addKnowl==TRUE){
		fea=kfeat
	}
	cat("Number of features =", length(fea), "\n" )
	#cat(fea)
	#cat("\n ******* ******* ******** ********* \n")
	myperf=perf=0
	if(KFcross & lim < 300){
	cat("###### K-FOLD CROSS VALIDATION BEGINS ######\n") 	
	#myperf=perf=0
	for(k in 1:K){
		n=nrow(myData)
		m=ncol(myData)
	#	mdash=m-1
		index = sample(n, round(0.67*n)) # sample the training and test data
		#train.d = dd[train,]
		trainset=myData[index,fea] # BUG DETECTED BY SIMONE FIXED
		trainset=cbind(as.matrix(trainset), Essentiality=ess[index])	
		testset=myData[-index,fea]
		f = ksvm(Essentiality ~., data = trainset, C = C, nu=Nu, epsilon = epsilon, type="eps-svr", kernel="laplacedot")
		cat("####################\n")
		es = predict(f, testset, type = "response", coupler = "minpair")
		tempperf=cor(es, ess[-index], method="spearman")
# 		if(tempperf>myperf){
# 			myperf=tempperf
# 			bestfit=f
# 		}
# 		perf=perf+tempperf
		p=perf/k
	}
	print(paste("Cross Validation performance is =", p))
	}
	allData=cbind(as.matrix(myData[,fea]), Essentiality=ess)	
	fit=ksvm(Essentiality ~., data = allData, C = C, nu=Nu, epsilon = epsilon, type="eps-svr", kernel="laplacedot")
	model=list(fit=fit, features=fea)#, bestfit=bestfit, perf=p)
	#nu=sigest(Essentiality~.,data =temp )
		#nu=nu[3]-nu[1]
		#test.d = temp[-train,]
		#newdata = test.d
		#act=ess[-train]
		#trainset= temp #apply(temp[train,], 2, as.numeric)
		#colnames(trainset)=c(cfea, "Essentiality")
		#testset=apply(dd[-train,fea], 2, as.character)
		#testset=apply(testset, 2 , as.numeric)
		#cat(paste("####### CV ", k, "TH -FOLD RUNNING SETS COMPILED ######\n", sep=""))
		#cat(head(fea))
		#cat("$$$$$ \n")
		#cat(length(fea))
		#cat("@@@@@ \n")
		#cat(tail(fea))
		#cat("***** \n")
		#testset=testset[,fea]
		#print(colnames(trainset))
		#rownames(testset)=rownames(trainset)=NULL
		#f = ksvm(Essentiality ~., data = temp, C = 2, epsilon = epsilon, type="eps-svr", kernel="laplacedot")
		#es= predict(f, myData[67:99,fea], type = "response", coupler = "minpair")
		#print(act)
		#cat(class(act))
		#cat("\n")
		#print(es[[k]])
		#cat(class(es[[k]]))
		#cat("\n")
		#x=c(x, cor(act, es[[k]]))
		#y=c(y, cor(act, es[[k]], method="spearman"))
		#print(x)
		cat("\n######## SVR COMPLETED ###########\n")
		rm(fea)#testset, trainset)
	#}
	#allres=list(p=x, s=y)#, model=f
	return(model)
}
#resmat=matrix(0, nrow=ncol(essTraP2), ncol=33)
selectFeatures=function(myData, ess, fn, seltype){
	m=ncol(myData)
	if(seltype=="top"){
		corV=c()
		cat("####### COMPUTING FEATURES ######\n")
			for(s in 1:m){
				corV=c(corV,cor(as.numeric(myData[,s]),as.numeric(ess), method="spearman"))#
			}
 		names(corV)=colnames(myData)
 		corV=sort(corV, decreasing=TRUE)
 		fea=names(corV)[1:fn]
	}
	if(seltype=="significant"){
		corV=c()
		cat("####### COMPUTING FEATURES ######\n")
 		for(s in 1:m){
 			corV=c(corV,cor.test(as.numeric(myData[,s]),as.numeric(ess), method="spearman")$p.value)#
 		}
 		names(corV)=colnames(myData)
 		n=which(corV < 0.05)
 		fea=names(corV[n])
	}
	if(seltype=="anova"){
		corV=c()
		cat("####### COMPUTING FEATURES ######\n")
 		for(s in 1:m){
 			corV=c(corV,cor(as.numeric(myData[,s]),as.numeric(ess), method="spearman"))#
 		}
 		names(corV)=colnames(myData)
 		corV=sort(corV, decreasing=TRUE)
 		fea=names(corV)[1:fn]
	}
	return(fea)
}


runLearning=function(EXtrain, EXtest, cnv.train=cnvTrain, cnv.test=cnvTest, yvar,  use.cnv=TRUE, cross=TRUE, K=10, fn=4000, featureSelection="top", svrParams="autoOpt", addKnowl=FALSE, knowProb=NULL, outGCT=FALSE, filename="prediction.gct"){
if(ncol(EXtrain)!=ncol(yvar)){
	stop("Training and output not of same dimensions. \n")
}
# if(colnames(train)!=colnames(yvar)){
# 	stop("Colnames of train and output should be same. \n")
# }
# if(rownames(train)!=rownames(testset)){
# 	stop("The probe names in test and training data should be the same. \n")
# }
test=as.matrix(t(EXtest))
cnvtest=as.matrix(t(cnv.test))
#resmat=matrix(0, ncol=nrow(yvar), nrow=ncol(test))
allgenes=rownames(yvar)
geneSize=length(allgenes)
resmat=matrix(0, ncol=nrow(test), nrow=geneSize)
perfo=0
for(j in 1:geneSize){
	ess=as.numeric(yvar[allgenes[j],])
	#datSel4=apply(as.matrix(datSel3),2,as.character)
	#datSel4=apply(datSel4, 2, as.numeric)#cbind(, Essentiality=as.numeric(c(ess)))
	#corV=c()
	#for(s in 1:ncol(datSel)){
	#corV=c(corV, cor(datSel2[,s], datSel2[,12952]))
	#}
	#names(corV)=colnames(datSel2)[1:12951]
	#corV=sort(corV, decreasing=TRUE)
	#fea=names(corV)[1:2000]
	cat(paste("Running SVR for gene ", j, allgenes[j],"\n"))
	
	if(!use.cnv){
		if(addKnowl){
			mod=runSVR(EXtrain, ess, KFcross=cross, K=3, addKnowl=TRUE, fn=fn, kfeat=knowProb,featureSelection="top",svrParams="autoOpt", lim=j)
		}
		else{
			mod=runSVR(EXtrain, ess, KFcross=cross, K=3, fn=fn, addKnowl=FALSE, featureSelection="top",svrParams="autoOpt", lim=j)
		}
	}
	if(use.cnv){	
		if(addKnowl){
			mod=runSVR(EXtrain, ess, cnv.train=cnv.train, cnv.test=cnv.test, yvar,  use.cnv=TRUE, KFcross=cross, K=3, addKnowl=TRUE, fn=fn, kfeat=knowProb,featureSelection="top",svrParams="autoOpt", lim=j)
		}
		else{
			mod=runSVR(EXtrain, ess, KFcross=cross, K=3, fn=fn, featureSelection="top",addKnowl=FALSE, svrParams="autoOpt", lim=j)
		}
	}
	feat=mod$features
	#perfo=(perfo+mod$perf)
	#cat(paste("Overall performance= ", perfo/j, "\n"))
	if(use.cnv){
		leaderset=cbind(t(EXtest), cnvtest)
		leaderset=leaderset[,feat]
	}
	if(!use.cnv){
		leaderset=t(EXtest)[,feat]
	}
	#prin(dim(leaderset))
	e= predict(mod$fit, leaderset, type = "response", coupler = "minpair")
	#testset=t(as.matrix(datex_leader2))
	#testset=testset[-1,]	
	#testset=as.matrix(testset)
	#testset=apply(testset, 2, as.character)
	#testset=apply(testset, 2, as.numeric)
	#rownames(testset)=rownames(t(as.matrix(datex_leader2)))[2:34]
#	colnames(testset)=paste("X", colnames(testset), sep="")
#	e= predict(mod, testset, type = "response", coupler = "minpair")
	resmat[j,]=e
#	resFrame=rbind(resFrame, data.frame(Pearson=mean(re[[j]]$p), Spearman=mean(re[[j]]$s)))
#	print(resFrame)
#	print(mean(resFrame$Spearman))
	
	if(j%%100==0)
		gc()
 }
 rownames(resmat)=allgenes
 colnames(resmat)=rownames(test)
 if(outGCT){
	cat("Writing the GCT file \n")
	write.gct(resmat, gctFn=filename)
	cat(paste("GCT file written in your working directory: ", getwd(), "\n file name: ", filename, "\n"))
 }
 return(resmat)
}



write.gct <- function(exprMat, gctFn){
  nGenes = nrow(exprMat)
  nConds = ncol(exprMat)
  write("#1.2", file = gctFn, append = F) #  dummy header line.
  write(paste(nGenes, nConds, sep = "\t"), file = gctFn, append = T)
  write(paste("Name", "Description", paste(colnames(exprMat), collapse = "\t"), sep = "\t"), file = gctFn, append = T)
  # The second column of the .gct file, "Description", is filled out with "na"'s. 
  rownames(exprMat) = paste(rownames(exprMat), rownames(exprMat), sep = "\t") # Append "\tna" to every gene name. 
  write.table(exprMat, file = gctFn, append = T, quote = F, sep = "\t", na = "", row.names = T, col.names = F)
}
