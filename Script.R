source("SVMRFE-featureSelect.R") # All data sets 
library(FSelector)
library(party)
library(e1071)
library(kernlab)
library(reshape)
library(ggplot2)
library(FNN)

## Leave one out Cross validation Clinical

act=td$MMSE24
res=t(data.frame(act))
# pb = txtProgressBar(1, nrow(td), 1, style=3)
# flush.console()
f=7
it=c(3,4, 5,10,15)
for(g in 1:5){
iter=it[g]
	prediction.DT=predBL=predDT_bag=predBL_bag=predDT_bagMax=predBL_bagMax=c()
		for(i in 1:nrow(td)){
			X=td
			#X[, c(4,5,10)]=scale(X[, c(4,5,10)]) 
# 			flush.console()
			test=X[i,]
			train=X[-i,]
			cat(i)
			cat("\n")
			ep=(3*sd(train$MMSE24)*(sqrt(log(nrow(train)))/(nrow(train))))
# 			cat("1###\n")
			C=max(abs(mean(train$MMSE24)+3*sd(train$MMSE24)), abs(mean(train$MMSE24)-3*sd(train$MMSE24)))#min(, j+1)
# 			cat("2###\n")
# 			f.svm = e1071::svm(train[, -ncol(train)], train[, ncol(train)],cost=C,kernel="sigmoid",epsilon=ep, type="eps-regression", scale=FALSE)
			features=wrapperFS(train, scale=FALSE) # RUN RFE-SVR feature selection
			features=c(features$allfeatures[features$feature.ids[1:f]], "MMSE24")
			f.svm = ksvm(MMSE24 ~., data = train[,features], eps=ep,C =C, type = "eps-svr", kernel="laplacedot", na.action = na.omit)#[,-6]
#			f.mlp=fit(MMSE24~CN+MCI+AGE+EDU+A2+A3+A4+cts_mmse30,data_here, model="mlpe")
			BLres.svm=predict(f.svm, test)
			BLBagres=bootbag(train=train, test=test, iter=iter, aggregate="mean", type="eps-svr", kernel="laplacedot", f=7)
			predBL=c(predBL, BLres.svm)
			#predBL_bag=c(predBL_bag, BLBagres)
			BLBagresMax=bootbag(train=train, test=test[,colnames(train)], iter=iter, aggregate="max", type="eps-svr", kernel="laplacedot", f=7)
			predBL_bagMax=c(predBL_bagMax, BLBagresMax)
			if(test$CN==1){
				data_here=subset(train, CN==1)
				features=wrapperFS(data_here[,-c(5,6,7)], scale=FALSE)
				features=c(features$allfeatures[features$feature.ids[1:f]], "MMSE24")
				data_here=data_here[,features]
				ep=(3*sd(data_here$MMSE24)*(sqrt(log(nrow(data_here)))/(nrow(data_here))))
# 				cat("1###\n")
				C=max(abs(mean(data_here$MMSE24)+sd(data_here$MMSE24)), abs(mean(data_here$MMSE24)-sd(data_here$MMSE24)))#min(, j+1)
# 				cat("2###\n")
# 				f.svm = e1071::svm(train[, -ncol(train)], train[, ncol(train)],cost=C,kernel="sigmoid",epsilon=ep, type="eps-regression", scale=FALSE)
				f.svm = ksvm(MMSE24 ~., data = data_here, eps=ep,C =C, type = "eps-bsvr", kernel="laplacedot", na.action = na.omit)#[,-6]
#				f.mlp=fit(MMSE24~CN+MCI+AGE+EDU+A2+A3+A4+cts_mmse30,data_here, model="mlpe")
				res.svm=predict(f.svm, test)
				prediction.DT=c(prediction.DT, res.svm)
				DTBagres=bootbag(train=train, test=test, iter=iter, aggregate="mean", type="eps-bsvr", kernel="laplacedot", f=7)
				predDT_bag=c(predDT_bag, DTBagres)
				DTBagresMax=bootbag(train=train, test=test[,colnames(train)], iter=iter, aggregate="max", type="eps-bsvr", kernel="laplacedot", f=7)
				predDT_bagMax=c(predDT_bagMax, DTBagresMax)
			}
			if(test$MCI==1){
				data_here=subset(train, AD==1|MCI==1|CN==1)#rbind(train)#, subset(X,AD==1), subset(X,CN==1))
				features=wrapperFS(data_here[,-c(5,6,7)])
				features=c(features$allfeatures[features$feature.ids[1:f]], "MMSE24")
				data_here=data_here[,features]
				ep=(3*sd(data_here$MMSE24)*(sqrt(log(nrow(data_here)))/(nrow(data_here))))
# 				cat("1###\n")
				C=max(abs(mean(data_here$MMSE24)+3*sd(data_here$MMSE24)), abs(mean(data_here$MMSE24)-3*sd(data_here$MMSE24)))#min(, j+1)
				f.svm = e1071::svm(train[, -ncol(train)], train[, ncol(train)],cost=C,kernel="sigmoid",epsilon=ep, type="eps-regression", scale=FALSE)
# 				cat("2###\n")
				f.svm = ksvm(MMSE24 ~., data = data_here, eps=0.01,C =0.05, type = "eps-bsvr", kernel="laplacedot", na.action = na.omit)#[,-6]# 0.01, 0.5
# 				f.mlp=fit(MMSE24~CN+MCI+AGE+EDU+A2+A3+A4+cts_mmse30,data_here, model="mlpe")
				res.svm=predict(f.svm, test)
				prediction.DT=c(prediction.DT, res.svm)
				DTBagres=bootbag(train=train, test=test, iter=iter, aggregate="mean", type="eps-bsvr", kernel="laplacedot", f=7)
				predDT_bag=c(predDT_bag, DTBagres)
				DTBagresMax=bootbag(train=train, test=test[,colnames(train)], iter=iter, aggregate="max", type="eps-bsvr", kernel="laplacedot", f=7)
				predDT_bagMax=c(predDT_bagMax, DTBagresMax)
			}
			if(test$AD==1){
# 				if(test$A4==2){
# 					data_here=subset(train, A4==2 & AD==1)
# 					features=wrapperFS(data_here[,-c(5,6,7,8,9,10)])
# 					features=c(features$allfeatures[features$feature.ids[1:f]], "MMSE24")
# 					data_here=data_here[,features]
# # 					f.svm = e1071::svm(train[, -ncol(train)], train[, ncol(train)],cost=C,kernel="sigmoid",epsilon=ep, type="eps-regression", scale=FALSE)
# # 					ep=(3*sd(data_here$MMSE24)*(sqrt(log(nrow(data_here)))/(nrow(data_here))))
# # 					cat("1###\n")
# # 					C=max(abs(mean(data_here$MMSE24)+3*sd(data_here$MMSE24)), abs(mean(data_here$MMSE24)-3*sd(data_here$MMSE24)))#min(, j+1)
# # 					cat("2###\n")
# 					f.svm = ksvm(MMSE24 ~., data = train, eps=ep,C =C, type = "nu-svr", kernel="laplacedot", na.action = na.omit)#[,-6]# ep, 1
# # 					f.mlp=fit(MMSE24~AGE+EDU+A2+A3+A4+cts_mmse30,train, model="mlpe")
# 					res.svm=predict(f.svm, test)
# 					prediction.DT=c(prediction.DT, res.svm)
# 				}else{
				data_here=subset(train, AD==1|MCI==1)
				features=wrapperFS(data_here[,-c(5,6,7)])
				features=c(features$allfeatures[features$feature.ids[1:f]], "MMSE24")
				data_here=data_here[,features]
				f.svm = e1071::svm(train[, -ncol(train)], train[, ncol(train)],cost=C,kernel="sigmoid",epsilon=ep, type="eps-regression", scale=FALSE)
				ep=(3*sd(data_here$MMSE24)*(sqrt(log(nrow(data_here)))/(nrow(data_here))))
# 				cat("1###\n")
				C=max(abs(mean(data_here$MMSE24)+3*sd(data_here$MMSE24)), abs(mean(data_here$MMSE24)-3*sd(data_here$MMSE24)))#min(, j+1)
# 				cat("2###\n")
				f.svm = ksvm(MMSE24 ~., data = train, eps=ep,C =C, type = "nu-svr", kernel="laplacedot", na.action = na.omit)#[,-6]# ep, 1
# 				f.mlp=fit(MMSE24~AGE+EDU+A2+A3+A4+cts_mmse30,train, model="mlpe")
				res.svm=predict(f.svm, test)
				prediction.DT=c(prediction.DT, res.svm)
				DTBagres=bootbag(train=train, test=test, iter=iter, aggregate="mean", type="nu-svr", kernel="laplacedot", f=7)
				predDT_bag=c(predDT_bag, DTBagres)
				DTBagresMax=bootbag(train=train, test=test[,colnames(train)], iter=iter, aggregate="max", type="eps-bsvr", kernel="laplacedot", f=7)
				predDT_bagMax=c(predDT_bagMax, DTBagresMax)
# 				}
			}
		cat(paste("Score for DTSVR= ", cor(act[1:i], prediction.DT), "\n"))
		cat(paste("Score for BL-SVR= ", cor(act[1:i], predBL), "\n"))
		cat(paste("Score for DT-SVRBag= ", cor(act[1:i], predDT_bag), "\n"))
		cat(paste("Score for BL-SVRBag= ", cor(act[1:i], predBL_bag), "\n"))
		cat(paste("Score for BL-SVRBagMax= ", cor(act[1:i], predBL_bagMax), "\n"))
		cat(paste("Score for DT-SVRBagMax= ", cor(act[1:i], predDT_bagMax), "\n"))
		}
	res=rbind(res, predBL, prediction.DT, predDT_bag)
	#res=rbind(res, predBL, prediction.DT)
}

res=t(res)

res=data.frame(Actual=act, Baseline=predBL, DTSVR=prediction.DT)
cor(res[,1], res[,2])
cor(res[,1], res[,3])
cor(res[,1], res[,3], method="spearman")
cor(res[,1], res[,2], method="spearman")

rownames(res)=rownames(td)
cor(res[rownames(subset(td, AD==1)),1], res[rownames(subset(td, AD==1)),2])
cor(res[rownames(subset(td, AD==1)),1], res[rownames(subset(td, AD==1)),3])
cor(res[rownames(subset(td, CN==1)),1], res[rownames(subset(td, CN==1)),2])
cor(res[rownames(subset(td, CN==1)),1], res[rownames(subset(td, CN==1)),3])
cor(res[rownames(subset(td, MCI==1)),1], res[rownames(subset(td, MCI==1)),3])
cor(res[rownames(subset(td, MCI==1)),1], res[rownames(subset(td, MCI==1)),2])
############################################################################
### Ensemble learning by adaptive bootstrap bagging (via nearest neighbor)
############################################################################



bootbag=function(train, test, iter, aggregate=c("mean", "max", "min","meadian", "MP.RNK"), fs=svmrfe,type="eps-svr", kernel="laplacedot", f=7,ep=NULL, C=NULL){
	y_here=c()
# 	print(colnames(train))
# 	print(colnames(test))
# 	index=get.knnx(train[,-ncol(train)], test[,-ncol(test)], k=350, algorithm="kd_tree")
# 	index=c(index$nn.index)
# 	train=train[index,]
	for(i in 1:iter){
		n=nrow(train)
		fract=round(nrow(train)/iter)
		ind=sample(n, fract)
		train_here=train[-ind,]
		train_here=train_here[,which(!apply(train_here==0,2,all))]
		if(is.null(ep)){
			ep=(3*sd(train_here$MMSE24)*(sqrt(log(nrow(train_here)))/(nrow(train_here))))
		}
		if(is.null(C)){
			C=max(abs(mean(train_here$MMSE24)+3*sd(train_here$MMSE24)), abs(mean(train_here$MMSE24)-3*sd(train_here$MMSE24)))#min(, j+1)
		}
		if(fs=="svmrefe"){
		features=wrapperFS(train_here, scale=FALSE) # RUN RFE-SVR feature selection
		features=c(features$allfeatures[features$feature.ids[1:f]], "MMSE24")
		}
		if(fs=="rfor"){
		features=random.forest.importance(MMSE24~., train_here, importance.type = 1)
		features=colnames(td)[order(features, decreasing=TRUE)[1:f]]
		}
		features=c(features, "MMSE24")
		f.svm = ksvm(MMSE24 ~., data = train_here[,features], eps=ep,C =C, type = type, kernel=kernel, na.action = na.omit)#[,-6]
		res_here=predict(f.svm, test)
		y_here=c(y_here, res_here)
	}
	if(aggregate=="mean"){
		y_here=mean(y_here)
	}
	if(aggregate=="median"){
		y_here=median(y_here)
	}
	if(aggregate=="max"){
		sign=mean(y_here)/abs(mean(y_here))
		y_here=max(y_here)
		y_here=sign*y_here
		
	}
	if(aggregate=="min"){
		y_here=min(y_here)
	}
	if(aggregate=="MP.RNK"){
		y_here=mprank(y_here)
	}
	return(y_here)
}







#### Submission Without feature selection
############################clini
ftsdal=cbind(geno.final, ftsd)
train_mod0=subset(td,CN==1)
 train_mod1=subset(td,MCI==1)
 train_mod2=subset(td,AD==1)
 ep0=min(3*sd(subset(td, CN==1)$MMSE24)*(sqrt(log(nrow(subset(td, CN==1, select=c(AGE, EDU,ap2,ap3,ap4,cts_mmse30,MMSE24))))/nrow(subset(td, CN==1)))),1)
 ep2=min(3*sd(subset(td, MCI==1)$MMSE24)*(sqrt(log(nrow(subset(td, MCI==1, select=c(AGE, EDU,ap2,ap3,ap4,cts_mmse30,MMSE24))))/nrow(subset(td, MCI==1)))),0.5)
 ep3=min(3*sd(subset(td, AD==1)$MMSE24)*(sqrt(log(nrow(subset(td, AD==1,select=c(AGE, EDU,ap2,ap3,ap4,cts_mmse30,MMSE24))))/nrow(subset(td, AD==1)))),1)
 f0 = ksvm(MMSE24 ~., data = train_mod0, C = 10,epsilon=ep0, type = "eps-svr", kernel="laplacedot", na.action = na.omit, verbose=FALSE)#[,-6]
 test_mod00=subset(ftsd,CN==1)
 test_mod11=subset(ftsd,MCI==1)
 test_mod22=subset(ftsd,AD==1)
 g00 = predict(f0, test_mod00, type = "response", coupler = "minpair")#[,-6]
 f1 = ksvm(MMSE24 ~., data = train_mod1 , C = 20, epsilon=ep1, type = "eps-svr", kernel="laplacedot", na.action = na.omit, verbose=FALSE)
 g11= predict(f1, test_mod11, type = "response", coupler = "minpair")
 f2 = ksvm(MMSE24 ~., data = train_mod2, C = 1000, epsilon=ep2, type = "eps-svr", kernel="laplacedot", na.action = na.omit, verbose=FALSE)
 g22 = predict(f2, test_mod22, type = "response", coupler = "minpair")
 
 rownames(g00)=rownames(test_mod00)
 rownames(g11)=rownames(test_mod11)
 rownames(g22)=rownames(test_mod22)
 
 G=rbind(g00,g11,g22)
 De=G[rownames(ftsd),]#-tsd[,6]
##############################Geno Final
train_mod0=subset(tdal,CN==1)
 train_mod1=subset(tdal,MCI==1)
 train_mod2=subset(tdal,AD==1)
 ep0=min(3*sd(subset(td, CN==1)$MMSE24)*(sqrt(log(nrow(subset(td, CN==1, select=c(AGE, EDU,ap2,ap3,ap4,cts_mmse30,MMSE24))))/nrow(subset(td, CN==1)))),1)
 ep2=min(3*sd(subset(td, MCI==1)$MMSE24)*(sqrt(log(nrow(subset(td, MCI==1, select=c(AGE, EDU,ap2,ap3,ap4,cts_mmse30,MMSE24))))/nrow(subset(td, MCI==1)))),0.5)
 ep3=min(3*sd(subset(td, AD==1)$MMSE24)*(sqrt(log(nrow(subset(td, AD==1,select=c(AGE, EDU,ap2,ap3,ap4,cts_mmse30,MMSE24))))/nrow(subset(td, AD==1)))),1)
 f0 = ksvm(MMSE24 ~., data = train_mod0, C = 10,epsilon=ep0, type = "eps-svr", kernel="laplacedot", na.action = na.omit, verbose=FALSE)#[,-6]
 test_mod00=subset(ftsdal,CN==1)
 test_mod11=subset(ftsdal,MCI==1)
 test_mod22=subset(ftsdal,AD==1)
 g00 = predict(f0, test_mod00, type = "response", coupler = "minpair")#[,-6]
 f1 = ksvm(MMSE24 ~., data = train_mod1 , C = 20, epsilon=ep1, type = "eps-svr", kernel="laplacedot", na.action = na.omit, verbose=FALSE)
 g11= predict(f1, test_mod11, type = "response", coupler = "minpair")
 f2 = ksvm(MMSE24 ~., data = train_mod2, C = 100, epsilon=ep2, type = "eps-svr", kernel="laplacedot", na.action = na.omit, verbose=FALSE)
 g22 = predict(f2, test_mod22, type = "response", coupler = "minpair")
 
 rownames(g00)=rownames(test_mod00)
 rownames(g11)=rownames(test_mod11)
 rownames(g22)=rownames(test_mod22)
 
 G=rbind(g00,g11,g22)
 De=G[rownames(tsd),]#-tsd[,6]
