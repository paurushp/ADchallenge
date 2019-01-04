# essen_train2=read.csv("/home/praveen/Paurush/Dream8/geno/Phase2/Achilles_v2.11_training_phase2.gct", skip=2, head=TRUE, sep="\t")
# datex_train2=read.csv("/home/praveen/Paurush/Dream8/geno/Phase2/CCLE_expression_training_phase2.gct", skip=2, head=TRUE, sep="\t", row.names=1)
# datex_leader2=read.csv("/home/praveen/Paurush/Dream8/geno/Phase2/CCLE_expression_leaderboard_phase2.gct", skip=2, head=TRUE, sep="\t", row.names=1)
# cnv_train2=read.csv("/home/praveen/Paurush/Dream8/geno/Phase2/CCLE_copynumber_training_phase2.gct", skip=2, head=TRUE, sep="\t", row.names=1)
# cnv_leader=read.csv("/home/praveen/Paurush/Dream8/geno/Phase2/CCLE_copynumber_leaderboard_phase2.gct", skip=2, head=TRUE, sep="\t", row.names=1)
# cnv_leader=cnv_leader[,-1]
# cnv_train2=cnv_train2[,-1]
# datex_train2=datex_train2[,-1]
# rownames(essen_train2)=essen_train2[,1]
# essen_train2=essen_train2[,-c(1,2)]
# datex_leader2=datex_leader2[,-1]
# save(datex_train2, essen_train2, datex_leader2, cnv_train2, cnv_leader, onco, file="testInput.rda")
DIR="/home/praveen/Paurush/Dream8/GEP/"
source(paste(DIR, "svrcodesGEPalpha.R", sep=""))
load(paste(DIR, "testInput.rda", sep=""))
myres=runLearning(train=datex_train2, yvar=essen_train2,test=datex_leader2, cross=TRUE, featureSelection="top",svrParams="autoOpt", outGCT=TRUE, filename="prediction.gct")
#### Anovadot nu SVR correlated+knowl 
myres=runLearning(train=datex_train2, yvar=essen_train2,test=datex_leader2, cross=TRUE, K=3, featureSelection="top", svrParams="autoOpt", addKnowl=TRUE, knowProb=onco, outGCT=FALSE, fie="prediction.gct")
#### Anovadot nu SVR correlated 
myres=runLearning(train=datex_train2, yvar=essen_train2,test=datex_leader2, cross=TRUE, K=3, featureSelection="top", svrParams="autoOpt", addKnowl=TRUE, knowProb=onco, outGCT=FALSE, fie="prediction.gct")
#### Anovadot eps-SVR correlated 
myres=runLearning(train=datex_train2, yvar=essen_train2,test=datex_leader2, cross=TRUE, K=3, featureSelection="top", svrParams="autoOpt", addKnowl=TRUE, knowProb=onco, outGCT=FALSE, fie="prediction.gct")
#### Anovadot eps-SVR correlated +knowl +CNV
myres=runLearning(EXtrain=datex_train2, yvar=essen_train2, use.cnv=TRUE, EXtest=datex_leader2, cnv.train=cnv_train2, cnv.test=cnv_leader, cross=FALSE, K=3,  featureSelection="top", svrParams="autoOpt", addKnowl=TRUE, knowProb=onco, fn=2000, outGCT=FALSE, file="prediction.gct")

#### Anovadot eps-SVR correlated +knowl +CNV
myres=runLearning(EXtrain=datex_train2, yvar=essen_train2, use.cnv=TRUE, EXtest=datex_leader2, cnv.train=cnv_train2, cnv.test=cnv_leader, cross=FALSE, K=3,  featureSelection="top", svrParams="autoOpt", addKnowl=TRUE, knowProb=onco, fn=3000, outGCT=FALSE, file="prediction.gct")

writeGCT <- function(gct, filename, check.file.extension = TRUE){
	if(check.file.extension){
		filename <- .checkExtension(filename, ".gct")
	}
	f <- file(filename, "w")
	on.exit (close (f))
	cat("#1.2", "\n", file = f, append = TRUE, sep = "")
	cat(dim (gct$data)[1], "\t", dim(gct$data)[2], "\n", file = f, append = TRUE, sep = "")
	cat("Name", "\t", file = f, append = TRUE, sep = "")
	cat("Description", file = f, append = TRUE, sep = "")
	names <- colnames(gct$data)
	for(j in 1:length(names)) {
		cat("\t", names[j], file = f, append = TRUE, sep = "")
	}
	cat("\n", file = f, append = TRUE, sep = "")
	m <- matrix(nrow = dim (gct$data)[1], ncol = 2)
	m[, 1] <- row.names(gct$data)
	if(!is.null(gct$row.descriptions)) {
		m[, 2] <- gct$row.descriptions
	} else{
		m[, 2] <- ""
	}
	m <- cbind(m, gct$data)
	write.table(m, file = f, append = TRUE, quote = FALSE, sep = "\t", eol = "\n", col.names = FALSE, row.names = FALSE)
	filename
}
