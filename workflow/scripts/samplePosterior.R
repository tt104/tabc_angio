npost<-snakemake@config[["npost"]]
ntest<-snakemake@config[["ntest"]]

params<-as.matrix(read.csv(snakemake@input[[1]],header=FALSE))
test_params<-as.matrix(read.csv(snakemake@input[[2]],header=FALSE))

abcstats<-as.matrix(read.csv(snakemake@input[[3]],header=FALSE))
teststats<-as.matrix(read.csv(snakemake@input[[4]],header=FALSE))

X<-data.frame(abcstats)
theta<-data.frame(params)
for(i in c(1:ntest))
{
	y<-teststats[i,]
	d<-sweep(X,2,y,'-')
	distsp<-sqrt(rowSums(d*d))
	post<-order(distsp)[1:npost]
	thetapost<-theta[post,]

	write.table(thetapost,snakemake@output[[i]],row.names=FALSE,col.names=FALSE,sep=',')
}
