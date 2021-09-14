npred<-snakemake@config[["npred"]]
npost<-snakemake@config[["npost"]]

posterior<-as.matrix(read.csv(snakemake@input[[1]],header=FALSE))

ix<-sample(1:npred,replace=TRUE)

predictive<-posterior[ix,]

write.table(predictive,snakemake@output[[1]],row.names=FALSE,col.names=FALSE,sep=',')

