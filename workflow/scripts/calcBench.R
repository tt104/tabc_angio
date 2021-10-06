library(FNN)

ntest<-snakemake@config[["ntest"]]

# Relies on posteriors in input being ntest*2, first ntest image post. then ntest tabc post.
test_theta<-as.matrix(read.csv(snakemake@input[[1]],header=FALSE))

im_ents<-NULL
tabc_ents<-NULL

im_rsse<-NULL
tabc_rsse<-NULL
for(i in c(1:ntest))
{
	theta_im<-as.matrix(read.csv(snakemake@input[[i+1]],header=FALSE))
	ent<-entropy(theta_im,k=4)[4]
	im_ents<-c(im_ents,ent)

	rho<-theta_im[,1]-test_theta[i,1]
	chi<-theta_im[,2]-test_theta[i,2]
	rsse<-sqrt(sum(rho*rho+chi*chi))
	im_rsse<-c(im_rsse,rsse)

	theta_tabc<-as.matrix(read.csv(snakemake@input[[i+1+ntest]],header=FALSE))
	ent<-entropy(theta_tabc,k=4)[4]
	tabc_ents<-c(tabc_ents,ent)

	rho<-theta_tabc[,1]-test_theta[i,1]
	chi<-theta_tabc[,2]-test_theta[i,2]
	rsse<-sqrt(sum(rho*rho+chi*chi))
	tabc_rsse<-c(tabc_rsse,rsse)
}

benchdf<-data.frame(test=1:ntest,Statistics="Image",Entropy=im_ents,RSSE=im_rsse)
benchdf<-rbind(benchdf,data.frame(test=1:ntest,Statistics="Topological",Entropy=tabc_ents,RSSE=tabc_rsse))

write.table(benchdf,snakemake@output[[1]],row.names=FALSE,col.names=TRUE,sep=',')

meanstats<-data.frame(Statistics="Image",MeanEntropy=mean(benchdf[benchdf$Statistic=="Image","Entropy"]),MeanRSSE=mean(benchdf[benchdf$Statistic=="Image","RSSE"]))
meanstats<-rbind(meanstats,data.frame(Statistics="Topological",MeanEntropy=mean(benchdf[benchdf$Statistic=="Topological","Entropy"]),MeanRSSE=mean(benchdf[benchdf$Statistic=="Image","RSSE"])))

write.table(meanstats,snakemake@output[[2]],row.names=FALSE,col.names=TRUE,sep=',')
