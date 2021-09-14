library(FNN)
library(ggplot2)

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

pdf(snakemake@output[[2]],width=6,height=5)
p<-ggplot(benchdf)+geom_col(aes(x=test,y=Entropy,fill=Statistics),position=position_dodge())+xlab("Data set")+scale_x_continuous(breaks=1:10)
print(p)
dev.off()

pdf(snakemake@output[[3]],width=6,height=5)
p<-ggplot(benchdf)+geom_col(aes(x=test,y=RSSE,fill=Statistics),position=position_dodge())+xlab("Data set")+scale_x_continuous(breaks=1:10)
print(p)
dev.off()
