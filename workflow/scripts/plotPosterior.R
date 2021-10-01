library(ggplot2)
library(colorspace)

thetas<-read.csv(snakemake@input[[1]],header=FALSE)
test_param<-read.csv(snakemake@input[[2]],header=FALSE)

test_param_x<-test_param[as.numeric(snakemake@params[["index"]])+1,1]
test_param_y<-test_param[as.numeric(snakemake@params[["index"]])+1,2]

pdf(snakemake@output[[1]],width=3,height=3)
p<-ggplot(thetas,aes(V1,V2))+geom_density_2d(aes(color=..level..))+geom_point(aes(x=test_param_x,y=test_param_y),colour='red',stroke=1,size=2,shape=4)+xlim(0,0.5)+ylim(0,0.5)+labs(x=expression(rho),y=expression(chi))+scale_color_continuous_sequential(palette="ag_Sunset")+theme(legend.position="none",text = element_text(size = 14))
print(p)
dev.off()
