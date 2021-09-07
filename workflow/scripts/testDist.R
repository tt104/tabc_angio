nsamp<-10

# Sample 10 parameters from prior
samples_hapt<-runif(nsamp,min=0,max=0.5)
samples_chi<-runif(nsamp,min=0,max=0.5)

samples<-cbind(samples_hapt,samples_chi)
write.table(samples,"inputs/test_params.csv",row.names=FALSE,col.names=FALSE,sep=',')

