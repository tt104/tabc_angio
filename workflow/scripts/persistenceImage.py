import numpy as np
import sys
from gudhi.representations.vector_methods import PersistenceImage

pidim = int(snakemake.config["pi_dim"])
bw = float(snakemake.config["pi_bw"])

doPI = PersistenceImage(bandwidth=bw,resolution=[pidim, pidim],im_range=[0, 50, 0, 50])

nper = 4
ndim = 3

pimstats = np.zeros((nper,ndim,pidim*pidim))

# read in extended persistence
pdata = np.load(snakemake.input[0],allow_pickle=True)
# ordinary, relative, ext+, ext-
for j in range(nper):
    pd = pdata[()]['EPD'][j]
    for k in range(ndim):
        if j==0 or j==2:
            diag = [(a,b) for (d,(a,b)) in pd if d==k]
        else:
            diag = [(b,a) for (d,(a,b)) in pd if d==k]
        if len(diag)>0:
            npd = np.array(diag)
            pim = doPI(npd)
            pimstats[j,k,:] = pim
        else:
            pimstats[j,k,:] = 0

pimtxt = pimstats.reshape((1,nper*ndim*pidim*pidim))

np.savetxt(snakemake.output[0],pimtxt,delimiter=",",fmt='%1.9f')
