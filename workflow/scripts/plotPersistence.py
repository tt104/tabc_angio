import numpy as np
import matplotlib.pyplot as plt

pidim = int(snakemake.config["pi_dim"])

im = np.loadtxt(snakemake.input[0],delimiter=',')
im = im.reshape((3,pidim,pidim))

plt.matshow(im[0])
plt.savefig(snakemake.output[0],format="pdf",bbox_inches="tight")
plt.show()
plt.close()

plt.matshow(im[1])
plt.savefig(snakemake.output[1],format="pdf",bbox_inches="tight")
plt.show()
plt.close()

plt.matshow(im[2])
plt.savefig(snakemake.output[2],format="pdf",bbox_inches="tight")
plt.show()
plt.close()
