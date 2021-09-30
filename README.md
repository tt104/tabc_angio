# Topological ABC for Parameter Inference of an Angiogenesis Model

This repository contains the code to reproduce the results from the publication [Topological Approximate Bayesian Computation for Parameter Inference of an Angiogenesis Model](https://arxiv.org/abs/2108.11640).

The code is formatted as a [Snakemake](https://snakemake.github.io/) workflow that uses Conda environments to install the necessary dependencies. Having installed Snakemake the workflow can be run from the root of this repository using:

	snakemake --use-conda --cores N

where N is the number of CPU cores to use in the workflow. Outputs are placed in the results folder, with the file *benchmarks.csv* containing the metrics calculated in the paper and the plots folder containing plots of the metrics, posteriors and posterior predictive simulations.

