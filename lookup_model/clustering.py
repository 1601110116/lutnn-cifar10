import numpy as np
import torch

from lookup_model.compression import kmeans_sr

# samples: (N * nblocks, subD)
# return: (npts, subD)


@torch.no_grad()
def kmeans_stochastic_relaxation(samples, npts, n_iters=100):
    codebook, codes = kmeans_sr.src(
        training_set=samples, k=npts, n_iters=n_iters)
    return codebook
