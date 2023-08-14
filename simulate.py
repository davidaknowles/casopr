import numpy as np
import pyro.distributions as dist
import torch

def simulate_sumstats(ld_blk, blk_size, n_gwas, p, prop_nz = 0.2, beta_sd = 0.1, sigma_noise = 1. ): 
    sigma_over_sqrt_n = sigma_noise / torch.sqrt(torch.tensor(n_gwas))
    nz = torch.rand(p) < prop_nz
    beta_true = torch.where(nz, beta_sd * torch.randn(p), torch.zeros(p))

    annotations = torch.stack([torch.ones(p),nz]).T

    beta_mrg = torch.zeros(p)
    mm = 0
    for kk in range(len(ld_blk)):
        idx_blk = torch.arange(mm,mm+blk_size[kk])
        ld_torch = torch.tensor(ld_blk[kk], dtype = torch.float)
        L, V = torch.linalg.eigh(ld_torch)
        L[L < 0.] = 0.

        beta_mrg[idx_blk] = ld_torch @ beta_true[idx_blk] + sigma_over_sqrt_n * (V @ torch.diag(L.sqrt())) @ torch.randn(blk_size[kk])
        #ld_torch @ beta_true[idx_blk], 
        # covariance_matrix = ld_torch * sigma_over_sqrt_n**2).rsample()
        mm += blk_size[kk]

    return beta_true, beta_mrg, annotations
