import torch
import pyro
from pyro import poutine
import pyro.distributions as dist
from pyro.nn import PyroSample, PyroModule
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal, AutoGuideList, AutoDelta, init_to_value
from pyro.infer import SVI, Trace_ELBO, RenyiELBO
from pyro.infer import Predictive
from pyro.distributions import constraints
from pyro.distributions.util import eye_like

from torch.distributions.transforms import AffineTransform, SigmoidTransform

from scipy.stats import norm
import scipy.stats

import numpy as np
import pandas as pd
from dataclasses import dataclass

import scipy as sp


def get_posterior_stats(
    model,
    guide, 
    data, 
    num_samples=100, 
    collapsed = True
): 
    
    """ extract posterior samples (somewhat weirdly this is done with `Predictive`) """
    #guide.requires_grad_(False)
    predictive = Predictive(
        model,             
        guide=guide, 
        num_samples=num_samples) 

    samples = predictive(*data)
    
    if collapsed: 
        beta_mrg, n, p, ld_blk, ld_blk_chol, blk_size = data
        samples["beta"] = torch.zeros(num_samples,p)
        for i in range(num_samples):
            
            phi = samples["sqrt_phi"][i]**2
            psi = samples["sqrt_psi"][i]**2
            v = phi * psi
            
            #sigma2 = samples["sigma_noise"][i]**2
            mm = 0
            for kk in range(len(ld_blk)):
                idx_blk = torch.arange(mm,mm+blk_size[kk])
                # should technically sample here
                samples["beta"][i,idx_blk] = torch.linalg.solve( ld_blk[kk] + torch.diag(v[idx_blk]), beta_mrg[idx_blk] )
                mm += blk_size[kk]

    posterior_stats = { k : {
                "mean": torch.mean(v, 0),
                "std": torch.std(v, 0),
                "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0],
                "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0],
            } for k, v in samples.items() }

    return posterior_stats


def model_uncollapsed(beta_mrg, n, p, ld_blk, ld_blk_chol, blk_size): 
    
    n_blk = len(ld_blk)
    
    sqrt_phi = pyro.sample( "sqrt_phi", dist.HalfCauchy(1.) )
    
    sqrt_psi = pyro.sample(
        "sqrt_psi",
         dist.HalfCauchy(1.).expand([p]).to_event(1) # PRSCS uses Strawderman-Berger here which is slightly different
    )
        
    sigma_noise = pyro.sample(
        "sigma_noise", 
        dist.HalfCauchy(1.) # try Jeffery's instead? Think this would be Gamma(eps,rate=eps). 
    )
    
    torch_n = torch.tensor(n, dtype = torch.float)
    beta = pyro.sample(
        "beta", 
        dist.Normal(0., (sqrt_phi * sigma_noise / torch.sqrt(torch_n)) * sqrt_psi).to_event(1) # already uses scale conveniently
    )
    
    mm = 0
    for kk in range(n_blk):
        assert(blk_size[kk] > 0)
        
        idx_blk = torch.arange(mm,mm+blk_size[kk])
        
        obs = pyro.sample(
            "obs_%i" % kk, 
            dist.MultivariateNormal(
                ld_blk[kk] @ beta[idx_blk], 
                scale_tril = sigma_noise * ld_blk_chol[kk] / n, 
            ), 
            obs = beta_mrg[idx_blk])
        
        mm += blk_size[kk]
        
        
def model_uncollapsed_structured(beta_mrg, n, p, ld_blk, ld_blk_chol, blk_size): 
    
    n_blk = len(ld_blk)
    
    sqrt_phi = pyro.sample( "sqrt_phi", dist.HalfCauchy(1.) )
    
    sqrt_psi = pyro.sample(
        "sqrt_psi",
         dist.HalfCauchy(1.).expand([p]).to_event(1) # PRSCS uses Strawderman-Berger here which is slightly different
    )
        
    sigma_noise = pyro.sample(
        "sigma_noise", 
        dist.HalfCauchy(1.) # try Jeffery's instead? Think this would be Gamma(eps,rate=eps). 
    )
    
    torch_n = torch.tensor(n, dtype = torch.float)
    beta_global_scale = sqrt_phi * sigma_noise / torch.sqrt(torch_n)
    
    mm = 0
    for kk in range(n_blk):
        assert(blk_size[kk] > 0)
        
        idx_blk = torch.arange(mm,mm+blk_size[kk])

        beta = pyro.sample(
            "beta_%i" % kk, 
            dist.Normal(0., beta_global_scale * sqrt_psi[idx_blk]).to_event(1) # already uses scale conveniently
        )
        
        obs = pyro.sample(
            "obs_%i" % kk, 
            dist.MultivariateNormal(
                ld_blk[kk] @ beta, 
                scale_tril = sigma_noise * ld_blk_chol[kk] / n, 
            ), 
            obs = beta_mrg[idx_blk])
        
        mm += blk_size[kk]

def model_collapsed(beta_mrg, n, p, ld_blk, ld_blk_chol, blk_size): 
    
    #print("model")
    
    n_blk = len(ld_blk)
    
    sqrt_phi = pyro.sample( "sqrt_phi", dist.HalfCauchy(1.) )
    phi = sqrt_phi**2
    
    sqrt_psi = pyro.sample( # constrain < 1? 
        "sqrt_psi",
         dist.HalfCauchy(1.).expand([p]).to_event(1) # PRSCS uses Strawderman-Berger here which is slightly different
    )
    psi = sqrt_psi**2
    
    v = psi * phi
            
    sigma_noise = pyro.sample(
        "sigma_noise", 
        dist.HalfCauchy(1.) # try Jeffery's instead
    )
    sigma2_noise = sigma_noise**2
    
    mm = 0
    for kk in range(n_blk):
        assert(blk_size[kk] > 0)
        
        idx_blk = torch.arange(mm,mm+blk_size[kk])
        
        cov = sigma2_noise * ( (ld_blk[kk] * v[idx_blk]) @ ld_blk[kk] + ld_blk[kk] ) / n
        
        if cov.det().item()==0.: 
            #print("Degenerate covariance, adding to diagonal")
            cov += 1e-2 * torch.eye(blk_size[kk])
            
        # equivalent to
        # cov = sigma2_noise * ( (ld_blk[kk] @ torch.diag(v[idx_blk]) @ ld_blk[kk] + ld_blk[kk] ) / n
        obs = pyro.sample(
            "obs_%i" % kk, 
            dist.MultivariateNormal(
                torch.zeros(blk_size[kk]), # will need to get correct dtype and device here
                covariance_matrix = cov
            ), 
            obs = beta_mrg[idx_blk])
        
        mm += blk_size[kk]
        
def psi_guide(beta_mrg, n, p, ld_blk, ld_blk_chol, blk_size):
    
    #print("guide")
    
    logit_psi_loc = pyro.param("logit_psi_loc", torch.full([p],-1.))
    logit_psi_scale = pyro.param("logit_psi_scale", torch.full_like(logit_psi_loc, 0.1), constraint = constraints.positive)
    
    #logit_psi = pyro.sample(
    #    "logit_psi", 
    #    dist.Normal(logpsi_loc, logpsi_scale).to_event(1), 
    #    infer={'is_auxiliary': True}
    #)
    #psi = pyro.sample(
    #    "sqrt_psi", 
    #    dist.Delta(torch.sigmoid(logit_psi))
    #)
    
    base_dist = dist.Normal(logit_psi_loc, logit_psi_scale).to_event(1)
    
    sqrt_psi = pyro.sample("sqrt_psi", dist.TransformedDistribution(base_dist, SigmoidTransform()))
    
def vi(phi, sst_dict, n, ld_blk, blk_size, n_iter, chrom, out_dir, beta_std, seed, eps = 1e-4, use_renyi = False, lr = 0.03, print_every = 30, collapsed = True, structured_guide = True, stall_window = 10, max_particles = 32, end = "\r"):
    """Variational inference for PRSCS model
    
    Doesn't support different a,b currently (just a=b=0.5 implicitly by using HalfCauchy priors)."""
    
    print('... SVI ...')

    # derived stats
    beta_mrg = torch.tensor(sst_dict['BETA'], dtype = torch.float)
    maf = np.array(sst_dict['MAF']) # only used to rescale beta after inference
    p = len(sst_dict['SNP']) # number of SNPs
    n_blk = len(ld_blk) # number of LD blocks
    
    ld_blk = [ (torch.tensor(g, dtype = torch.float) + eps * torch.eye(g.shape[0])) for g in ld_blk ]

    ld_blk_chol = [ torch.linalg.cholesky(g) for g in ld_blk ]
    
    #print( Trace_ELBO()(model_uncollapsed, guide)(beta_mrg, n, p, ld_blk, ld_blk_chol, blk_size) )
    model = model_collapsed if collapsed else (model_uncollapsed_structured if structured_guide else model_uncollapsed)
    
    guide = AutoGuideList(model)
    guide.add(AutoDiagonalNormal(
        poutine.block(
            model,
            expose = ["sqrt_phi", "sigma_noise"]), # or could optimize these? 
        init_loc_fn = init_to_value(values={
            "sqrt_phi" : torch.sqrt(torch.tensor(0.01)),
            "sigma_noise" : torch.sqrt(torch.tensor(0.5))
        })))
    
    guide.add(psi_guide)
    
    if not collapsed: 
        if structured_guide: 
            for kk in range(n_blk):
                beta_name = "beta_%i" % kk
                guide.add(AutoMultivariateNormal( 
                    poutine.block(model, expose = [beta_name]), 
                    init_loc_fn = init_to_value(values={ beta_name : torch.zeros(blk_size[kk]) })))
        else: 
            guide.add(AutoDiagonalNormal(
                 poutine.block(model, expose = ["beta"]), 
                 init_loc_fn = init_to_value(values={ "beta" : torch.zeros(p) })))
    
    adam = pyro.optim.Adam({"lr": lr})
    svi = SVI(model, guide, adam, loss=RenyiELBO() if use_renyi else Trace_ELBO() ) 

    data = [ beta_mrg, n, p, ld_blk, ld_blk_chol, blk_size ]
    
    # train/fit model
    pyro.clear_param_store()

    losses = []
    num_particles = 1
       
    while num_particles <= max_particles: 
        svi = SVI(model, guide, adam, loss=Trace_ELBO(num_particles = num_particles, vectorize_particles = False))
        iteration = 0 
        while True: 
            loss = svi.step(*data)
            losses.append(loss)
            iteration += 1
            if iteration % print_every == 0:
                print("[iteration %04d] loss: %.4f" % (iteration + 1, loss), end = end)
            if iteration > stall_window: 
                R,p = scipy.stats.pearsonr(np.arange(stall_window), losses[-stall_window:])
                if p>0.05 or R>0. or iteration > n_iter: 
                    num_particles *= 2
                    print("Stalled after %i iterations. Increasing num_particles to %i." % (iteration + 1, num_particles))
                    break
                    
    stats = get_posterior_stats(model, guide, data, collapsed = collapsed)
    beta_est = stats["beta"]["mean"].squeeze()
    phi_est = stats["sqrt_phi"]["mean"].item()**2
    
    # convert standardized beta to per-allele beta
    if beta_std == 'False':
        beta_est /= sp.sqrt(2.0*maf*(1.0-maf))

    # write posterior effect sizes
    eff_file = out_dir + '_pst_eff_phi%s_chr%d.txt' % ("auto" if (phi is None) else str(phi_est), chrom)

    with open(eff_file, 'w') as ff:
        for snp, bp, a1, a2, beta in zip(sst_dict['SNP'], sst_dict['BP'], sst_dict['A1'], sst_dict['A2'], beta_est):
            ff.write('%d\t%s\t%d\t%s\t%s\t%.6e\n' % (chrom, snp, bp, a1, a2, beta))

    # print estimated phi
    if phi is None:
        print("Estimated global shrinkage parameter phi=%1.2e" % phi_est )
              
    print("Estimated noise std=%1.2e" % stats["sigma_noise"]["mean"])

    return losses

