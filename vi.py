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
from dataclasses import dataclass

@dataclass
class Data:
    beta_mrg: torch.Tensor
    n: int
    p: int
    ld_blk: list
    ld_blk_chol: list
    blk_size: list
    
def get_posterior_stats(
    model,
    guide, 
    data, 
    phi_known, 
    num_samples=100, 
    collapsed = True
): 
    
    """ extract posterior samples (somewhat weirdly this is done with `Predictive`) """
    #guide.requires_grad_(False)
    predictive = Predictive(
        model,             
        guide=guide, 
        num_samples=num_samples) 

    samples = predictive(data)
    
    samples["beta"] = torch.zeros(num_samples,data.p)
    if collapsed: 
        for i in range(num_samples):
            phi = samples["sqrt_phi"][i]**2 if (phi_known is None) else phi_known
            psi = samples["sqrt_psi"][i]**2
            sigma2_noise = samples["sigma_noise"][i]**2
            v = phi * psi * sigma2_noise / data.n

            #sigma2 = samples["sigma_noise"][i]**2
            mm = 0
            for kk in range(len(data.ld_blk)):
                idx_blk = torch.arange(mm,mm+data.blk_size[kk])

                # should technically sample here
                samples["beta"][i,idx_blk] = torch.linalg.solve( data.ld_blk[kk] + torch.diag(v[idx_blk]), data.beta_mrg[idx_blk] )

                mm += data.blk_size[kk]
    else: 
        mm = 0
        for kk in range(len(data.ld_blk)):
            idx_blk = torch.arange(mm,mm+data.blk_size[kk])
            samples["beta"][:,idx_blk] = samples["beta_%i" % kk]
            del samples["beta_%i" % kk] # save a bit of memory

    posterior_stats = { k : {
                "mean": torch.mean(v, 0),
                "std": torch.std(v, 0),
                "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0],
                "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0],
            } for k, v in samples.items() }

    return posterior_stats

def convertr(hyperparam, name, device): 
    return torch.tensor(hyperparam, device = device) if (
        type(hyperparam) in [float,np.float32,torch.float]
    ) else pyro.sample(name, hyperparam)
        
def model_uncollapsed(data, sqrt_phi = dist.HalfCauchy(1.), desired_min_eig = 1e-6): 
    
    device = data.beta_mrg.device
    
    n_blk = len(data.ld_blk)
    
    sqrt_phi = convertr(sqrt_phi, "sqrt_phi", device = device)
    
    sqrt_psi = pyro.sample(
        "sqrt_psi",
         dist.HalfCauchy(1.).expand([data.p]).to_event(1) # PRSCS uses Strawderman-Berger here which is slightly different
    )

    sigma_noise = pyro.sample(
        "sigma_noise", 
        dist.HalfCauchy(1.) # try Jeffery's instead? Think this would be Gamma(eps,rate=eps). 
    )
    
    torch_n = torch.tensor(data.n, dtype = torch.float)
    sigma_over_sqrt_n = sigma_noise / torch.sqrt(torch_n)
    beta_global_scale = sqrt_phi * sigma_over_sqrt_n
    
    mm = 0
    for kk in range(len(blk_size)):
        assert(data.blk_size[kk] > 0)
        
        idx_blk = torch.arange(mm,mm+data.blk_size[kk])

        beta = pyro.sample(
            "beta_%i" % kk, 
            dist.Normal(0., beta_global_scale * sqrt_psi[idx_blk]).to_event(1) # already uses scale conveniently
        )
        
        obs = pyro.sample(
            "obs_%i" % kk, 
            dist.MultivariateNormal(
                ld_blk[kk] @ beta, 
                scale_tril = sigma_over_sqrt_n * data.ld_blk_chol[kk], 
            ), 
            obs = data.beta_mrg[idx_blk])
        
        mm += data.blk_size[kk]

def model_collapsed(data, sqrt_phi = dist.HalfCauchy(1.), desired_min_eig = 1e-6): 
    
    device = data.beta_mrg.device
    
    sqrt_phi = convertr(sqrt_phi, "sqrt_phi", device = device)
    phi = sqrt_phi**2
    
    sqrt_psi = pyro.sample( # constrain < 1? 
        "sqrt_psi",
         dist.HalfCauchy(1.).expand([data.p]).to_event(1) # PRSCS uses Strawderman-Berger here which is slightly different
    )
    psi = sqrt_psi**2
    
    v = psi * phi
            
    sigma_noise = pyro.sample(
        "sigma_noise", 
        dist.HalfCauchy(1.) # try Jeffery's instead
    )
    sigma2_noise = sigma_noise**2
    
    torch_n = torch.tensor(data.n, dtype = torch.float)
    sigma_over_sqrt_n = sigma_noise / torch.sqrt(torch_n)
    
    mm = 0
    for kk in range(len(data.ld_blk)):
        assert(data.blk_size[kk] > 0)
        
        idx_blk = torch.arange(mm,mm+data.blk_size[kk])
        
        #first_term = data.ld_blk[kk] @ torch.diag(v[idx_blk]) @ data.ld_blk[kk].T
        cov = (data.ld_blk[kk] * v[idx_blk]) @ data.ld_blk[kk] + data.ld_blk[kk] 
        cov = 0.5 * (cov + cov.T) # isn't _quite_ symmetric otherwise
        
        try: # faster than the eigendecomposition if already PSD
            chol_cov = torch.linalg.cholesky(cov)
        except torch._C._LinAlgError: 
            L, V = torch.linalg.eigh(cov)
            cov_min_eig = L.min().item()
            if cov_min_eig < desired_min_eig: 
                print("Degenerate cov (min eigenvalue=%1.3e)" % cov_min_eig)
                # smallest addition to diagonal to make min(eig) = min_eig
                cov += (desired_min_eig - cov_min_eig) * torch.eye(data.blk_size[kk])
            chol_cov = torch.linalg.cholesky(cov)
        #if cov.logdet().item() == -np.inf: 
            #print("Degenerate covariance, adding to diagonal")
        #    cov += 1e-2 * torch.eye(data.blk_size[kk])
            
        # equivalent to
        obs = pyro.sample(
            "obs_%i" % kk, 
            dist.MultivariateNormal(
                torch.zeros(data.blk_size[kk]), # will need to get correct dtype and device here
                scale_tril = chol_cov * sigma_over_sqrt_n
            ), 
            obs = data.beta_mrg[idx_blk])
        
        mm += data.blk_size[kk]
        
def psi_guide(data):
    
    logit_psi_loc = pyro.param("logit_psi_loc", torch.full([data.p],-1.))
    logit_psi_scale = pyro.param("logit_psi_scale", torch.full_like(logit_psi_loc, 0.1), constraint = constraints.positive)
    
    base_dist = dist.Normal(logit_psi_loc, logit_psi_scale).to_event(1)
    
    sqrt_psi = pyro.sample("sqrt_psi", dist.TransformedDistribution(base_dist, SigmoidTransform()))
    
def my_optimizer(model, guide, data, end = "\r", print_every = 10, min_iterations = 100, max_iterations = 1000, max_particles = 32, stall_window = 10, use_renyi = False, lr = 0.03):
    
    adam = pyro.optim.Adam({"lr": lr})
    svi = SVI(model, guide, adam, loss=RenyiELBO() if use_renyi else Trace_ELBO() ) 

    # train/fit model
    pyro.clear_param_store()
    
    losses = []
    num_particles = 1
       
    while num_particles <= max_particles: 
        svi = SVI(model, guide, adam, loss=Trace_ELBO(num_particles = num_particles, vectorize_particles = False))
        iteration = 0 
        while True: 
            loss = svi.step(data)
            losses.append(loss)
            iteration += 1
            if iteration % print_every == 0:
                print("[iteration %04d] loss: %.4f" % (iteration + 1, loss), end = end)
            if (num_particles > 1 or iteration > min_iterations) and iteration > stall_window: 
                R,p = scipy.stats.pearsonr(np.arange(stall_window), losses[-stall_window:])
                if p>0.05 or R>0. or iteration > max_iterations: 
                    num_particles *= 2
                    print("Done after %i iterations. Increasing num_particles to %i." % (iteration + 1, num_particles))
                    break
                    
    return losses

def preprocess_ld(ld_blk, blk_size, desired_min_eig = 1e-8):

    ld_fix = []
    ld_chol = []
    for kk,ld in enumerate(ld_blk): 
        ld = torch.tensor(ld, dtype = torch.float)
        L, V = torch.linalg.eigh(ld)
        ld_min_eig = L.min().item()
        if ld_min_eig < desired_min_eig: 
            print("Degenerate LD mat (min eigenvalue=%1.3e), fixing to %1.3e" % (ld_min_eig, desired_min_eig))
            # smallest addition to diagonal to make min(eig) = desired_min_eig
            ld += (desired_min_eig - ld_min_eig) * torch.eye(blk_size[kk])
            #L, V = torch.linalg.eigh(ld)
            #print("New min eigenvalue:%1.3e" % L.min().item())
        
        ld_fix.append(ld)
        ld_chol.append(torch.linalg.cholesky(ld))
        
    return ld_fix, ld_chol

    
def vi(
    phi, 
    sst_dict, 
    n, 
    ld_blk, 
    blk_size, 
    n_iter = 1000, 
    seed = 42, 
    eps = 1e-4, 
    collapsed = True, 
    structured_guide = True, 
    constrain_psi = True, 
    desired_min_eig = 1e-8, 
    **opt_kwargs
):
    """Variational inference for PRSCS model
    
    Doesn't support different a,b currently (just a=b=0.5 implicitly by using HalfCauchy priors)."""
    
    print('... SVI ...')

    # derived stats
    #ld_blk = [ (torch.tensor(g, dtype = torch.float) + eps * torch.eye(g.shape[0])) for g in ld_blk ]
    ld_fix, ld_blk_chol = preprocess_ld(ld_blk, blk_size, desired_min_eig = desired_min_eig)
    
    data = Data(
        beta_mrg = torch.tensor(sst_dict['BETA'], dtype = torch.float),
        p = len(sst_dict['SNP']), # number of SNPs
        n = n, 
        ld_blk = ld_fix, 
        ld_blk_chol = ld_blk_chol,
        blk_size = blk_size
    )
    
    n_blk = len(ld_blk) # number of LD blocks
    
    #print( Trace_ELBO()(model_uncollapsed, guide)(beta_mrg, n, p, ld_blk, ld_blk_chol, blk_size) )
    sqrt_phi = dist.HalfCauchy(1.) if (phi is None) else np.sqrt(phi).astype("float32")
    
    model_ = model_collapsed if collapsed else model_uncollapsed
    model = lambda dat: model_(dat, sqrt_phi = sqrt_phi, desired_min_eig = desired_min_eig)    
    
    guide = AutoGuideList(model)
    guide.add(AutoDiagonalNormal(
        poutine.block(
            model,
            expose = ["sqrt_phi", "sigma_noise"]), # or could optimize these? 
        init_loc_fn = init_to_value(values={
            "sqrt_phi" : torch.sqrt(torch.tensor(0.01)),
            "sigma_noise" : torch.sqrt(torch.tensor(0.5))
        })))
    
    guide.add(psi_guide if constrain_psi else AutoDiagonalNormal(
        poutine.block(
            model,
            expose = ["sqrt_psi"]),
        init_loc_fn = init_to_value(values={"sqrt_psi" : torch.sqrt(torch.full([p],0.1))})))
    
    if not collapsed: 
        guide_func = AutoMultivariateNormal if structured_guide else AutoDiagonalNormal
        for kk in range(n_blk):
            beta_name = "beta_%i" % kk
            guide.add(AutoMultivariateNormal( 
                poutine.block(model, expose = [beta_name]), 
                init_loc_fn = init_to_value(values={ beta_name : torch.zeros(blk_size[kk]) })))
                
    losses = my_optimizer(model, guide, data, **opt_kwargs)
                    
    stats = get_posterior_stats(model, guide, data, phi, collapsed = collapsed)
    beta_est = stats["beta"]["mean"].squeeze().cpu().numpy()
    phi_est = stats["sqrt_phi"]["mean"].item()**2 if (phi is None) else phi

    # print estimated phi
    if phi is None:
        print("Estimated global shrinkage parameter phi=%1.2e" % phi_est )
              
    print("Estimated noise std=%1.2e" % stats["sigma_noise"]["mean"])

    return losses, beta_est, phi_est, stats["sigma_noise"]["mean"].item()
