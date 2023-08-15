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
    annotations: torch.Tensor
    torch_type: dict
    
def get_posterior_stats(
    model,
    guide, 
    data, 
    phi_known, 
    phi_as_prior, 
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
    
    samples["beta"] = torch.zeros(num_samples,data.p,**data.torch_type)
    if collapsed: 
        for i in range(num_samples):
            if data.annotations is None: 
                phi = samples["sqrt_phi"][i]**2 if (phi_known is None) else phi_known
            else: 
                sqrt_phi = torch.nn.functional.softplus(data.annotations @ samples["annotation_weights"][i])
                phi = sqrt_phi**2
            psi = samples["sqrt_psi"][i]**2
            v = psi if phi_as_prior else psi*psi

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

def model_uncollapsed(data, sigma_noise = 1., phi_as_prior = True, sqrt_phi = dist.HalfCauchy(1.), desired_min_eig = 1e-6): 
    
    device = data.beta_mrg.device
    
    zero = torch.tensor(0., **data.torch_type)
    one = torch.tensor(1., **data.torch_type)
    
    if not data.annotations is None:
        n_annotations = data.annotations.shape[1] 
        annotation_weights = pyro.sample(
            "annotation_weights",
            dist.Normal(zero, one).expand([n_annotations]).to_event(1) 
        )

        sqrt_phi = torch.nn.functional.softplus(data.annotations @ annotation_weights) # or exp?
        
        sqrt_psi = pyro.sample( # constrain < 1? 
            "sqrt_psi",
             dist.HalfCauchy(sqrt_phi if phi_as_prior else torch.ones(data.p, **data.torch_type)).to_event(1) 
        )
    else: 
        sqrt_phi = convertr(sqrt_phi, "sqrt_phi", device = device)
        sqrt_psi = pyro.sample( # constrain < 1? 
            "sqrt_psi",
             dist.HalfCauchy(sqrt_phi if phi_as_prior else one).expand([data.p]).to_event(1)) # PRSCS uses Strawderman-Berger here
    
    beta_scale = sqrt_psi if phi_as_prior else sqrt_phi*sqrt_psi
    
    sigma_noise = convertr(sigma_noise, "sigma_noise", device = device)
    sigma2_noise = sigma_noise**2
    
    torch_n = torch.tensor(data.n, **data.torch_type)
    sigma_over_sqrt_n = sigma_noise / torch.sqrt(torch_n)
    
    mm = 0
    for kk in range(len(data.blk_size)):
        assert(data.blk_size[kk] > 0)
        
        idx_blk = torch.arange(mm,mm+data.blk_size[kk])

        beta = pyro.sample(
            "beta_%i" % kk, 
            dist.Normal(zero, sigma_over_sqrt_n * beta_scale[idx_blk]).to_event(1) # *sqrt_psi[idx_blk] # already uses scale conveniently
        )
        
        obs = pyro.sample(
            "obs_%i" % kk, 
            dist.MultivariateNormal(
                data.ld_blk[kk] @ beta, 
                scale_tril = sigma_over_sqrt_n * data.ld_blk_chol[kk], 
            ), 
            obs = data.beta_mrg[idx_blk])
        
        mm += data.blk_size[kk]

def model_collapsed(data, sigma_noise = 1., phi_as_prior = True, sqrt_phi = dist.HalfCauchy(1.), desired_min_eig = 1e-6): 
    
    device = data.beta_mrg.device
    
    zero = torch.tensor(0., **data.torch_type)
    one = torch.tensor(1., **data.torch_type)
    
    if not data.annotations is None:
        n_annotations = data.annotations.shape[1] 
        annotation_weights = pyro.sample(
            "annotation_weights",
            dist.Normal(zero, one).expand([n_annotations]).to_event(1) 
        )

        sqrt_phi = torch.nn.functional.softplus(data.annotations @ annotation_weights) # or exp?
        
        sqrt_psi = pyro.sample( # constrain < 1? 
            "sqrt_psi",
             dist.HalfCauchy(sqrt_phi if phi_as_prior else torch.ones(data.p, **data.torch_type)).to_event(1) 
        )
    else: 
        sqrt_phi = convertr(sqrt_phi, "sqrt_phi", device = device)
        sqrt_psi = pyro.sample( # constrain < 1? 
            "sqrt_psi",
             dist.HalfCauchy(sqrt_phi if phi_as_prior else one).expand([data.p]).to_event(1)) # PRSCS uses Strawderman-Berger here
    
    phi = sqrt_phi**2
    psi = sqrt_psi**2
    
    v = psi if phi_as_prior else phi*psi
    
    initial_trace_flag = (v > 1).any().item() # hack. better way to do this? 
            
    sigma_noise = convertr(sigma_noise, "sigma_noise", device = device)
    sigma2_noise = sigma_noise**2
    #sigma2_noise = pyro.sample("sigma_noise", dist.InverseGamma(one * 0.01, one * 0.01))
    #sigma_noise = sigma2_noise.sqrt()
    
    torch_n = torch.tensor(data.n, **data.torch_type)
    sigma_over_sqrt_n = sigma_noise / torch.sqrt(torch_n)
    
    mm = 0
    for kk in range(len(data.ld_blk)):
        assert(data.blk_size[kk] > 0)
        
        idx_blk = torch.arange(mm,mm+data.blk_size[kk])
        
        #first_term = data.ld_blk[kk] @ torch.diag(v[idx_blk]) @ data.ld_blk[kk].T
        cov = (data.ld_blk[kk] * v[idx_blk]) @ data.ld_blk[kk] + data.ld_blk[kk] 
        cov = 0.5 * (cov + cov.T) # isn't _quite_ symmetric otherwise
        
        if initial_trace_flag: 
            chol_cov = torch.eye(data.blk_size[kk], **data.torch_type)
        else: 
            try: # faster than the eigendecomposition if already PSD
                chol_cov = torch.linalg.cholesky(cov)
            except torch._C._LinAlgError: 
                L, V = torch.linalg.eigh(cov)
                cov_min_eig = L.min().item()
                if cov_min_eig < desired_min_eig: 
                    print("Degenerate cov (min eigenvalue=%1.3e)" % cov_min_eig)
                    # smallest addition to diagonal to make min(eig) = min_eig
                    cov += (desired_min_eig - cov_min_eig) * torch.eye(data.blk_size[kk], **data.torch_type)
                chol_cov = torch.linalg.cholesky(cov)
        #if cov.logdet().item() == -np.inf: 
            #print("Degenerate covariance, adding to diagonal")
        #    cov += 1e-2 * torch.eye(data.blk_size[kk])
            
        # equivalent to
        obs = pyro.sample(
            "obs_%i" % kk, 
            dist.MultivariateNormal(
                torch.zeros(data.blk_size[kk], **data.torch_type), # will need to get correct dtype and device here
                scale_tril = chol_cov * sigma_over_sqrt_n
            ), 
            obs = data.beta_mrg[idx_blk])
        
        mm += data.blk_size[kk]

        
def psi_guide(data):
    
    logit_psi_loc = pyro.param("logit_psi_loc", torch.full([data.p],-1., **data.torch_type))
    logit_psi_scale = pyro.param("logit_psi_scale", torch.full_like(logit_psi_loc, 0.1, **data.torch_type), constraint = constraints.positive)
    
    base_dist = dist.Normal(logit_psi_loc, logit_psi_scale).to_event(1)
    
    sqrt_psi = pyro.sample("sqrt_psi", dist.TransformedDistribution(base_dist, SigmoidTransform()))
    
def sigma_guide(data):
    
    logit_sigma_loc = pyro.param("logit_sigma_loc", torch.tensor(0., **data.torch_type))
    logit_sigma_scale = pyro.param("logit_sigma_scale", torch.tensor(0.1, **data.torch_type), constraint = constraints.positive)
    
    base_dist = dist.Normal(logit_sigma_loc, logit_sigma_scale)
    
    sigma_noise = pyro.sample("sigma_noise", dist.TransformedDistribution(base_dist, SigmoidTransform()))
    
def beta_guide(data): 
    
    for which_block in range(len(data.ld_blk)):
        beta_loc = pyro.param("beta_loc_%i" % which_block, torch.zeros([data.blk_size[which_block]], **data.torch_type))
        beta_ld_scale = pyro.param("beta_ld_%i" % which_block, torch.full_like(beta_loc, 10., **data.torch_type), constraint = constraints.positive)
        beta_scale = pyro.param("beta_scale_%i" % which_block, torch.full_like(beta_loc, 0.1, **data.torch_type), constraint = constraints.positive)

        beta = pyro.sample("beta_%i" % which_block, dist.MultivariateNormal(
            beta_loc, 
            precision_matrix = beta_ld_scale[:,None] * data.ld_blk[which_block] * beta_ld_scale + torch.diag(1./beta_scale)))


def my_optimizer(model, guide, data, end = "\r", print_every = 10, min_iterations = 200, max_iterations = 1000, min_particles = 1, max_particles = 16, stall_window = 30, use_renyi = False, lr = 0.03):
    
    adam = pyro.optim.Adam({"lr": lr})

    Loss = RenyiELBO if use_renyi else Trace_ELBO

    # train/fit model
    pyro.clear_param_store()
    
    losses = []
    num_particles = min_particles
       
    while num_particles <= max_particles: 
        svi = SVI(model, guide, adam, loss=Loss(num_particles = num_particles, vectorize_particles = False))
        iteration = 0 
        while True: 
            loss = svi.step(data)
            losses.append(loss)
            iteration += 1
            if iteration % print_every == 0:
                print("[iteration %04d] loss: %.4f" % (iteration + 1, loss), end = end)
            if (num_particles > min_particles or iteration > min_iterations) and iteration > stall_window: 
                R,p = scipy.stats.pearsonr(np.arange(stall_window), losses[-stall_window:])
                if p>0.05 or R>0. or iteration > max_iterations: 
                    num_particles *= 2
                    print("Done after %i iterations. Increasing num_particles to %i." % (iteration + 1, num_particles))
                    break
                    
    return losses

def preprocess_ld(ld_blk, blk_size, desired_min_eig = 1e-8, torch_type = {}):

    ld_fix = []
    ld_chol = []
    for kk,ld in enumerate(ld_blk): 
        ld = torch.tensor(ld, **torch_type)
        L, V = torch.linalg.eigh(ld)
        ld_min_eig = L.min().item()
        if ld_min_eig < desired_min_eig: 
            print("Degenerate LD mat (min eigenvalue=%1.3e), fixing to %1.3e" % (ld_min_eig, desired_min_eig))
            # smallest addition to diagonal to make min(eig) = desired_min_eig
            ld += (desired_min_eig - ld_min_eig) * torch.eye(blk_size[kk], **torch_type)
            #L, V = torch.linalg.eigh(ld)
            #print("New min eigenvalue:%1.3e" % L.min().item())
        
        ld_fix.append(ld)
        ld_chol.append(torch.linalg.cholesky(ld))
        
    return ld_fix, ld_chol

    
def vi(
    sst_dict, 
    n, 
    ld_blk, 
    blk_size, 
    device = "cpu",
    annotations = None,
    sigma_noise = None, 
    phi = None, 
    phi_as_prior = True,
    collapsed = True, 
    beta_guide_type = "DiagonalNormal", # alternative: MultivariateNormal, LDbased
    constrain_psi = True, 
    constrain_sigma = False,
    desired_min_eig = 1e-8, 
    **opt_kwargs
):
    """Variational inference for PRSCS model
    
    Doesn't support different a,b currently (just a=b=0.5 implicitly by using HalfCauchy priors)."""
    
    print('... SVI ...')
    
    torch_type = {"dtype":torch.float, "device": device}

    # derived stats
    #ld_blk = [ (torch.tensor(g, dtype = torch.float) + eps * torch.eye(g.shape[0])) for g in ld_blk ]
    ld_fix, ld_blk_chol = preprocess_ld(ld_blk, blk_size, desired_min_eig = desired_min_eig, torch_type = torch_type)
    
    data = Data(
        beta_mrg = torch.tensor(sst_dict['BETA'], **torch_type),
        p = len(sst_dict['SNP']), # number of SNPs
        n = n, 
        ld_blk = ld_fix, 
        ld_blk_chol = ld_blk_chol,
        blk_size = blk_size,
        annotations = None if (annotations is None) else annotations.to(device), 
        torch_type = torch_type
    )
    
    n_blk = len(ld_blk) # number of LD blocks
    
    #print( Trace_ELBO()(model_uncollapsed, guide)(beta_mrg, n, p, ld_blk, ld_blk_chol, blk_size) )
    one = torch.tensor(1., **torch_type)
    sqrt_phi = dist.HalfCauchy(one) if (phi is None) else np.sqrt(phi).to(**torch_type)
    
    sigma_noise = dist.Gamma(2. * one, 2. * one) if (sigma_noise is None) else sigma_noise # dist.HalfCauchy(one)
    
    model_ = model_collapsed if collapsed else model_uncollapsed
    model = lambda dat: model_(dat, sigma_noise = sigma_noise, phi_as_prior = phi_as_prior, sqrt_phi = sqrt_phi, desired_min_eig = desired_min_eig)    
    
    guide = AutoGuideList(model)
    
    to_expose = []
    if not (annotations is None): to_expose.append("annotation_weights")
    if phi is None: to_expose.append("sqrt_phi")
    
    if len(to_expose) > 0: 
        guide.add(AutoDiagonalNormal(
            poutine.block(
                model,
                expose = to_expose), # or could optimize these? 
            init_loc_fn = init_to_value(values={
                "sqrt_phi" : torch.sqrt(torch.tensor(0.01, **torch_type))
            })))
        
    if (type(sigma_noise) != float):
        guide.add(sigma_guide if constrain_sigma else AutoDiagonalNormal(
            poutine.block(
                model,
                expose = ["sigma_noise"]),
            init_loc_fn = init_to_value(values={"sigma_noise" : one})))    
    
    guide.add(psi_guide if constrain_psi else AutoDiagonalNormal(
        poutine.block(
            model,
            expose = ["sqrt_psi"]),
        init_loc_fn = init_to_value(values={"sqrt_psi" : torch.sqrt(torch.full([p],0.1,**torch_type))})))
    
    if not collapsed: 
        if beta_guide_type == "LDbased": 
            guide.add(beta_guide)
        else: 
            guide_func = AutoMultivariateNormal if (beta_guide_type == "MultivariateNormal") else AutoDiagonalNormal
            for kk in range(n_blk):        
                beta_name = "beta_%i" % kk
                guide.add(guide_func( 
                    poutine.block(model, expose = [beta_name]), 
                    init_loc_fn = init_to_value(values={ beta_name : torch.zeros(blk_size[kk],**torch_type) })))
    
    losses = my_optimizer(model, guide, data, **opt_kwargs)
                    
    stats = get_posterior_stats(model, guide, data, phi, phi_as_prior = phi_as_prior, collapsed = collapsed)
    beta_est = stats["beta"]["mean"].squeeze().cpu().numpy()
    if data.annotations is None: 
        phi_est = stats["sqrt_phi"]["mean"]**2 if (phi is None) else phi
    else: 
        sqrt_phi = data.annotations @ stats["annotation_weights"]["mean"]
        phi_est = sqrt_phi**2

    return losses, beta_est, phi_est, stats
