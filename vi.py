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
import time
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
    num_samples=100
): 
    
    """ extract posterior samples (somewhat weirdly this is done with `Predictive`) """
    ## Predicitve is a class
    #guide.requires_grad_(False)
    predictive = Predictive(
        model,             
        guide=guide, 
        num_samples=num_samples) 
    
    samples = predictive(data) ## get posterior samples
    samples["beta"] = torch.zeros(num_samples,data.p,**data.torch_type)
    
    ## phi is the global scaling, psi is the local one.
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


def model_collapsed(data, path, sigma_noise = 1., phi_as_prior = False, sqrt_phi = dist.HalfCauchy(1.), desired_min_eig = 1e-6, gaussian_anno_weight = True): 
    
    device = data.beta_mrg.device
    
    zero = torch.tensor(0., **data.torch_type)
    one = torch.tensor(1., **data.torch_type)
    
    ## handling annotations
    if not data.annotations is None:
        n_annotations = data.annotations.shape[1] 
        
        ## original one
        if gaussian_anno_weight:
            annotation_weights = pyro.sample(
                "annotation_weights",
                dist.Normal(zero, one).expand([n_annotations]).to_event(1) 
            )
        else:  
            prior = torch.ones(n_annotations) / (n_annotations + 1)
            n_zero = n_annotations//3
            random_positions = torch.randperm(n_annotations)[:n_zero]
            prior[random_positions] = 1e-10
            annotation_weights = pyro.sample("annotation_weights", dist.Dirichlet(prior))
            
       
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
    for kk in range(len(data.ld_blk)): ## for every LD block
        assert(data.blk_size[kk] > 0)
        
        idx_blk = torch.arange(mm,mm+data.blk_size[kk])
        
        #first_term = data.ld_blk[kk] @ torch.diag(v[idx_blk]) @ data.ld_blk[kk].T
        #cov = (data.ld_blk[kk].double() * v[idx_blk]) @ data.ld_blk[kk].double() + data.ld_blk[kk] 
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


def my_optimizer(model, guide, data, end = "\r", print_every = 10, min_iterations = 200, max_iterations = 1000, min_particles = 1, max_particles = 8, stall_window = 30, use_renyi = False, lr = 0.03):
    """
    Perform stochastic optimization

    This function conducts stochastic optimization using the Pyro library's SVI (Stochastic Variational Inference).
    It iterates over different numbers of particles, gradually increasing the number until convergence criteria are met.

    Args:
        model (callable): The generative model.
        guide (callable): The guide (approximate posterior).
        data: Data required by the model and guide.
        end (str, optional): End character for printing (default is "\\r").
        print_every (int, optional): Print loss every 'print_every' iterations (default is 10).
        min_iterations (int, optional): Minimum number of iterations before particle count can be increased (default is 200).
        max_iterations (int, optional): Maximum number of iterations before stopping (default is 1000).
        min_particles (int, optional): Minimum number of particles (default is 1).
        max_particles (int, optional): Maximum number of particles (default is 16).
        stall_window (int, optional): Number of iterations to check for stall (default is 30).
        use_renyi (bool, optional): Use RenyiELBO if True, else use Trace_ELBO (default is False).
        lr (float, optional): Learning rate for the optimization (default is 0.03).

    Returns:
        list: List of losses for each iteration.
    """
    
    adam = pyro.optim.Adam({"lr": lr})

    Loss = RenyiELBO if use_renyi else Trace_ELBO

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
    sst_df, 
    n, 
    ld_blk, 
    blk_size, 
    device = "cpu",
    annotations = None,
    sigma_noise = None, 
    phi = None, 
    phi_as_prior = False,
    constrain_psi = True, 
    constrain_sigma = False,
    desired_min_eig = 1e-3, 
    gaussian_anno_weight = True, 
    path = '/gpfs/commons/home/tlin/pic/casioPR/simulation',
    **opt_kwargs
):
    """Variational inference for PRSCS model
    
    Doesn't support different a,b currently (just a=b=0.5 implicitly by using HalfCauchy priors).
    
    Args:
        sst_df (dict): A pandas dataframe of summary statistics, including 'BETA' and 'SNP'.
        n (int): Number of individuals in the underlying GWAS.
        ld_blk: List of LD (Linkage Disequilibrium) blocks.
        blk_size (list of int): Size of LD blocks.
        device (str, optional): torch Device to perform computations on (default is 'cpu').
        annotations (torch.Tensor, optional): Annotation data: a SNPs x annotations torch.tensor. One column should be all 1s (intercept) (default is None). Note: Annotations need to be float tensor!
        sigma_noise (optional): If None, sigma_noise will be inferred. If a float, will be fixed to that value (default is None).
        phi (float, optional): If None, phi will be inferred. If a float, will be fixed to that value (default is None).
        phi_as_prior (bool, optional): Whether to represent sqrt_psi ~ HalfCauchy(sqrt_phi) rather than having prior_variance \propto phi * psi. These models are equivalent in principle but inference works quite differently (default changed from true to false).
        constrain_psi (bool, optional): Constrain psi parameter to [0,1] (default is True).
        constrain_sigma (bool, optional): Constrain sigma_noise parameter to [0,1] (default is False).
        desired_min_eig (float, optional): Desired minimum eigenvalue of LD matrices and covariances (default is 1e-3).
        gaussian_anno_weight (bool, optional) : The distribution of the annotations. If false, weights will follow dirichlet(defult is Normal distribution).
        path(string, optional): The file name + absolute path to saved the simulated weights for the annotations. 
        **opt_kwargs: Additional keyword arguments for optimization (see `my_optimizer`).

    Returns:
        tuple: A tuple containing the following elements:
            - losses (list): List of optimization losses (negative ELBOs)
            - beta_est (numpy.ndarray): Estimated beta coefficients.
            - phi_est (float): Estimated phi hyperparameter.
            - stats (dict): Dictionary containing posterior statistics.
    """
    
    print('... Starting SVI ...')
    t0 = time.time()
    torch_type = {"dtype":torch.float, "device": device}

    # derived stats
    #ld_blk = [ (torch.tensor(g, dtype = torch.float) + eps * torch.eye(g.shape[0])) for g in ld_blk ]
    ld_fix, ld_blk_chol = preprocess_ld(ld_blk, blk_size, desired_min_eig = desired_min_eig, torch_type = torch_type)
    
    data = Data(
        beta_mrg = torch.tensor(sst_df['BETA'].values, **torch_type),
        p = len(sst_df['SNP']), # number of SNPs
        n = n, 
        ld_blk = ld_fix, 
        ld_blk_chol = ld_blk_chol,
        blk_size = blk_size,
        annotations = None if (annotations is None) else annotations.to(device), 
        torch_type = torch_type
    )
    
    n_blk = len(ld_blk) # number of LD blocks
    
    one = torch.tensor(1., **torch_type)
    sqrt_phi = dist.HalfCauchy(one) if (phi is None) else np.sqrt(phi).to(**torch_type)
    sigma_noise = dist.HalfCauchy(one) if (sigma_noise is None) else sigma_noise # dist.Gamma(2. * one, 2. * one), dist.InverseGamma(0.001 * one, 0.001 * one)
    model = lambda dat: model_collapsed(dat, sigma_noise = sigma_noise, phi_as_prior = phi_as_prior, sqrt_phi = sqrt_phi, desired_min_eig = desired_min_eig, gaussian_anno_weight = gaussian_anno_weight, path = path)       
    
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
    
    losses = my_optimizer(model, guide, data, **opt_kwargs)
    stats = get_posterior_stats(model, guide, data, phi, phi_as_prior = phi_as_prior)
    beta_est = stats["beta"]["mean"].squeeze().cpu().numpy()
    if data.annotations is None: 
        phi_est = stats["sqrt_phi"]["mean"]**2 if (phi is None) else phi
    else: 
        sqrt_phi = data.annotations @ stats["annotation_weights"]["mean"]
        phi_est = sqrt_phi**2
        
    print('Done in %0.2f seconds \n'%(time.time() - t0))
    return losses, beta_est, phi_est, stats
