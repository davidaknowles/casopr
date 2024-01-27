import numpy as np
import pyro.distributions as dist
import torch
import parse_genet
import matplotlib.pyplot as plt


def plot_perfect_anno_noise(noise, perfect, noise_size, path):
    fig, ax = plt.subplots(1,2, figsize=(8, 8))
    
    true_count = torch.sum(perfect).item()
    labels = ['True', 'False']
    counts = [true_count, len(perfect) - true_count]
    
    ax[0].bar(labels, counts, color=['dodgerblue', 'tomato'])
    ax[0].set_ylabel('Count')
    for i, count in enumerate(counts):
        ax[0].text(i, count , str(count), ha='center', va='bottom')
        
    box1 = ax[1].boxplot([noise[perfect]], positions=[0],  widths=0.3, patch_artist=True, notch=True)
    box2 = ax[1].boxplot([noise[~perfect]], positions=[0.4], widths=0.3, patch_artist=True,notch=True)
    ax[1].set_xticklabels(["True","False"])
    colors = ['dodgerblue', 'tomato']
    for boxplot, color in zip([box1, box2], colors):
        for box in boxplot['boxes']:
            box.set(facecolor=color)
    
    fig.suptitle('noise = $\pm$ %s' % noise_size, size = 15)
    fig.savefig(path + 'perfect_anno_noise.pdf' ,format ='pdf',bbox_inches='tight')
    plt.show()
    
    
## add anno
def simulate_sumstats(ld_blk, blk_size, n_gwas, n_variant, sst_dict, path, anno_path=None, prop_nz = 0.2, beta_sd = 0.1, sigma_noise = 1., chrom=22, use_sumstat_beta = False,  noise_size = 0.1): 
    
    sigma_over_sqrt_n = sigma_noise / torch.sqrt(torch.tensor(n_gwas))
    #print('prop_nz = %f'%prop_nz)
    nz = torch.rand(n_variant) < prop_nz ## filter the snp with p threshold < prop_nz ## creating perfect annotation
    ## nz: the perfect annotaion (1 for causal, 0 for not); ## should add some noise here too
    
    ## sim beta for every SNP
    if not use_sumstat_beta:
        print('... simulating betas...')
        beta_true = torch.where(nz, beta_sd * torch.randn(n_variant), torch.zeros(n_variant)) ## torch.randn = random normal distribution
    else:
        print('... using real betas from sumstats ...')
        beta_true = sst_dict['BETA']

    ### reading annotations
    if anno_path == False : ## perfect anno
        print('... simulating perfect anno...')
     
        print('... add noise between +- %s for the perfect anno ...'% noise_size)
        noise = (2 * torch.rand(n_variant)- 1 )* noise_size
        perfect = nz
        nz = nz + noise
        #plot_perfect_anno_noise(nz,perfect, noise_size, path)

        annotations = torch.stack([torch.ones(n_variant),nz,torch.randn(n_variant)]).T # intercept, useful annotation, random annotation
        anno_names = ["perfect anno",'random anno']
       
    else: ## either use anno provided in anno_path (can be single, multiple, or none)
        annotations, anno_names = parse_genet.parse_anno(anno_path, sst_dict, chrom = chrom)
    

    
    beta_mrg = torch.zeros(n_variant)
    mm = 0
    for kk in range(len(ld_blk)):
        idx_blk = torch.arange(mm,mm+blk_size[kk])
        ld_torch = torch.tensor(ld_blk[kk], dtype = torch.float)
        L, V = torch.linalg.eigh(ld_torch)
        L[L < 0.] = 0.

        beta_mrg[idx_blk] = ld_torch @ beta_true[idx_blk]  + sigma_over_sqrt_n * (V @ torch.diag(L.sqrt())) @ torch.randn(blk_size[kk])
        #ld_torch @ beta_true[idx_blk], 
        # covariance_matrix = ld_torch * sigma_over_sqrt_n**2).rsample()
        mm += blk_size[kk]
    #annotations_double = annotations.double()
    
    
    return beta_true, beta_mrg, annotations, anno_names

# def simulate_perfect_anno(sst, prop_nz = 0.2):
#     nz = torch.tensor(sst['P'] < prop_nz)
#     n = sst.shape[0]
#     annotations = torch.stack([torch.ones(n),nz,torch.randn(n)]).T
#     anno_names = ["perfect anno",'random anno']
#     return annotations, anno_names
    
    
def simulate_sumstats_easy(ld_blk, blk_size, n_gwas, p, prop_nz = 0.2, beta_sd = 0.1, sigma_noise = 1. ): 
    sigma_over_sqrt_n = sigma_noise / torch.sqrt(torch.tensor(n_gwas))
    nz = torch.rand(p) < prop_nz ## filter the snp with p threshold < prop_nz ## creating perfect annotation
    ## nz: the perfect annotaion (1 for causal, 0 for not); ## should add some noise here too
    beta_true = torch.where(nz, beta_sd * torch.randn(p), torch.zeros(p)) ## torch.randn = random normal distribution
    annotations = torch.stack([torch.ones(p),nz,torch.randn(p)]).T # intercept, useful annotation, random annotation

    beta_mrg = torch.zeros(p)
    mm = 0
    for kk in range(len(ld_blk)): ## training for each LD block
        idx_blk = torch.arange(mm,mm+blk_size[kk])
        ld_torch = torch.tensor(ld_blk[kk], dtype = torch.float)
        L, V = torch.linalg.eigh(ld_torch)
        L[L < 0.] = 0.

        beta_mrg[idx_blk] = ld_torch @ beta_true[idx_blk] + sigma_over_sqrt_n * (V @ torch.diag(L.sqrt())) @ torch.randn(blk_size[kk])
        #ld_torch @ beta_true[idx_blk], 
        # covariance_matrix = ld_torch * sigma_over_sqrt_n**2).rsample()
        mm += blk_size[kk]

    return beta_true, beta_mrg, annotations



def stratified_LDSC(annotations, beta_mrg, ld_blk, blk_size):
    [p,k] = annotations.shape
    ldscore = torch.zeros([p,k])
    
    mm = 0
    for kk in range(len(ld_blk)):
        idx_blk = torch.arange(mm,mm+blk_size[kk])
        ld_torch = torch.tensor(ld_blk[kk], dtype = torch.float)
        ldscore[idx_blk,:] = ld_torch**2 @ annotations[idx_blk,:]
        mm += blk_size[kk]
    # min_iterations = 100, max_iterations = 1000, min_particles = 1, max_particles = 32, stall_window = 10, use_renyi = False, lr = 0.03

    chi2 = beta_mrg**2 
    tau = torch.linalg.solve(ldscore.T @ ldscore, ldscore.T @ chi2)

    return tau


