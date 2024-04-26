'''
Usage:
python test_simulation.py --save_fig_name --anno_path --test_on (defult sim) --refit_time (default: 10) --chrom (defult 22) --gaussian_anno_weight (default True) --noise_size (default 0.1)

- anno_path: If set to False, will use the perfect annotation. If set to None, won't use any annotation. If you have annotations, just put the path of the annotation.

- test_on: either sim (simulate 1k SNP) or chr (real data)

'''

import os
import sys
import getopt

import parse_genet
import vi

import importlib
import simulate
import torch
import matplotlib.pyplot as plt
import numpy as np
import mcmc_gtb
import pyro.distributions as dist

import scipy.stats
import seaborn as sns
import pandas as pd
import numpy as np
import string
import random
import time
import datetime
from scipy.stats import mannwhitneyu
import argparse
from tqdm import tqdm

## create the file name
def get_name(save_fig_name, refit_time):
    random_name = ''.join(random.choices(string.ascii_lowercase +string.digits, k=3))
    date = pd.Timestamp(datetime.date.today()).strftime("%m%d")
    save_dir = '/gpfs/commons/home/tlin/pic/casioPR/simulation/' + date
    os.makedirs(save_dir, exist_ok=True)
    path= save_dir + '/' + save_fig_name + '_' + 'iter%d'%refit_time +'_'+random_name +'_'
    return(path)

## get the relationship between true beta and the estimate beta
def get_beta_stats(betas):
    pearson_r = betas.iloc[:, 2:].apply(lambda column: column.corr(betas.beta_true))
    mse_values = betas.iloc[:, 2:].apply(lambda column: np.mean((column - betas.beta_true)**2))
    utest= betas.iloc[:, 2:].apply(lambda column: mannwhitneyu(column,betas.beta_true, alternative='two-sided'))
    beta_stats = pd.DataFrame({'pearsonR': pearson_r, 'MSE': mse_values,'Mannwhitney': utest[1:].values.flatten()})
    return(beta_stats)

def plot_pearsonr(beta_stats, include_prscs, refit_time, path):
    if (refit_time >20):
        plt.figure(num=None, figsize=(17, 8))
    else:
        plt.figure()
        
    pearsonr_list = beta_stats['pearsonR']
    if (include_prscs):
        prscs_r = pearsonr_list[len(pearsonr_list)-1]
        pearsonr_list = pearsonr_list[:-1]
        plt.axhline(y = prscs_r, color = 'steelblue', linestyle = '-', alpha = 0.7,label = 'PRSCS(MCMC)') 
    
    plt.plot(range(refit_time), pearsonr_list, marker='o', linestyle='-.', color='lightcoral', label = 'CasioPR(VI)')
    ax = plt.axes()
    ax.set_ylim(0, 1)
    plt.xticks(list(range(refit_time)),list(range(1,refit_time+1)))
    plt.xlabel('Iteration')
    plt.title(r'pearson r for est $\beta$ with marginal $\beta$ in every re-fitted model')
    plt.legend(bbox_to_anchor = (0.82, 0.2), loc = 'center') 
    plt.savefig(path+'betas.pdf',format ='pdf',bbox_inches='tight')
    plt.show()  

def load_data(chrom,ref_df,  path, anno_path, prop_nz, noise_size, bim_prefix, sst_file_name , n_gwas, ref_dir):
    ref_df = ref_df[ref_df.CHR == chrom]
    # vld_df = parse_genet.parse_bim(bim_prefix + ".bim")
    # vld_df = vld_df[vld_df.CHR == chrom]
    vld_df = parse_genet.parse_bim(bim_prefix + "%s.bim"%chrom) if sst_file_name != "test_data/sumstats.txt" else parse_genet.parse_bim(bim_prefix + ".bim")
    sst_dict = parse_genet.parse_sumstats(ref_df, vld_df, sst_file_name, n_gwas)
#    sst_dict = parse_genet.parse_sumstats_no_vld(ref_df, sst_file_name, n_gwas)
    
    ld_blk, ld_blk_sym, blk_size = parse_genet.parse_ldblk(ref_dir, sst_dict, chrom)
    print("There are %s ld_block. in chr%s\n" %(len(ld_blk),chrom))
    beta_true, beta_mrg, annotations, anno_names = simulate.simulate_sumstats(ld_blk, blk_size, n_gwas, len(sst_dict), sst_dict, path, anno_path = anno_path, chrom=chrom,prop_nz = prop_nz, noise_size = noise_size)
    print('finish load')
    return ref_df, vld_df, sst_dict, ld_blk, ld_blk_sym, blk_size, beta_true, beta_mrg, annotations, anno_names

def check_sim_result(save_fig_name, anno_path, beta_prior_a = 0, use_sim_dict = False, gaussian_anno_weight = True, noise_size = 0, refit_time = 10,prop_nz = 0.2, phi_as_prior = False, constrain_sigma = True, lr = 0.03, chrom=22, run_prscs = True):
    ## initializing
    use_sim_dict = bool(use_sim_dict)
    #17k 'bim_prefix' : '/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/ADSP_vcf/17K_final/annotated_filtered_hg37/plink/vcf_filt/ADSP_annotated_chr',
    ## prscs_ref:/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD_PRScs/ldblk_ukbb_eur
    chr_dict = { 
    'bim_prefix' : '/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/ADSP_vcf/36K_QC/annotated_hg37_plink_ibd/qc/qc_chr',
    'sst_file' : 'test_data/wightman4prscs.tsv',
    'n_gwas' : 762971
    }
    
    sim_dict = {
        'bim_prefix' : "test_data/test",
        'sst_file' : "test_data/sumstats.txt",
        'n_gwas' : 200000
    }
    
    if use_sim_dict :
        param_dict = sim_dict
        print('simulate 1k SNP')
        
    else:
        param_dict = chr_dict
    param_dict['ref_dir']='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD_ADSP36K_4PRScs/snps_only/ldblk_adsp_chr/' ## ADSP ld
    param_dict['n_iter'] = 1000
    
    ## change the parameters to the right dtypes
    if type(refit_time == str):
        refit_time = int(refit_time)
        # prop_nz = float(prop_nz)
        # phi_as_prior = bool(phi_as_prior)
        # constrain_sigma = bool(constrain_sigma)
        
    if (beta_prior_a == 'None'):
        beta_prior_a = None
    elif (beta_prior_a == 'inf'):
        beta_prior_a = np.inf
    else:
        beta_prior_a = float(beta_prior_a)
    
    anno_path = False if anno_path == 'False' else None if anno_path == 'None' else anno_path
        
    ## handling the pic saving repo
    path = get_name(save_fig_name, refit_time)
    print("Figs will be saved in %s "%path)
    
    ## start the function
    if '1kg' in os.path.basename(param_dict['ref_dir']):
        ref_df = parse_genet.parse_ref(param_dict['ref_dir'] + '/snpinfo_1kg_hm3')
    elif 'ukbb' in os.path.basename(param_dict['ref_dir']):
        ref_df = parse_genet.parse_ref(param_dict['ref_dir'] + '/snpinfo_ukbb_hm3')
    elif 'ldblk_adsp_chr' in param_dict['ref_dir']:
        ref_df = parse_genet.parse_ref(param_dict['ref_dir'] + '/v2snpinfo_adsp_hm3')

    if (chrom == 22):
        print('running on chr%s'%chrom)
        ref_df, vld_df, sst_dict, ld_blk, ld_blk_sym, blk_size, beta_true, beta_mrg, annotations, anno_names = load_data(chrom,ref_df, path,anno_path, prop_nz, noise_size, param_dict['bim_prefix'], param_dict['sst_file'],param_dict['n_gwas'],param_dict['ref_dir'] )
      
    else:
        ld_blk, ld_blk_sym, blk_size = [],[],[]
        annotations, beta_true, beta_mrg = torch.tensor([]), torch.tensor([]), torch.tensor([])
        sst_dict = pd.DataFrame()
        
        for i in tqdm(range(chrom ,23)):
            print('running on chr%s'%i)
            ref_df_chr, vld_df_chr, sst_dict_chr, ld_blk_chr, ld_blk_sym_chr, blk_size_chr, beta_true_chr, beta_mrg_chr, annotations_chr, anno_names = load_data(i, ref_df, path, anno_path, prop_nz, noise_size, param_dict['bim_prefix'], param_dict['sst_file'],param_dict['n_gwas'],param_dict['ref_dir'])
 
            ld_blk = ld_blk + ld_blk_chr
            ld_blk_sym = ld_blk_sym + ld_blk_sym_chr
            blk_size = blk_size+ blk_size_chr
            annotations = torch.cat((annotations,annotations_chr))
            beta_true = torch.cat((beta_true, beta_true_chr))
            beta_mrg = torch.cat((beta_mrg, beta_mrg_chr))
            sst_dict = pd.concat([sst_dict, sst_dict_chr])

    sst_dict["BETA"] = beta_mrg
    #sst_dict.to_csv('whole_genome_wightman.tsv', sep = '\t', index = False)
    
    if anno_path != None:
        anno_names.insert(0,'intercept')    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("... start VI...")
    plt.figure()
    anno_list=pd.DataFrame()
    betas = pd.DataFrame({'beta_true':beta_true, 'beta_marginal':beta_mrg})
    betas.to_csv(path+'betas.tsv', sep = '\t', index = False)
    print(betas)
    for i in tqdm(range(refit_time)):
        print('Re-train the model %d time(s)'% (i+1))
        losses, beta, phi_est, stats =  vi.vi(sst_dict, param_dict['n_gwas'], ld_blk, blk_size, device = device, annotations = annotations, max_iterations = param_dict['n_iter'], beta_prior_a = beta_prior_a, max_particles=4, desired_min_eig = 1e-3, min_iterations = 200, stall_window = 30, phi_as_prior = phi_as_prior, lr = lr, constrain_sigma = constrain_sigma, gaussian_anno_weight = gaussian_anno_weight, path = path)
        column_name = f'beta_casioPR_{i + 1}'
        betas[column_name] = beta
        plt.plot(losses);plt.title('losses')
        
        if anno_path != None:
            anno_df = pd.DataFrame.from_dict(stats["annotation_weights"])
            anno_list = anno_list.append(anno_df["mean"].to_frame().T, ignore_index=True) 
        if i == refit_time-1:
            plt.savefig(path+'loss.pdf',format ='pdf',bbox_inches='tight'); plt.show()
    
    if (run_prscs):
        beta_prscs = mcmc_gtb.mcmc(1, 0.5, None, sst_dict, param_dict['n_gwas'], ld_blk, blk_size, param_dict['n_iter'], 500, 5, int(chrom), path, False, 42)
        betas['beta_prscs'] = beta_prscs.flatten()
    
    ##  save beta
    print('saving beta')
    betas.to_csv(path+'betas.tsv', sep = '\t', index = False)

    
    # get pearson, MSE, and mannwhitney U test 
    beta_stats = get_beta_stats(betas)
    beta_stats.to_csv(path+'betas_stat.tsv', sep = '\t', index = False)
    
    #  plot pearson r between the marginal beta and betea of PRSCS/CasioPR 
    plot_pearsonr(beta_stats, run_prscs, refit_time, path)
         

    ##  check anno_weight (only when annotation exist)
    if anno_path != None:
        anno_list.to_csv(path+'anno_weight.csv')
        plt.figure()
        plt.bar(anno_list.mean().index, anno_list.mean(), yerr=anno_list.std(), capsize=3, color='turquoise')
        plt.xticks(anno_list.mean().index, anno_names, rotation=90)
        plt.axhline(y = 0, linestyle = '--', color = 'darkgrey') 
        plt.ylabel('weight'); plt.title('iterate %d times'%(refit_time))
        plt.savefig(path+'anno_weight.pdf',format ='pdf',bbox_inches='tight');plt.show()
        print('anno_weight')
        print(anno_list)
        
        ## only_for_perfect_anno
            # plt.figure()
            # correlation = anno_list.iloc[:][1].corr(anno_list.iloc[:][2])
            # sns.regplot(anno_list.iloc[:][1],anno_list.iloc[:][2], ci=None,marker="p", color="b", line_kws=dict(color="r", alpha = 0.5))
            # plt.text(anno_list.iloc[:][1].mean(), anno_list.iloc[:][2].mean(), f'Correlation: {correlation:.2f}', fontsize=12, color='blue', va='top',ha='center')
            # # plt.plot(anno_list.iloc[:][1],anno_list.iloc[:][2], marker='X', linestyle='None', color='y')
            # plt.xlabel('perfect anno')
            # plt.ylabel('random anno')
            # plt.title('annotation weights')
            # plt.savefig(path+'anno_weight_scatter.pdf',format ='pdf');plt.show()   
    return(anno_list,betas)   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation")
    parser.add_argument("--save_fig_name", type=str, default = 'test', help="Save figure name")
    parser.add_argument("--anno_path", type=str, default = None, help="Annotation path")
    parser.add_argument('--beta_prior_a', type = str, default = None, help = 'half-cauchy')
    parser.add_argument("--gaussian_anno_weight", type=bool, default = False, help="gaussian or dirichlet anno weights")
    parser.add_argument("--noise_size", type=float, default = 0.1)
    parser.add_argument("--refit_time", type=int, default=10, help="Refit time (default: 10)")
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate (default: 0.03)")
    parser.add_argument("--chrom_start", type=int, default=21, help="Chromosome (default: 21)")
    parser.add_argument("--use_sim_dict", type=bool, default=False, help="load dict from real data or the simulated one from PRSCS(default False)")
    #parser.add_argument("--use_sim", action='store_true',help="load dict from real data or the simulated one from PRSCS(default False)")
    args = parser.parse_args()
    print(' ')
    print(' ')
    print('Parameters used:')
    for arg_name, arg_value in vars(args).items():
        print(f"{arg_name}: {arg_value}")
    
    print(' ')
    print(' ')
    print(' ')
   
    print('====== Start Running CasioPR ====== \n')
    check_sim_result(args.save_fig_name, args.anno_path,beta_prior_a = args.beta_prior_a,  gaussian_anno_weight = args.gaussian_anno_weight, noise_size = args.noise_size, refit_time = args.refit_time, lr = args.lr, chrom = args.chrom_start, use_sim_dict = args.use_sim_dict)
    

    
'''
Note:
anno_path: If set to False, will use the perfect annotation. If set to None, won't use any annotation. If you have annotations, just put the patn of the annotation.

'''