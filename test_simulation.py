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

import pyro.distributions as dist

import scipy.stats
import seaborn as sns
import pandas as pd
import numpy as np
import string
import random
import time
import datetime
import argparse

def check_sim_result(save_fig_name, anno_path, test = 'chr22', refit_time=1,prop_nz = 0.2,phi_as_prior = True, constrain_sigma = True, lr = 0.03, chrom=22, prscs_file = 'test_data/ibd_wightman_pst_eff_a1_b0.5_phi1e-02_chr22.txt'):
    ## initializing
    chr22_dict =  {
    'ref_dir' : '/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD_PRScs/ldblk_ukbb_eur',
    'bim_prefix' : "test_data/ADSP_qc_chr22",
    'sst_file' : "test_data/wightman_chr22.tsv",
    'n_gwas' : 762971, 
    'out_dir' : "test_data",
    "seed" : 42, 
    "beta_std" : "False", 
    "n_iter" : 1000
    }

    sim_dict = {
        'ref_dir' : '/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD_PRScs/ldblk_ukbb_eur', 
        'bim_prefix' : "test_data/test",
        'sst_file' : "test_data/sumstats.txt",
        'n_gwas' : 200000, 
        'out_dir' : "test_data",
        "seed" : 42, 
        "beta_std" : "False", 
        "n_iter" : 1000
    }
    
    if test == 'chr22':
        param_dict = chr22_dict
        print('running on chr22')
    else:
        param_dict = sim_dict
        print('sim 1k SNP')
    
    ## change the parameters to the right dtypes.
    if type(refit_time == str):
        refit_time = int(refit_time)
        prop_nz = float(prop_nz)
        phi_as_prior = bool(phi_as_prior)
        constrain_sigma = bool(constrain_sigma)
        
    if anno_path=='False':
        anno_path=False
    if anno_path=='None':
        anno_path = None
    
    ## handling the pic saving repo
    random_name = ''.join(random.choices(string.ascii_lowercase +string.digits, k=3))
    date = pd.Timestamp(datetime.date.today()).strftime("%m%d")
    save_dir = '/gpfs/commons/home/tlin/pic/casioPR/simulation/' + date
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        print('creating dir %s'%save_dir)
    path= save_dir + '/' + save_fig_name + '_' + 'iter%d'%refit_time +'_'+random_name +'_'
    print("fig will be saved in %s"%path)
    
    ## start the function
    if '1kg' in os.path.basename(param_dict['ref_dir']):
        ref_df = parse_genet.parse_ref(param_dict['ref_dir'] + '/snpinfo_1kg_hm3')
    elif 'ukbb' in os.path.basename(param_dict['ref_dir']):
        ref_df = parse_genet.parse_ref(param_dict['ref_dir'] + '/snpinfo_ukbb_hm3')

    ref_df = ref_df[ref_df.CHR == chrom]
    vld_df = parse_genet.parse_bim(param_dict['bim_prefix'] + ".bim")
    vld_df = vld_df[vld_df.CHR == chrom]
    sst_dict = parse_genet.parse_sumstats(ref_df, vld_df, param_dict['sst_file'], param_dict['n_gwas'])
    
                                                                                                                             
    ## define if simulating sumstat (beta  
    if param_dict['sst_file'] == "test_data/wightman_chr22.tsv": # real case doesn't need simulated betas  
        prscs_beta = pd.read_csv(prscs_file, sep = '\t', header = None, names = ['CHR','SNP','BP','A1','A2','PRSCS_beta'])
        sst_dict = sst_dict.merge( prscs_beta[['SNP','PRSCS_beta']], on = 'SNP') 
        ld_blk, ld_blk_sym, blk_size = parse_genet.parse_ldblk(param_dict['ref_dir'], sst_dict, chrom)        
        print("There are %s ld_block. \n" %(len(ld_blk)))
        print("using annotations from %s"%anno_path)
        annotations, anno_names =  parse_genet.parse_anno(anno_path, sst_dict, chrom, prop_nz)
        beta_true = sst_dict['PRSCS_beta']
    else: ## need to simulate beta
        ld_blk, ld_blk_sym, blk_size = parse_genet.parse_ldblk(param_dict['ref_dir'], sst_dict, chrom)
        print("There are %s ld_block. \n" %(len(ld_blk)))
        beta_true, beta_mrg, annotations, anno_names = simulate.simulate_sumstats(ld_blk, blk_size, param_dict['n_gwas'], len(sst_dict), sst_dict,anno_path = anno_path, chrom=chrom,prop_nz = prop_nz)
        sst_dict["BETA"] = beta_mrg
    
    
    '''
    this might be wrong!!! not really passing through the prop_nz
    '''
    # if param_dict['anno_path'] == False:  ## This will be sim
    #     ld_blk, ld_blk_sym, blk_size = parse_genet.parse_ldblk(param_dict['ref_dir'], sst_dict, chrom)
    #     print("There are %s ld_block. \n" %(len(ld_blk)))
    #     beta_true, beta_mrg, annotations, anno_names = simulate.simulate_sumstats(ld_blk, blk_size, param_dict['n_gwas'], len(sst_dict), sst_dict,anno_path = param_dict['anno_path'], chrom=chrom,prop_nz = prop_nz)
    #     sst_dict["BETA"] = beta_mrg
    # else: ## real case doesn't need simulated betas   
    #     prscs_beta = pd.read_csv(prscs_file, sep = '\t', header = None, names = ['CHR','SNP','BP','A1','A2','PRSCS_beta'])
    #     sst_dict = sst_dict.merge( prscs_beta[['SNP','PRSCS_beta']], on = 'SNP') 
    #     ld_blk, ld_blk_sym, blk_size = parse_genet.parse_ldblk(param_dict['ref_dir'], sst_dict, chrom)        
    #     print("There are %s ld_block. \n" %(len(ld_blk)))
    #     annotations, anno_names =  parse_genet.parse_anno(param_dict["anno_path"], sst_dict, chrom = chrom,prop_nz = prop_nz)
    #     beta_true = sst_dict['PRSCS_beta']
    if anno_path != None:
        anno_names.insert(0,'intercept')
        

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("start VI...")
    #one = torch.tensor(1., device=device)
    
    pearsonr_list=[]
    anno_list=pd.DataFrame()

    for i in range(refit_time):
        print('Re-train the model %d time(s)'% (i+1))
        losses, beta, phi_est, stats = vi.vi(sst_dict, param_dict['n_gwas'], ld_blk, blk_size, device = device, annotations = annotations, max_iterations = param_dict['n_iter'], max_particles=4, desired_min_eig = 1e-3, min_iterations = 200, stall_window = 30, phi_as_prior = phi_as_prior, lr = 0.03, constrain_sigma = constrain_sigma)
        print('phi_prior %s'%phi_as_prior)
        print('constrain_sigma %s'%constrain_sigma)
        plt.plot(losses);plt.title('losses');#plt.savefig(save_dir + '/loss/'+ param_dict['save_fig_name'] + '_' + 'iter%d'%refit_time +'_'+random_name +'_' +'loss_%s.png'%i);
        
        anno_df = pd.DataFrame.from_dict(stats["annotation_weights"] )
        anno_list = anno_list.append(anno_df["mean"].to_frame().T, ignore_index=True) 
        if i == refit_time-1:
            plt.savefig(path+'loss.pdf',format ='pdf',bbox_inches='tight'); plt.show()
            
        r=scipy.stats.pearsonr(beta_true,beta)[0]   
        pearsonr_list.append(r)

    ## check anno_weight (only when annotation exist)
    if anno_path != None:
        anno_list.to_csv(path+'_annoweight.csv')
        plt.figure()
        plt.bar(anno_list.mean().index, anno_list.mean(), yerr=anno_list.std(), capsize=3, color='turquoise')
        plt.xticks(anno_list.mean().index, anno_names, rotation=90)
        plt.ylabel('weight')
        plt.title('iterate %d times'%(refit_time))
        plt.savefig(path+'anno_weight.pdf',format ='pdf',bbox_inches='tight');plt.show()
        
        print('anno_weight')
        print(anno_list)
        
        ## only_for_perfect_anno
        if anno_path == False:        
            plt.figure()
            correlation = anno_list.iloc[:][1].corr(anno_list.iloc[:][2])
            sns.regplot(anno_list.iloc[:][1],anno_list.iloc[:][2], ci=None,marker="p", color="b", line_kws=dict(color="r", alpha = 0.5))
            plt.text(anno_list.iloc[:][1].mean(), anno_list.iloc[:][2].mean(), f'Correlation: {correlation:.2f}', fontsize=12, color='blue', va='top',ha='center')
            # plt.plot(anno_list.iloc[:][1],anno_list.iloc[:][2], marker='X', linestyle='None', color='y')
            plt.xlabel('perfect anno')
            plt.ylabel('random anno')
            plt.title('annotation weights')
            plt.savefig(path+'anno_weight_scatter.pdf',format ='pdf');plt.show()   
           
    ## plot pearson r between the beta of PRSCS and CasioPR
    if (refit_time >20):
        plt.figure(num=None, figsize=(17, 8))
    else:
        plt.figure()
    plt.plot(range(refit_time), pearsonr_list, marker='o', linestyle='--', color='c')
    ax = plt.axes()
    ax.set_ylim(0, 1)
    plt.xticks(list(range(refit_time)),list(range(1,refit_time+1)))
    plt.title('pearson r for betas in every iteration')
    plt.savefig(path+'betas.pdf',format ='pdf');plt.show()  
    
    f = open(path+'person_r.txt', "w")
    f.write(str(pearsonr_list))
    f.close()
 
    print('\n')
    print('person R list')
    print(pearsonr_list)
    
    return(anno_list,pearsonr_list)        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation")
    parser.add_argument("save_fig_name", type=str, help="Save figure name")
    parser.add_argument("anno_path", type=str, help="Annotation path")
    parser.add_argument("test_on", type=str, help="chr22 or sim")
    parser.add_argument("prop_nz", type=str, help="Proportion of non-zero values (default: 0.2)")
    parser.add_argument("phi_as_prior", type=bool, help="Phi as prior (default: True)")
    parser.add_argument("constrain_sigma", type=bool, help="Constrain sigma (default: True)")
    parser.add_argument("--refit_time", type=int, default=10, help="Refit time (default: 20)")
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate (default: 0.03)")
    parser.add_argument("--chrom", type=int, default=22, help="Chromosome (default: 22)")
    args = parser.parse_args()
    
    for arg_name, arg_value in vars(args).items():
        print(f"{arg_name}: {arg_value}")
    
    print("start testing params")
    check_sim_result(args.save_fig_name, args.anno_path, args.test_on, refit_time = args.refit_time, prop_nz = args.prop_nz, phi_as_prior = args.phi_as_prior,  constrain_sigma = args.constrain_sigma, lr = args.lr, chrom = args.chrom)
    
    

    