#!/usr/bin/env python

"""
CasioPR: Continuous Annotation-aware ShrInkage Of Polygenic Risk score.

Usage:
(for simulation only)
python casioPR.py --save_fig_name --anno_path --test_on --refit_time

python casioPR.py --ref_dir=PATH_TO_REFERENCE --bim_prefix=VALIDATION_BIM_PREFIX --sst_file=SUM_STATS_FILE --n_gwas=GWAS_SAMPLE_SIZE 

python casioPR.py --ref_dir=PATH_TO_REFERENCE --bim_prefix=VALIDATION_BIM_PREFIX --sst_file=SUM_STATS_FILE --n_gwas=GWAS_SAMPLE_SIZE --out_dir=OUTPUT_DIR
                [--a=PARAM_A --b=PARAM_B --phi=PARAM_PHI --n_iter=MCMC_ITERATIONS ]
                
"""

import os
import sys
import getopt

import argparse

    
    
def sim_result_dict():
    param_dict = parse_param()      
    
    ## initializing
    if param_dict['test_on'] == 'sim':
        print('simulate 1k SNP')
        param_dict['bim_prefix'] = "test_data/test"
        param_dict['sst_file'] = "test_data/sumstats.txt"
        
    else:
        print('running on chr22')
        param_dict['bim_prefix'] = "test_data/ADSP_qc_chr22"
        param_dict['sst_file'] = "test_data/wightman_chr22.tsv"
        param_dict['n_gwas'] =762971
    
    if param_dict['anno_path']=='False':
        param_dict['anno_path']= False
    if param_dict['anno_path']:
        param_dict['anno_path'] = None
    
    for key in param_dict:
        print('--%s=%s' % (key, param_dict[key]))
    

def parse_param():
    long_opts_list = ['ref_dir=', 'bim_prefix=', 'sst_file=', 'a=', 'b=', 'phi=', 'n_gwas=', 'n_iter=', 'n_burnin=', 'thin=', 'out_dir=', 'chrom=', 'beta_std=', 'seed=', 'help' ,'save_fig_name=','anno_path=','test_on=']
    
    param_dict = {'ref_dir': '/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD_PRScs/ldblk_ukbb_eur', 'bim_prefix': None, 'sst_file': None, 'a': 1, 'b': 0.5, 'phi': None, 'n_gwas': 200000, 'n_iter': 1000, 'n_burnin': 500, 'thin': 5, 'out_dir': 'test_data', 'chrom': 22, 'beta_std': 'False', 'seed': 42,'save_fig_name': '/gpfs/commons/home/tlin/pic/casioPR/simulation/test_test','anno_path':None,'test_on':'sim'}
    
    ## out_dir, a, b, thin, n_burnin, thin are for PRSCS 
    
    if len(sys.argv) > 1:    
        try:
            opts, args = getopt.getopt(sys.argv[1:], "h", long_opts_list)
     
        except:
            print('Option not recognized.')
            print('Use --help for usage information.\n')
            sys.exit(2)

        for opt, arg in opts:
            if opt == "-h" or opt == "--help":
                print(__doc__)
                sys.exit(0)
            # elif opt == "--ref_dir": param_dict['ref_dir'] = arg
            # elif opt == "--bim_prefix": param_dict['bim_prefix'] = arg
            # elif opt == "--sst_file": param_dict['sst_file'] = arg
            # elif opt == "--a": param_dict['a'] = float(arg)
            # elif opt == "--b": param_dict['b'] = float(arg)
            # elif opt == "--n_gwas": param_dict['n_gwas'] = int(arg)
            # elif opt == "--n_iter": param_dict['n_iter'] = int(arg)
            # elif opt == "--out_dir": param_dict['out_dir'] = arg
            elif opt == "--chrom": param_dict['chrom'] = arg.split(',')
            # elif opt == "--beta_std": param_dict['beta_std'] = arg
            # elif opt == "--seed": param_dict['seed'] = int(arg)
            elif opt == "--save_fig_name": param_dict['save_fig_name'] = arg
            elif opt == "--anno_path": param_dict['anno_path'] = arg
            elif opt == "--test_on": param_dict['test_on'] = param_dict['chrom']
    else:
        print(__doc__)
        print('### Please put input as suggested in the usage above ###')
        sys.exit(0)

    
    # if param_dict['ref_dir'] == None:
    #     print('* Please specify the directory to the reference panel using --ref_dir\n')
    #     sys.exit(2)
    # elif param_dict['bim_prefix'] == None:
    #     print('* Please specify the directory and prefix of the bim file for the target dataset using --bim_prefix\n')
    #     sys.exit(2)
    # elif param_dict['sst_file'] == None:
    #     print('* Please specify the summary statistics file using --sst_file\n')
    #     sys.exit(2)
    # elif param_dict['n_gwas'] == None:
    #     print('* Please specify the sample size of the GWAS using --n_gwas\n')
    #     sys.exit(2)
    # elif param_dict['out_dir'] == None:
    #     print('* Please specify the output directory using --out_dir\n')
    #     sys.exit(2)

    for key in param_dict:
        print('--%s=%s' % (key, param_dict[key]))
        
    print('\n')
    return param_dict


def main():
    sim_result_dict()
    #check_sim_result(args.save_fig_name, args.anno_path, args.test_on, refit_time = args.refit_time, prop_nz = args.prop_nz, phi_as_prior = args.phi_as_prior,  constrain_sigma = args.constrain_sigma, lr = args.lr, chrom = args.chrom)
    
if __name__ == '__main__':
    main()
