#!/usr/bin/env python

"""
Parse the reference panel, summary statistics, validation set, and annotations.

"""


import os
import scipy as sp
import numpy as np
from scipy.stats import norm
from scipy import linalg
import h5py
import pandas as pd
import time
import torch
import simulate

def parse_ref(ref_file):
    print('... parse reference file: %s ...' % ref_file)
    return pd.read_csv(ref_file, sep="\t")

def parse_bim(bim_file):
    
    return pd.read_csv(bim_file, sep="\t", names = ["CHR", "SNP", "huh", "pos", "A1", "A2"], usecols = ["CHR", "SNP", "A1", "A2"])

def parse_sumstats(ref_dict, vld_dict, sst_file, n_subj):
    print('... parse sumstats file: %s ...' % sst_file)

    ATGC = ['A', 'T', 'G', 'C']
    
    sst_dict = pd.read_csv(sst_file, sep = "\t")
    sst_dict = sst_dict[sst_dict.A1.isin(ATGC) & sst_dict.A2.isin(ATGC)]

    mapping = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

    vld_snp = set(zip(vld_dict['SNP'], vld_dict['A1'], vld_dict['A2']))

    ref_snp = set(zip(ref_dict['SNP'], ref_dict['A1'], ref_dict['A2'])) | set(zip(ref_dict['SNP'], ref_dict['A2'], ref_dict['A1'])) | \
              set(zip(ref_dict['SNP'], [mapping[aa] for aa in ref_dict['A1']], [mapping[aa] for aa in ref_dict['A2']])) | \
              set(zip(ref_dict['SNP'], [mapping[aa] for aa in ref_dict['A2']], [mapping[aa] for aa in ref_dict['A1']]))
    
    sst_snp = set(zip(sst_dict['SNP'], sst_dict['A1'], sst_dict['A2'])) | set(zip(sst_dict['SNP'], sst_dict['A2'], sst_dict['A1'])) | \
              set(zip(sst_dict['SNP'], [mapping[aa] for aa in sst_dict['A1']], [mapping[aa] for aa in sst_dict['A2']])) | \
              set(zip(sst_dict['SNP'], [mapping[aa] for aa in sst_dict['A2']], [mapping[aa] for aa in sst_dict['A1']]))

    comm_snp = vld_snp & ref_snp & sst_snp

    print('Found %d common SNPs in the reference, sumstats, and validation set \n' % len(comm_snp))
    
    n_sqrt = sp.sqrt(n_subj)
    sst_eff = {}
    with open(sst_file) as ff:
        header = (next(ff).strip()).split()
        header = [col.upper() for col in header]
        for line in ff:
            ll = (line.strip()).split()
            snp = ll[0]; a1 = ll[1]; a2 = ll[2]
            if a1 not in ATGC or a2 not in ATGC:
                continue
            if (snp, a1, a2) in comm_snp or (snp, mapping[a1], mapping[a2]) in comm_snp:
                if 'BETA' in header:
                    beta = float(ll[3])
                elif 'OR' in header:
                    beta = sp.log(float(ll[3]))

                p = max(float(ll[4]), 1e-323)
                beta_std = sp.sign(beta)*abs(norm.ppf(p/2.0))/n_sqrt
                sst_eff[snp] = beta_std
            elif (snp, a2, a1) in comm_snp or (snp, mapping[a2], mapping[a1]) in comm_snp:
                if 'BETA' in header:
                    beta = float(ll[3])
                elif 'OR' in header:
                    beta = sp.log(float(ll[3]))

                p = max(float(ll[4]), 1e-323)
                beta_std = -1*sp.sign(beta)*abs(norm.ppf(p/2.0))/n_sqrt
                sst_eff[snp] = beta_std


    sst_dict = {'CHR':[], 'SNP':[], 'BP':[], 'A1':[], 'A2':[], 'MAF':[], 'BETA':[], 'FLP':[]}
    ref_dict.reset_index(inplace=True)
    
    for (ii, snp) in enumerate(ref_dict['SNP']):
        if snp in sst_eff:
            sst_dict['SNP'].append(snp)
            sst_dict['CHR'].append(ref_dict['CHR'][ii])
            sst_dict['BP'].append(ref_dict['BP'][ii])
            sst_dict['BETA'].append(sst_eff[snp])

            a1 = ref_dict['A1'][ii]; a2 = ref_dict['A2'][ii]
            if (snp, a1, a2) in comm_snp:
                sst_dict['A1'].append(a1)
                sst_dict['A2'].append(a2)
                sst_dict['MAF'].append(ref_dict['MAF'][ii])
                sst_dict['FLP'].append(1)
            elif (snp, a2, a1) in comm_snp:
                sst_dict['A1'].append(a2)
                sst_dict['A2'].append(a1)
                sst_dict['MAF'].append(1-ref_dict['MAF'][ii])
                sst_dict['FLP'].append(-1)
            elif (snp, mapping[a1], mapping[a2]) in comm_snp:
                sst_dict['A1'].append(mapping[a1])
                sst_dict['A2'].append(mapping[a2])
                sst_dict['MAF'].append(ref_dict['MAF'][ii])
                sst_dict['FLP'].append(1)
            elif (snp, mapping[a2], mapping[a1]) in comm_snp:
                sst_dict['A1'].append(mapping[a2])
                sst_dict['A2'].append(mapping[a1])
                sst_dict['MAF'].append(1-ref_dict['MAF'][ii])
                sst_dict['FLP'].append(-1)
    
    
    sst_df = pd.DataFrame(sst_dict)
    sst_file = pd.read_csv(sst_file, sep = "\t")
    sst_df = sst_df.merge(sst_file[['SNP', 'P']], on='SNP', how='left')
    
    return sst_df



def parse_ldblk(ldblk_dir, sst_dict, chrom):
    print('... parse reference LD on chromosome %s ...' % chrom)

    if '1kg' in os.path.basename(ldblk_dir):
        chr_name = ldblk_dir + '/ldblk_1kg_chr' + str(chrom) + '.hdf5'
    elif 'ukbb' in os.path.basename(ldblk_dir):
        chr_name = ldblk_dir + '/ldblk_ukbb_chr' + str(chrom) + '.hdf5'

    hdf_chr = h5py.File(chr_name, 'r') ## read ldblk(in hdf5 format)
    n_blk = len(hdf_chr)
    ld_blk = [sp.array(hdf_chr['blk_'+str(blk)]['ldblk']) for blk in range(1,n_blk+1)]

    snp_blk = []
    for blk in range(1,n_blk+1):
        snp_blk.append([bb.decode("UTF-8") for bb in list(hdf_chr['blk_'+str(blk)]['snplist'])])

    blk_size = []
    ld_blk_sym = [] #symmetric matrices -> to gets eigen values
    ld_blk_filt = [] # get the correctly flipped ones
    mm = 0 ## mm is the number of SNPs
    
    for blk in range(n_blk):
        idx = [ii for (ii, snp) in enumerate(snp_blk[blk]) if snp in sst_dict['SNP'].to_numpy() ]
        #print(len(idx))
        if len(idx) == 0: 
            continue
            
        blk_size.append(len(idx))
        
        idx_blk = np.arange(mm,mm+len(idx))
        flip = sst_dict['FLP'][idx_blk]
        ld_blk_here = ld_blk[blk][sp.ix_(idx,idx)]*sp.outer(flip,flip)
        ld_blk_filt.append(ld_blk_here)
        
        #_, s, v = linalg.svd(ld_blk_here)
        #h = sp.dot(v.T, sp.dot(sp.diag(s), v)) # just weird way of getting transpose?! 
        #ld_blk_sym.append( (ld_blk_here+h)/2 )
        
        ld_blk_sym.append( (ld_blk_here+ld_blk_here.T)/2 )

        mm += len(idx)
        
    return ld_blk_filt, ld_blk_sym, blk_size



def parse_ldblk_test(ldblk_dir, sst_dict, chrom, sim=False):
    #print('... parse reference LD on chromosome %s ...' % chrom)

    if '1kg' in os.path.basename(ldblk_dir):
        chr_name = ldblk_dir + '/ldblk_1kg_chr' + str(chrom) + '.hdf5'
    elif 'ukbb' in os.path.basename(ldblk_dir):
        chr_name = ldblk_dir + '/ldblk_ukbb_chr' + str(chrom) + '.hdf5'

    hdf_chr = h5py.File(chr_name, 'r') ## read ldblk(in hdf5 format)
    n_blk = len(hdf_chr)
    ld_blk = [sp.array(hdf_chr['blk_'+str(blk)]['ldblk']) for blk in range(1,n_blk+1)]

    snp_blk = []
    for blk in range(1,n_blk+1):
        snp_blk.append([bb.decode("UTF-8") for bb in list(hdf_chr['blk_'+str(blk)]['snplist'])])

    blk_size = []
    ld_blk_sym = [] #symmetric matrices -> to gets eigen values
    ld_blk_filt = [] # get the correctly flipped ones
    mm = 0 ## mm is the number of SNPs
    for blk in range(n_blk):
        ## getes the blks where the SNPs were found in the intersection sst_dict
        if sim:
            print('sim')
            idx = [ii for (ii, snp) in enumerate(snp_blk[blk])] 
            print(len(idx))
    
        else:
            idx = [ii for (ii, snp) in enumerate(snp_blk[blk]) if snp in sst_dict['SNP'].to_numpy()]
            
        #print(len(idx))
        if len(idx) == 0: 
            continue
        
        blk_size.append(len(idx))
        
        idx_blk = np.arange(mm,mm+len(idx))
        flip = sst_dict['FLP'][idx_blk]
        ld_blk_here = ld_blk[blk][sp.ix_(idx,idx)]*sp.outer(flip,flip)
        ld_blk_filt.append(ld_blk_here)
        
        #_, s, v = linalg.svd(ld_blk_here)
        #h = sp.dot(v.T, sp.dot(sp.diag(s), v)) # just weird way of getting transpose?! 
        #ld_blk_sym.append( (ld_blk_here+h)/2 )
        
        ld_blk_sym.append( (ld_blk_here+ld_blk_here.T)/2 )

        mm += len(idx)
        
    return ld_blk_filt, ld_blk_sym, blk_size



def parse_anno(anno_file, sst_dict, chrom, flipping=False):
    """
    If no annotation is fed(anno_file = False), will simulate perfect annotations defined in simulate.py
    
    If don't want to use any annotation, use anno_file = 'None'
    flipping annotations if A1,A2 is opposite with the sst (default is false)
    """
    print('... parse annotations ...')
    t0 = time.time()
    if anno_file is None:
        print('No annotation used \n')
        return(None, None)
    else:
        anno_files = anno_file.split(',') if ',' in anno_file else [anno_file]
        anno_list = []
        for anno in anno_files:
            anno_path = anno + f'{chrom}.annot.gz'
            if not os.path.exists(anno_path):
                raise IOError(f'Cannot find annotation file {anno}')
            anno_list.append(anno_path)
        print("Reading annotations from %d file(s)... "%len(anno_list))

        ## parsing multiple annoation files
        if len(anno_list) == 1:
            anno_df =pd.read_csv(anno_list[0], compression='gzip',sep = '\t') 
        else:
            anno_df_list = [pd.read_csv(file, compression='gzip',sep='\t') for file in anno_list]
            columns_match = all(anno_df_list[0][['CHR', 'SNP', 'BP', 'A1', 'A2']].equals(df[['CHR', 'SNP', 'BP', 'A1', 'A2']]) for df in anno_df_list[1:])
            if columns_match:
                anno_df = pd.concat(anno_df_list, axis=1)
                anno_df = anno_df.loc[:, ~anno_df.columns.duplicated()]
            else:
                raise IOError("Failed to merge annotations")
        print("Successfully loaded %d annotations for %d SNPs" %(anno_df.shape[1]-5, anno_df.shape[0]))

        anno_merge = sst_dict[['SNP','A1','A2']].merge(anno_df, on = 'SNP', suffixes=('', '_y')) 
        print('Total of %d SNPs left after merging with sumstat'%(anno_merge.shape[0]))

        ## flipping annotations if A1,A2 is opposite with the sst (default is false)
        if flipping:
            flipping = anno_merge.loc[anno_merge["A1"] == anno_merge['A2_y']]
            if flipping.shape[0] > 0 :
                print('Flipping annotaions for %d rows'% flipping.shape[0])
                for col_index in range(7, anno_merge.shape[1]):
                    anno_merge.loc[anno_merge["A1"] == anno_merge['A2_y'], anno_merge.columns[col_index]] = -anno_merge.iloc[:, col_index]

        anno_merge = anno_merge.drop(["A1_y", 'A2_y'], axis=1)
        anno_torch = torch.cat((torch.ones((anno_merge.shape[0],1)),torch.tensor(anno_merge.iloc[:,5:].values)), dim=1) ## because there are A1, A2, SNP, CHR, and BP. Add torch.ones to meet the requirement for interception
        print('Done in %0.2f seconds \n'%(time.time() - t0))
        return(anno_torch.float(),anno_df.columns[5:].tolist())