#!/usr/bin/python
"""
Read LD blocks and write as hdf5 (per chr)
"""

import numpy as np
import h5py,os,sys

chr = sys.argv[1]
BLK_DIR = '/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD_ADSP36K_4PRScs/not_na'

LABEL = "adsp"
# with open(os.path.join(BLK_DIR,'count','blk_chr'+chr)) as ff:
#     blk_chr = [int(line.strip()) for line in ff]

with open(os.path.join(BLK_DIR,'count','chr'+chr+'_blk_size')) as ff:
    blk_size = [int(line.strip()) for line in ff]

n_blk = len(blk_size)


#out file
out_dir =  os.path.join(BLK_DIR,f'ldblk_{LABEL}_chr')
os.makedirs(out_dir,exist_ok=True)
chr_name = os.path.join(out_dir,f'ldblk_{LABEL}_chr' + str(chr) )
print(f'... parse chomosome {chr} {chr_name} ...')

#hdf_chr = h5py.File(chr_name, 'w')
 
ld = []; snplist = []
# check that block is not empty
hdf_chr = h5py.File(chr_name +  '.hdf5', 'w')
for blk in range(1,n_blk+1):
    #print('start blk %d'%blk)
    if (blk_size[blk-1] > 0) : # check that block is not empty
        blk_root = os.path.join(BLK_DIR, 'ldblk','chr' + str(chr) + '_' + str(blk) )
        #read actual ld matrix
        with open(blk_root + '.ld') as ff:
            ld = [[float(val) for val in (line.strip()).split()] for line in ff]
            nan_count = np.isnan(ld).sum()
            if nan_count > 0 :
                ld = np.nan_to_num(ld, nan=0)
                print("convert %d nan to 0"%nan_count)
                
        print(f'blk {blk} {np.shape(ld)}')

        #get blocksnplist  
        with open(blk_root + '.snplist') as ff:
            snplist = [str(line.strip()) for line in ff]

        hdf_blk = hdf_chr.create_group('blk_%d' % (blk))
        hdf_blk.create_dataset('ldblk', data=np.array(ld), compression="gzip", compression_opts=9)
        data = [elem.encode() for elem in snplist]
        hdf_blk.create_dataset('snplist', data=data, compression="gzip", compression_opts=9)

    else:
        print('pass blk %d'%blk)

print('finished')

         
          
            