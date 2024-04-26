#!/usr/bin/python


"""
Read LD blocks and write as hdf5
"""

import numpy as np
import h5py,os,sys

chr= sys.argv[1]
BLK_DIR = '/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD_ADSP36K_4PRScs/'

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

# loop over blocks and 
blk = int(sys.argv[2])
ld = []; snplist = []
if (blk_size[blk] > 0) : # check that block is not empty
    hdf_chr_blk = h5py.File(chr_name + '_'+ str(blk) +  '.hdf5', 'w')
    blk_root = os.path.join(BLK_DIR, 'ldblk','chr' + str(chr) + '_' + str(blk) )
    #read actual ld matrix
    with open(blk_root + '.ld') as ff:
        ld = [[float(val) for val in (line.strip()).split()] for line in ff]
    print(f'blk {blk} {np.shape(ld)}')

    #get blocksnplist  
    with open(blk_root + '.snplist') as ff:
        snplist = [str(line.strip()) for line in ff]

    hdf_blk = hdf_chr_blk.create_group('blk_%d' % (blk))
    hdf_blk.create_dataset('ldblk', data=np.array(ld), compression="gzip", compression_opts=9)
    data = [elem.encode() for elem in snplist]
    hdf_blk.create_dataset('snplist', data=data, compression="gzip", compression_opts=9)

    print(f'wrote blk {blk} {blk_size[blk]}')

else:
    print('pass')

         
          
            