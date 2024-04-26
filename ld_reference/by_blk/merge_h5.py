import numpy as np
import h5py,os,sys


chr= sys.argv[1]
print('run  chr', chr)
BLK_DIR = '/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD_ADSP36K_4PRScs/'

LABEL = "adsp"
with open(os.path.join(BLK_DIR,'count','chr'+chr+'_blk_size')) as ff:
    blk_size = [int(line.strip()) for line in ff]

n_blk = len(blk_size)
file_name=BLK_DIR+'ldblk_adsp_chr/'+'ldblk_adsp_chr'+chr

with h5py.File(file_name+'.hdf5',mode='w') as blk_combined:
    for blk in range(1,n_blk+1):
        print("now at blk",blk)
        blk_file = file_name+'_'+str(blk)+'.hdf5'
        print(blk_file)
        h5_blk = h5py.File(blk_file,'r') 
        blk_combined.copy(h5_blk['blk_'+str(blk)], 'blk_'+str(blk))    
        #for obj in h5_blk.keys():        
        #for obj in enumerate(h5_blk['blk_'+str(blk)]):
            #h5_blk.copy(obj, blk_combined)       
            #hdf5write('check_v1.h5','/g2/coordinates_vel',all_data1, 'w', 'append')
  
