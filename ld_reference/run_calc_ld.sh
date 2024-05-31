#!/bin/bash 
for chr in {15..21}
do
        #echo start chr $chr 
        sbatch --export=chr=$chr calc_ld.sh
    
done


## 15-22 all