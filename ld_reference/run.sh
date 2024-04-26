#!/bin/bash 
for chr in {1..14}
do
        echo start chr $chr  	
        #sbatch calc_ld.sh $chr
        sbatch create_blk_chr_file.sh $chr
done


## 15-22 all
## 15-19_blk 32-40