#!/bin/bash
#SBATCH --job-name=chr1_calculate_ld
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=tlin@nygenome.org
#SBATCH --mem=30G
#SBATCH --time=15:00:00
#SBATCH --array=1-161%15
#SBATCH --output=/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD/LD_ADSP36K_4PRScs_OCT/ldblk/%x_%j.log

chr=1
dir_blk="/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD/LD_ADSP36K_4PRScs_OCT/"
dir_1kg='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD/plinkfile_hg38_rerun/' 
#dir_1kg="/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD/plink_file_hg38/"

cd $dir_blk

if [ ! -e ldblk ];then
    mkdir ldblk
fi


echo chr $chr
    
block=1
echo set_block $SLURM_ARRAY_TASK_ID
for i in ${dir_blk}/snplist_ldblk/chr${chr}_*
    do    
        if [ ! -e ldblk/chr${chr}_${block}.ld ] && ((block == SLURM_ARRAY_TASK_ID )) ; then
            echo run blk $block
            /gpfs/commons/home/tlin/plink \
            --bfile ${dir_1kg}/ADSP_EUR_chr${chr} \
            --keep-allele-order --extract $i --r square yes-really --out ${dir_blk}/ldblk/chr${chr}_${block} --memory 2000 --write-snplist -maf --make-just-bim --freq --snps-only 
            
        fi       
        ((block ++))
    done