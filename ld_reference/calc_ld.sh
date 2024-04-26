#!/bin/bash
#SBATCH --job-name=ld_chr7-14
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=tlin@nygenome.org
#SBATCH --mem=50G
#SBATCH --time=15:00:00
#SBATCH --array=1-99%5
#SBATCH --output=/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD_ADSP36K_4PRScs/snps_only/ldblk/%x_%j.log


# source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate
# conda activate polyfun

dir_blk="/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD_ADSP36K_4PRScs/"
dir_1kg="/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/ADSP_vcf/1KG/ADSP_EUR/plink"
#output='/nfs/scratch/tlin/'

cd $dir_blk/snps_only/

echo chr $chr

# block=1

# #echo set_block $run_block
# echo set_block $SLURM_ARRAY_TASK_ID
# for i in ${dir_blk}/snplist_ldblk/chr${chr}_*
#     do    
#        # echo block $block
#         if [ ! -e ${dir_blk}/ldblk/chr${chr}_${block}.ld ] && ((block == SLURM_ARRAY_TASK_ID )); then
#             /gpfs/commons/home/tlin/plink \
#             --bfile ${dir_1kg}/ADSP_bellenguez_snps_chr${chr} \
#             --keep-allele-order --extract $i --r square yes-really --out chr${chr}_${block} --memory 2000 --write-snplist -maf --make-just-bim --freq

#             echo finish block $block
#             for file_type in bim frq log snplist 
#                 do
#                     cp $output/chr${chr}_${block}.${file_type} ${dir_blk}/ldblk/
#                 done
#         fi       
#         ((block ++))
       
#     done
        
block=1
echo set_block $SLURM_ARRAY_TASK_ID
for i in ${dir_blk}/snplist_ldblk/chr${chr}_*
    do    
        if [ ! -e ldblk/chr${chr}_${block}.ld ] && ((block == SLURM_ARRAY_TASK_ID )) ; then
            echo run blk $block
            /gpfs/commons/home/tlin/plink \
            --bfile ${dir_1kg}/ADSP_bellenguez_snps_chr${chr} \
            --keep-allele-order --extract $i --r square yes-really --out ${dir_blk}/snps_only/ldblk/chr${chr}_${block} --memory 2000 --write-snplist -maf --make-just-bim --freq --snps-only 
            echo created ld
            break
            # python /gpfs/commons/home/tlin/script/casopr/ld_reference/write_ldblk_ge.py $chr $SLURM_ARRAY_TASK_ID
            # echo created h5
            # echo finish block $block
            
        fi       
        ((block ++))
    done