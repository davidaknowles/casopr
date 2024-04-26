#!/bin/bash
#SBATCH --job-name=chr19_hdf
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=tlin@nygenome.org
#SBATCH --mem=100G
#SBATCH --partition=bigmem
#SBATCH --time=40:00:00
#SBATCH --array=1-40%5
#SBATCH --output=/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD_ADSP36K_4PRScs/ldblk_adsp_chr/%x_%j.log


#--array=46-47%5
source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate
conda activate polyfun

echo run chr$chr ,block: $SLURM_ARRAY_TASK_ID
python /gpfs/commons/home/tlin/script/casopr/ld_reference/write_ldblk_ge.py $chr $SLURM_ARRAY_TASK_ID

#python /gpfs/commons/home/tlin/script/casopr/merge_h5.py 22