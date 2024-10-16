#!/bin/bash
#SBATCH --job-name=blk_h5
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=tlin@nygenome.org
#SBATCH --mem=100G
#SBATCH --time=15:00:00
#SBATCH --array=1-22%6
#SBATCH --output=/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD/LD_ADSP36K_4PRScs_OCT/%x_%j.log

source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate
conda activate polyfun

dir_blk="/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD/LD_ADSP36K_4PRScs_OCT/"
python write_ldblk_chr.py $SLURM_ARRAY_TASK_ID $dir_blk


## 15-22 all
## 15-19_blk 32-40