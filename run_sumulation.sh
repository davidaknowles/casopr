#!/bin/bash
#SBATCH --job-name=sigma_True
#SBATCH --partition=pe2
#SBATCH --nodes=1           # minimum number of nodes to be allocated
#SBATCH --ntasks=1          # number of tasks
#SBATCH --cpus-per-task=8   # number of cores on the CPU for the task
#SBATCH --mem=5G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/commons/home/tlin/pic/casioPR/simulation/%x_%j.log


source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate
conda activate polyfun


bl_anno='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/baseline/baseline_high_h2_chr'
deepsea='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/deepsea/deepsea_high_h2_chr'
enformer='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/enformer/enformer_high_h2_chr'

all_anno='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/baseline/baseline_high_h2_chr,/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/deepsea/deepsea_high_h2_chr,/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/enformer/enformer_high_h2_chr'



python test_simulation.py just_testing False sim 0.2 True True --refit_time 3

# usage: test_simulation.py [-h] [--refit_time REFIT_TIME] [--lr LR]
#                           [--chrom CHROM]
#                           save_fig_name anno_path test_on prop_nz phi_as_prior
#                           constrain_sigma

