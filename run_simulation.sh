#!/bin/bash
#SBATCH --job-name=test_prior
#SBATCH --partition=pe2
#SBATCH --nodes=1           # minimum number of nodes to be allocated
#SBATCH --ntasks=1          # number of tasks
#SBATCH --cpus-per-task=8   # number of cores on the CPU for the task
#SBATCH --mem=15G
#SBATCH --time=34:00:00
#SBATCH --output=/gpfs/commons/home/tlin/pic/casioPR/simulation/%x_%j.log


source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate
conda activate polyfun


bl_anno='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/baseline/baseline_high_h2_chr'
deepsea='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/deepsea/deepsea_high_h2_chr'
enformer='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/enformer/enformer_high_h2_chr'

all_anno='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/baseline/baseline_high_h2_chr,/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/deepsea/deepsea_high_h2_chr,/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/enformer/enformer_high_h2_chr'

python test_simulation.py --save_fig_name prior_learnt --anno_path False --test_on sim --beta_prior_a None --refit_time 10 &
python test_simulation.py --save_fig_name SB_prior --anno_path False --test_on sim --beta_prior_a 1 --refit_time 10 &
python test_simulation.py --save_fig_name inf_prior --anno_path False --test_on sim --beta_prior_a inf --refit_time 10 &
python test_simulation.py --save_fig_name horseshoe --anno_path False --test_on sim --beta_prior_a 0.5 --refit_time 10


#python test_simulation.py normal_dist_add_noise $bl_anno chr22 0.2 False True --refit_time 20
#python test_simulation.py normal_dist_add_bigger_noise False chr22 0.2 False True --refit_time 5

#python test_simulation.py normal_dist $bl_anno chr22 0.2 False True --refit_time 10


# usage: test_simulation.py [-h] [--refit_time REFIT_TIME] [--lr LR]
#                           [--chrom CHROM]
#                           save_fig_name anno_path test_on prop_nz phi_as_prior
#                           constrain_sigma

