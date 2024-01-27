#!/bin/bash
#SBATCH --job-name=iteration_chr
#SBATCH --partition=pe2
#SBATCH --nodes=1           # minimum number of nodes to be allocated
#SBATCH --ntasks=1          # number of tasks
#SBATCH --cpus-per-task=8   # number of cores on the CPU for the task
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/commons/home/tlin/pic/casioPR/simulation/%x_%j.log


source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate
conda activate polyfun


bl_anno='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/baseline/baseline_high_h2_chr'
deepsea='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/deepsea/deepsea_high_h2_chr'
enformer='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/enformer/enformer_high_h2_chr'

all_anno='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/baseline/baseline_high_h2_chr,/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/deepsea/deepsea_high_h2_chr,/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/enformer/enformer_high_h2_chr'

python test_simulation.py --save_fig_name iterate_chr20_22 --anno_path False --test_on chr22 --gaussian_anno_weight True --refit_time 1 --noise_size 0.5

#python test_simulation.py --save_fig_name mid_noise_gaussian --anno_path False --test_on chr22 --gaussian_anno_weight True --refit_time 20 --noise_size 0.5 

# python test_simulation.py --save_fig_name normal_big_noise --anno_path False --test_on chr22 --gaussian_anno_weight True --noise_size 0.1 --refit_time 20


# python test_simulation.py --save_fig_name dirich_noise --anno_path False --test_on chr22 --gaussian_anno_weight False --noise_size 0.01 --refit_time 20 &

# python test_simulation.py --save_fig_name dirich_big_noise --anno_path False --test_on chr22 --gaussian_anno_weight False --noise_size 0.1 --refit_time 20



#python test_simulation.py --save_fig_name normal_no_noise --anno_path False --test_on chr22 --gaussian_anno_weight False --noise_size 0 --refit_time 20

#python test_simulation.py --save_fig_name normal_big_noise --anno_path False --test_on chr22 --gaussian_anno_weight True --noise_size 0.1 --refit_time 20

# python test_simulation.py --save_fig_name dirich_bl_chr22 --anno_path $bl_anno --test_on chr22 --gaussian_anno_weight False --refit_time 20 

# python test_simulation.py --save_fig_name noise_chr22 --anno_path False --test_on chr22 --gaussian_anno_weight True --noise_size 0.01 --refit_time 20



#python test_simulation.py normal_dist_add_noise $bl_anno chr22 0.2 False True --refit_time 20
#python test_simulation.py normal_dist_add_bigger_noise False chr22 0.2 False True --refit_time 5

#python test_simulation.py normal_dist $bl_anno chr22 0.2 False True --refit_time 10


# usage: test_simulation.py [--refit_time REFIT_TIME] [--lr LR]
#                           [--chrom CHROM]
#                           save_fig_name anno_path test_on prop_nz phi_as_prior
#                           constrain_sigma

