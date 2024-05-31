#!/bin/bash
#SBATCH --job-name=casiopr_bellenguez19
#SBATCH --partition=pe2
#SBATCH --nodes=1           # minimum number of nodes to be allocated
#SBATCH --ntasks=1          # number of tasks
#SBATCH --cpus-per-task=8   # number of cores on the CPU for the task
#SBATCH --mem=40G
#SBATCH --time=39:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --array=1-10%5
#SBATCH --mail-user=tlin@nygenome.org
#SBATCH --output=/gpfs/commons/home/tlin/pic/casioPR/simulation/adsp_ld/%x_%j.log


source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate
conda activate polyfun

# prior=$SLURM_ARRAY_TASK_ID
# anno="genomewide_perfect_"
# if ((prior % 5 == 1)); then
#     prior="None"
#     name="learnt"
# elif ((prior % 5 == 2)); then
#     prior=1
#     name='SB'
# elif ((prior % 5 == 3)); then
#     prior=0.5
#     name='horseshoe'
# elif ((prior % 5 == 4)); then
#     prior=0
#     name='half_cauchy'
# else
#     prior="inf"
#     name='inf'
# fi

# name+=$anno
# echo $name
# echo $prior

bl_anno='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/baseline/baseline_high_h2_chr'
deepsea='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/deepsea/deepsea_high_h2_chr'
enformer='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/enformer/enformer_high_h2_chr'
all_anno='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/baseline/baseline_high_h2_chr,/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/deepsea/deepsea_high_h2_chr,/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/enformer/enformer_high_h2_chr'

python test_simulation.py --save_fig_name $name --anno_path False --beta_prior_a $prior --refit_time 1 --chrom_start 22 --which_dict bellenguez19
#python test.py --save_fig_name test_prscs --anno_path False --beta_prior_a 0 --refit_time 1 --chrom_start 22
#python test.py --save_fig_name test --anno_path False --beta_prior_a 0 --refit_time 1 --chrom_start 22 --which_dict bellenguez19

#python test_simulation.py --save_fig_name prior_learnt_chr10_22 --anno_path False --beta_prior_a None --refit_time 20 --chrom_start 10 
#python test_simulation.py --save_fig_name prior_SB_chr10_22 --anno_path False --beta_prior_a 1 --refit_time 20 --chrom_start 10 
# python test_simulation.py --save_fig_name prior_horseshoe_chr10_22 --anno_path False --beta_prior_a 0.5 --refit_time 20 --chrom_start 10 
# python test_simulation.py --save_fig_name half_cauchy_chr10_22 --anno_path False  --beta_prior_a 0 --refit_time 20 --chrom_start 10 
# python test_simulation.py --save_fig_name adsp_reference --anno_path False --beta_prior_a inf --refit_time 20 --chrom_start 20 


# python test_simulation.py --save_fig_name prior_learnt_sim --anno_path False --test_on sim --beta_prior_a None --refit_time 20
# python test_simulation.py --save_fig_name prior_SB_sim --anno_path False --test_on sim --beta_prior_a 1 --refit_time 20
# python test_simulation.py --save_fig_name prior_horseshoe_sim --anno_path False --test_on sim --beta_prior_a 0.5 --refit_time 20
# python test_simulation.py --save_fig_name half_cauchy_sim --anno_path False --test_on sim --beta_prior_a 0 --refit_time 20
# python test_simulation.py --save_fig_name inf_sim --anno_path False --test_on sim --beta_prior_a inf --refit_time 20



# usage: test_simulation.py [--save_fig_name] [--anno_path][--beta_prior_a]
#                           [--gaussian_anno_weight (bool)] [--noise_size][--refit_time REFIT_TIME] [--lr LR] 
#                           [--chrom_start CHROM][--use_sim_dict]
