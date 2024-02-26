#!/bin/bash

# Define combinations of arguments
arguments=(
    "--save_fig_name 36k_prior_learnt_chr1_22 --beta_prior_a None"
    "--save_fig_name 36k_prior_SB_chr1_22 --beta_prior_a 1"
    "--save_fig_name 36k_prior_horseshoe_chr1_22 --beta_prior_a 0.5"
    "--save_fig_name 36k_half_cauchy_chr1_22 --beta_prior_a 0"
    "--save_fig_name 36k_inf_chr1_22 --beta_prior_a inf"
)

# Submit jobs
for args in "${arguments[@]}"; do
    sbatch << EOF
#!/bin/bash
#SBATCH --job-name='chr1_22_36k'
#SBATCH --partition=pe2
#SBATCH --nodes=1           # minimum number of nodes to be allocated
#SBATCH --ntasks=1          # number of tasks
#SBATCH --cpus-per-task=8   # number of cores on the CPU for the task
#SBATCH --mem=100G
#SBATCH --time=99:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=tlin@nygenome.org
#SBATCH --output=/gpfs/commons/home/tlin/pic/casioPR/simulation/test_prior/%x_%j.log

source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate
conda activate polyfun

python test_simulation.py $args --anno_path False --refit_time 2 --chrom_start 1
EOF
done
