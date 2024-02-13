#!/bin/bash

# Define combinations of arguments
arguments=(
    "--save_fig_name prior_learnt_chr20_22 --beta_prior_a None"
    "--save_fig_name prior_SB_chr20_22 --beta_prior_a 1"
    "--save_fig_name prior_horseshoe_chr20_22 --beta_prior_a 0.5"
    "--save_fig_name half_cauchy_chr20_22 --beta_prior_a 0"
    "--save_fig_name inf_chr20_22 --beta_prior_a inf"
)

# Submit jobs
for args in "${arguments[@]}"; do
    sbatch << EOF
#!/bin/bash
#SBATCH --job-name='chr20_22_test'
#SBATCH --partition=pe2
#SBATCH --nodes=1           # minimum number of nodes to be allocated
#SBATCH --ntasks=1          # number of tasks
#SBATCH --cpus-per-task=8   # number of cores on the CPU for the task
#SBATCH --mem=100G
#SBATCH --time=99:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=tlin@nygenome.org
#SBATCH --output=/gpfs/commons/home/tlin/pic/casioPR/simulation/test_prior/%x_%j.log


python test_simulation.py $args --anno_path False --refit_time 20 --chrom_start 20	
EOF
done
