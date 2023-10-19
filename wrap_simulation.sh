#!/bin/bash
#SBATCH --job-name=perfect_0.1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=tlin@nygenome.org
#SBATCH --mem=10G
#SBATCH --time=50:00:00
#SBATCH --output=/gpfs/commons/home/tlin/pic/casioPR/simulation/%x_%j.log


source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate
conda activate polyfun


bl_anno='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/baseline/baseline_high_h2_chr'
deepsea='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/deepsea/deepsea_high_h2_chr'
enformer='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/enformer/enformer_high_h2_chr'

all_anno='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/baseline/baseline_high_h2_chr,/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/deepsea/deepsea_high_h2_chr,/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/annotations_high_h2/enformer/enformer_high_h2_chr'

# ## test PHI
# python test_simulation.py phi_prior_true_chr22 $all_anno chr22 0.2 True True --refit_time 10 &
# python test_simulation.py phi_prior_false_chr22 $all_anno chr22 0.2 False True --refit_time 10 

## test perfect anno
#python test_simulation.py prop_nz_0.1_chr22 False chr22 0.1 True True --refit_time 10	
#python test_simulation.py prop_nz_0.05_sim --refit_time 1
python test_simulation.py prop_nz_0.1_chr22 False chr22 0.1 True True --refit_time 10
#python test_simulation.py prop_nz_0.2_chr22 False chr22 0.2 True True --refit_time 10


# # # test sigma
# python test_simulation.py sigma_false_chr22 $all_anno chr22 0.2 True False --refit_time 10 &
# python test_simulation.py sigma_true_chr22 $all_anno chr22 0.2 True True --refit_time 10
