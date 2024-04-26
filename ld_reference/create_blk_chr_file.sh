#!/bin/bash
#SBATCH --job-name=hdf5
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=tlin@nygenome.org
#SBATCH --mem=150G
#SBATCH --time=15:00:00
#SBATCH --array=1-12%10
#SBATCH --output=/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD_ADSP36K_4PRScs/snps_only/ldblk_adsp_chr/%x_%j.log

# source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate
# conda activate polyfun

dir_blk="/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD_ADSP36K_4PRScs"
dir_1kg="/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/ADSP_vcf/1KG/ADSP_EUR/plink"

cd ${dir_blk}/snps_only/ldblk/
block=1
#chr=$SLURM_ARRAY_TASK_ID
chr=$1
if test -f  ../count/blk_chr${chr} ; then
            rm ../count/chr${chr}_blk_size
            rm ../count/blk_chr${chr}
fi

echo chr $chr
echo -e "CHR\tSNP\tBP\tA1\tA2\tMAF" > ../count/snpinfo_chr${chr}
for i in ../../snplist_ldblk/chr${chr}_*
    do    
        touch ../count/blk_chr${chr}
        echo $chr >> ../count/blk_chr${chr}
        
        touch ../count/chr${chr}_blk_size
        touch chr${chr}_${block}.bim && wc -l < chr${chr}_${block}.bim >> ../count/chr${chr}_blk_size
        # paste <(cut -f 1,2,4,5,6 chr${chr}_${block}.bim) <( sed -E 1d chr${chr}_${block}.frq |awk '{$1=$1;print}' | cut -d " " -f5) <(yes "$block" | head -n $(wc -l < "chr${chr}_${block}.bim")) >> ../count/snpinfo_chr${chr}
        echo run block $block
        awk -v chr="$chr" -v block="$block" '
        NR == FNR {
            bim[$2]
            next
        }
        ($2 in bim) {
            print $0 > "chr"chr"_"block"_maf.frq"
        }' "chr${chr}_${block}.bim" "chr${chr}_${block}.frq"
 
        paste -d '\t' <(awk '{print $1, $2, $4, $5, $6}' chr${chr}_${block}.bim) <(awk '{print $5}' chr${chr}_${block}_maf.frq) >> ../count/snpinfo_chr${chr}
        ((block ++))
       # paste  <(awk '{print $1, $2, $4, $5, $6}' chr${chr}_${block}.bim) <(tail -n +2 chr${chr}_${block}.frq | awk '{print $5}') >> ../count/snpinfo_chr${chr}

    done

sed -i 's/ /\t/g' ../count/snpinfo_chr${chr}
if grep -q '\b0\b' ../count/chr${chr}_blk_size; then
    echo "there is empty block"
fi 
#echo 'creating h5 files... '
python /gpfs/commons/home/tlin/script/casopr/ld_reference/write_ldblk_chr.py $chr


