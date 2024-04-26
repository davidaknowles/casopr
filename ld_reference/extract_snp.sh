file='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/ADSP_vcf/1KG/ADSP_EUR/plink/ADSP_bellenguez_snps_'
output='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD_ADSP36K_4PRScs'
preset_block='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD_ADSP36K_4PRScs/PRSCS_preset_ldblock.txt'

# awk -v end="$end" -v start="$start" '$4 <= end && $4 >= max {print $0}' ${file}chr1.bim 


read_ld_block() {
    local chr="$1"
    local start="$2"
    local end="$3"
    
    awk -v chr='$chr' -v end="$end" -v start="$start" '$4 <= end && $4 >= start {print $0}' ${file}${chr}.bim >  $output/snplist_ldblk/${chr}_${start}_${end}.txt
}

tail -n +2  $preset_block | while read -r chr start stop; do
    echo $chr $start $stop
    read_ld_block $chr $start $stop
    
done