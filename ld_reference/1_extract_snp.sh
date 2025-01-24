#file='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD/plink_file_hg38/ADSP_EUR_'
file='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD/plinkfile_hg38_rerun/ADSP_EUR_'   ##Updated 
#output='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD/LD_ADSP36K_4PRScs_OCT'
output='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD/LD_ADSP36K_nogap'
#preset_block='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD_ADSP36K_4PRScs/PRSCS_preset_ldblock.txt'
#preset_block='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD/ldblk_hg38.pos'
preset_block='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD/ldblk_hg38_nogap.pos'

# awk -v end="$end" -v start="$start" '$4 <= end && $4 >= max {print $0}' ${file}chr1.bim 

#mkdir $output/snplist_ldblk

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