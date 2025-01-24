# file='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD/plinkfile_hg38_rerun/ADSP_EUR_'   ##Updated 
# #output='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD/LD_ADSP36K_4PRScs_OCT'
# output='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD/LD_ADSP36K_nogap'
# #preset_block='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD/ldblk_hg38.pos'
# preset_block='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD/ldblk_hg38_nogap.pos'
 

# touch /gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD/snp_count_nogap.txt
# #touch /gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD/snp_count.txt

# read_ld_block() {
#     local chr="$1"
#     local start="$2"
#     local end="$3"
#     wc -l < "$output/snplist_ldblk/${chr}_${start}_${end}.txt"
# }

# # Read each line from $preset_block (excluding the first line) and call the function
# tail -n +2 "$preset_block" | while read -r chr start stop; do
#     line_count=$(read_ld_block "$chr" "$start" "$stop")
#     echo "$chr $start $stop $line_count" | tee -a /gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD/snp_count_nogap.txt
#     #echo $chr $start $stop $line_count | tee -a /gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD/snp_count.txt
# done


block=1
dir_blk='/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD/LD_ADSP36K_4PRScs_OCT/snplist_ldblk'
touch /gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD/LD_ADSP36K_4PRScs_OCT/snp_count.txt

for chr in {1..22}
do
    block=1
    echo chr$chr
    for i in ${dir_blk}/chr${chr}_*
    do
        file_name=$(basename $i)
        chr=$(echo "$file_name" | awk -F'_' '{print $1}')
        start=$(echo "$file_name" | awk -F'_' '{print $2}')
        end=$(echo "$file_name" | awk -F'_' '{print $3}' | sed 's/.txt//')
        snp_count=$(wc -l < "$i" | awk '{print $1}')
        echo $chr $block  $start $end $snp_count >> /gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD/LD_ADSP36K_4PRScs_OCT/snp_count.txt
        ((block ++))
        
    done
done