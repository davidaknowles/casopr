cd /gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD_ADSP36K_4PRScs/


for i in snplist_ldblk/chr*
    do    
        echo $i
        awk 'NR == FNR { file2_set[$1]; next } !($2 in file2_set)' not_na/snp_na.txt $i > not_na/$i
done


# i=snplist_ldblk/chr22_16050408_17674295.txt
# #awk 'FNR == NR {file2_set[$2]; nextfile} {for (value in file2_set) if ($2 == value) next} 1' not_na/snp_na.txt $i > not_na/$i

# awk 'NR == FNR { file2_set[$1]; next } !($2 in file2_set)' not_na/snp_na.txt $i > not_na/$i
