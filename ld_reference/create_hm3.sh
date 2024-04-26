cd /gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD_ADSP36K_4PRScs/snps_only/count
#rm snpinfo_adsp_hm3
cat snpinfo_chr1 > snpinfo_adsp_hm3

for chr in {2..22}
do
    echo snpinfo_chr${chr}
    cat snpinfo_chr${chr} | tail -n+2  >> snpinfo_adsp_hm3
done

cp snpinfo_adsp_hm3 ../ldblk_adsp_chr/c