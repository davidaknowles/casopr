cd /gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/LD/LD_ADSP36K_4PRScs_OCT/count
cat snpinfo_chr1 > snpinfo_adsp_hm3

for chr in {2..22}
do
    echo snpinfo_chr${chr}
    cat snpinfo_chr${chr} | tail -n+2  >> snpinfo_adsp_hm3
done 

#tr ' ' '\t' < snpinfo_adsp_hm3 > tmpfile && mv tmpfile snpinfo_adsp_hm3

cat snpinfo_adsp_hm3 | tr ' ' '\t'| awk '($6 != "NA") {print $0}' > snpinfo_adsp_hm3_2
mv snpinfo_adsp_hm3_2 snpinfo_adsp_hm3
cp snpinfo_adsp_hm3 ../ldblk/
