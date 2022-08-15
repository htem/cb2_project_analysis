

export SILENT_MODE=1
runscript='runscript_210629_1559'
echo '' > $runscript
for model in observed ; do
for grc_pct in 1 .5 ; do
for grc_pct_learned in 0 1 ; do
for sub_feature_mode in 0 1 ; do
for mfs_z_margin in 0 10000; do
echo "python batch_210627_snr.py --model $model --grc_pct $grc_pct --grc_pct_learned $grc_pct_learned --sub_feature_mode $sub_feature_mode --mfs_z_margin $mfs_z_margin "  >> $runscript
done
done
done
done
done
parallel < $runscript


export SILENT_MODE=1
runscript='runscript_210629_1036'
echo '' > $runscript
for model in observed ; do
for grc_pct in 1 .5 ; do
for grc_pct_learned in 0 1 ; do
for sub_feature_mode in 0 1 ; do
for mfs_z_margin in 0 10000; do
echo "python batch_210627_snr_with_dim.py --model $model --grc_pct $grc_pct --grc_pct_learned $grc_pct_learned --sub_feature_mode $sub_feature_mode --mfs_z_margin $mfs_z_margin "  >> $runscript
done
done
done
done
done
parallel < $runscript


export SILENT_MODE=1
runscript='runscript_210626_1818'
echo '' > $runscript
for model in local_random ; do
for grc_scale in 1 .5 ; do
echo "python batch_210623_dim_gt_grc_pct.py --model $model --grc_scale $grc_scale"  >> $runscript
done
done
parallel < $runscript


export SILENT_MODE=1
runscript='runscript_210624_1818'
echo '' > $runscript
for model in local_random; do
for mf_limit in .5 .33; do
echo "python batch_210623_dim_skewed_mfs.py --model $model --top_mf_mask 1 --mf_mask_limit $mf_limit"  >> $runscript
echo "python batch_210623_dim_skewed_mfs.py --model $model --top_mf_mask $mf_limit --mf_mask_limit $mf_limit"  >> $runscript
echo "python batch_210623_dim_skewed_mfs.py --model $model --top_mf_mask -$mf_limit --mf_mask_limit $mf_limit"  >> $runscript
done
done
parallel < $runscript
