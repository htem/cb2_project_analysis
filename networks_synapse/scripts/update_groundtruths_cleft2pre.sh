
# echo Fetching skeletons...
# python fetch_skeletons.py
skeleton_path=/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/evals_synapse/skeletons


function update_gt {
    vol=$1
    mkdir -p ${vol}/annotations/cleft2pre
    cp ${skeleton_path}/${vol}_skeleton.json ${vol}/annotations/cleft2pre/tree_geometry.json
    python extract_groundtruth2_cleft2pre.py $vol
    python write_synapsefile.py ${vol}/annotations/cleft2pre/ground_truth.csv ${vol}/annotations/cleft2pre/synapses.hdf
}

mask_offset=2000,1024,1024  # 200612
mask_shape=4000,6144,6144   # 200612

# for i in `seq 1 3`; do
# # for i in 1; do
#     # echo Fetching skeletons...
#     # python fetch_skeletons.py synapse_cutout${i}_skeleton
#     echo Writing cutout${i}/synapse_cleft.hdf...
#     mv synapse_cutout${i}_skeleton.json cutout${i}/tree_geometry_cleft2pre.json
#     python extract_groundtruth2_cleft2pre.py cutout${i}
#     python write_synapsefile.py cutout${i}/ground_truth_cleft2pre.csv cutout${i}/synapses_cleft2pre.hdf
#     echo Updating masks..
#     python add_mask2.py cutout${i}/cb2_synapse_cutout{i}.json ${mask_offset} ${mask_shape} cleft2pre
#     echo
# done

python add_mask2.py cutout1/cb2_synapse_cutout1.json ${mask_offset} ${mask_shape} cleft2pre
update_gt cutout1
python add_mask2.py cutout2/cb2_synapse_cutout2.json ${mask_offset} ${mask_shape} cleft2pre
update_gt cutout2
python add_mask2.py cutout3/cb2_synapse_cutout3.json ${mask_offset} ${mask_shape} cleft2pre
update_gt cutout3

python add_mask2.py pl2/pl2.json ${mask_offset} ${mask_shape} cleft2pre
update_gt pl2

python add_mask2.py ml1/ml1.json ${mask_offset} ${mask_shape} cleft2pre
update_gt ml1

python add_mask2.py ml0/ml0.json ${mask_offset} ${mask_shape} cleft2pre
update_gt ml0


# python add_mask2.py wfly1_gt_cutout0.json 4000,2048,2048 2000,4096,4096 s100_s150_hannah

