
# echo Fetching skeletons...
# python fetch_skeletons.py
skeleton_path=/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/evals_synapse/skeletons

function update_gt {
    vol=$1
    proj=$2
    mkdir -p ${vol}/annotations/${proj}
    cp ${skeleton_path}/${vol}_skeleton.json ${vol}/annotations/${proj}/tree_geometry.json
    python extract_groundtruth3.py $vol $proj
    python write_synapsefile.py ${vol}/annotations/${proj}/ground_truth.csv ${vol}/annotations/${proj}/synapses.hdf
}

mask_offset=2000,1024,1024  # 200612
mask_shape=4000,6144,6144   # 200612

mask_4x4x4_offset=2000,2048,2048
mask_4x4x4_shape=4000,4096,4096

mask_6x6x6_offset=1000,1024,1024
mask_6x6x6_shape=6000,6144,6144

python add_mask2.py cutout1/cb2_synapse_cutout1.json ${mask_offset} ${mask_shape} 200619
update_gt cutout1 200619
python add_mask2.py cutout2/cb2_synapse_cutout2.json ${mask_offset} ${mask_shape} 200619
update_gt cutout2 200619
python add_mask2.py cutout3/cb2_synapse_cutout3.json ${mask_offset} ${mask_shape} 200619
update_gt cutout3 200619

python add_mask2.py pl2/pl2.json ${mask_offset} ${mask_shape} 200619
update_gt pl2 200619

python add_mask2.py ml1/ml1.json ${mask_offset} ${mask_shape} 200619
update_gt ml1 200619

python add_mask2.py ml0/ml0.json ${mask_offset} ${mask_shape} 200619
update_gt ml0 200619


# python add_mask2.py wfly1_gt_cutout0.json 4000,2048,2048 2000,4096,4096 s100_s150_hannah

