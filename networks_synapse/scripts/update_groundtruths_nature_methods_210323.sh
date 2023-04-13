
# 201212: some proofreading changes from 200816 set
# is done to train on all except for cutout9 and ml0

skeleton_path=/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/evals_synapse/skeletons


function update_gt {
    vol=$1
    proj=$2
    mkdir -p ${vol}/annotations/${proj}
    cp ${skeleton_path}/${vol}_skeleton.json ${vol}/annotations/${proj}/tree_geometry.json
    python extract_groundtruth3_cleft2pre.py $vol $proj
    mv ${vol}/annotations/${proj}/ground_truth.csv ${vol}/annotations/${proj}/ground_truth_cleft2pre.csv
    python extract_groundtruth3.py $vol $proj
    mv ${vol}/annotations/${proj}/ground_truth.csv ${vol}/annotations/${proj}/ground_truth_pre2post.csv
    python extract_groundtruth4.py $vol $proj
    mv ${vol}/annotations/${proj}/ground_truth.csv ${vol}/annotations/${proj}/ground_truth_all.csv
    python write_synapsefile.py ${vol}/annotations/${proj}/ground_truth_pre2post.csv ${vol}/annotations/${proj}/synapses_pre2post.hdf
    python write_synapsefile.py ${vol}/annotations/${proj}/ground_truth_cleft2pre.csv ${vol}/annotations/${proj}/synapses_cleft2pre.hdf
    cp ${vol}/${vol}.zarr/annotations/${proj}/synapse_mask.json ${vol}/annotations/${proj}
    rm ${vol}/annotations/${proj}/tree_geometry.json
}

mask_4x4x4_offset=2000,2048,2048
mask_4x4x4_shape=4000,4096,4096
mask_6x6x6_offset=1000,1024,1024
mask_6x6x6_shape=6000,6144,6144

python add_mask2.py cutout1/cb2_synapse_cutout1.json ${mask_6x6x6_offset} ${mask_6x6x6_shape} nature_methods_210323
update_gt cutout1 nature_methods_210323
python add_mask2.py cutout2/cb2_synapse_cutout2.json ${mask_6x6x6_offset} ${mask_6x6x6_shape} nature_methods_210323
update_gt cutout2 nature_methods_210323
python add_mask2.py cutout3/cb2_synapse_cutout3.json ${mask_6x6x6_offset} ${mask_6x6x6_shape} nature_methods_210323
update_gt cutout3 nature_methods_210323
python add_mask2.py cutout4/cb2_synapse_cutout4.json ${mask_6x6x6_offset} ${mask_6x6x6_shape} nature_methods_210323
update_gt cutout4 nature_methods_210323
python add_mask2.py cutout5/cb2_synapse_cutout5.json ${mask_6x6x6_offset} ${mask_6x6x6_shape} nature_methods_210323
update_gt cutout5 nature_methods_210323
python add_mask2.py cutout6/cb2_synapse_cutout6.json ${mask_6x6x6_offset} ${mask_6x6x6_shape} nature_methods_210323
update_gt cutout6 nature_methods_210323
python add_mask2.py cutout7/cb2_synapse_cutout7.json ${mask_6x6x6_offset} ${mask_6x6x6_shape} nature_methods_210323
update_gt cutout7 nature_methods_210323
python add_mask2.py cutout8/cb2_synapse_cutout8.json ${mask_6x6x6_offset} ${mask_6x6x6_shape} nature_methods_210323
update_gt cutout8 nature_methods_210323
python add_mask2.py cutout9/cb2_synapse_cutout9.json ${mask_6x6x6_offset} ${mask_6x6x6_shape} nature_methods_210323
update_gt cutout9 nature_methods_210323

python add_mask2.py pl2/pl2.json ${mask_6x6x6_offset} ${mask_6x6x6_shape} nature_methods_210323
update_gt pl2 nature_methods_210323
python add_mask2.py ml1/ml1.json ${mask_6x6x6_offset} ${mask_6x6x6_shape} nature_methods_210323
update_gt ml1 nature_methods_210323
python add_mask2.py ml0/ml0.json ${mask_6x6x6_offset} ${mask_6x6x6_shape} nature_methods_210323
update_gt ml0 nature_methods_210323

# python add_mask2.py cutout1/cb2_synapse_cutout1.json ${mask_offset} ${mask_shape} nature_methods_210323
# update_gt cutout1
# python add_mask2.py cutout2/cb2_synapse_cutout2.json ${mask_offset} ${mask_shape} nature_methods_210323
# update_gt cutout2
# python add_mask2.py cutout3/cb2_synapse_cutout3.json ${mask_offset} ${mask_shape} nature_methods_210323
# update_gt cutout3

# python add_mask2.py pl2/pl2.json ${mask_offset} ${mask_shape} nature_methods_210323
# update_gt pl2

# python add_mask2.py ml1/ml1.json ${mask_offset} ${mask_shape} nature_methods_210323
# update_gt ml1

# python add_mask2.py ml0/ml0.json ${mask_offset} ${mask_shape} nature_methods_210323
# update_gt ml0


# python add_mask2.py wfly1_gt_cutout0.json 4000,2048,2048 2000,4096,4096 s100_s150_hannah

