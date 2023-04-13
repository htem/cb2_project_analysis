
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
function update_gt_cleft {
    vol=$1
    proj=$2
    mkdir -p ${vol}/annotations/${proj}
    cp ${skeleton_path}/${vol}_skeleton.json ${vol}/annotations/${proj}/tree_geometry.json
    python extract_groundtruth3_cleft2pre.py $vol $proj
    python write_synapsefile.py ${vol}/annotations/${proj}/ground_truth.csv ${vol}/annotations/${proj}/synapses.hdf
}

mask_4x4x4_offset=2000,2048,2048
mask_4x4x4_shape=4000,4096,4096
mask_6x6x6_offset=1000,1024,1024
mask_6x6x6_shape=6000,6144,6144

# for i in 1 2 3 4 5 7; do

#     python add_mask2.py cutout${i}/cb2_synapse_cutout${i}.json ${mask_6x6x6_offset} ${mask_6x6x6_shape} 200816
#     update_gt cutout${i} 200816

#     python add_mask2.py cutout${i}/cb2_synapse_cutout${i}.json ${mask_6x6x6_offset} ${mask_6x6x6_shape} cleft2pre_200816
#     update_gt_cleft cutout${i} cleft2pre_200816
# done

for i in ml0 ml1 pl2; do
    python add_mask2.py ${i}/${i}.json ${mask_6x6x6_offset} ${mask_6x6x6_shape} 200816
    update_gt ${i} 200816
    python add_mask2.py ${i}/${i}.json ${mask_6x6x6_offset} ${mask_6x6x6_shape} cleft2pre_200816
    update_gt_cleft ${i} cleft2pre_200816
done


# python add_mask2.py wfly1_gt_cutout0.json 4000,2048,2048 2000,4096,4096 s100_s150_hannah

