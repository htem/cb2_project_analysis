
# echo Fetching skeletons...
# python fetch_skeletons.py

for i in `seq 1 9`; do
# for i in 1; do
    echo Fetching skeletons...
    python fetch_skeletons.py synapse_cutout${i}_skeleton
    echo Writing cutout${i}/synapse.hdf...
    mv synapse_cutout${i}_skeleton.json cutout${i}/tree_geometry.json
    python extract_groundtruth2.py cutout${i}
    python write_synapsefile.py cutout${i}/ground_truth.csv cutout${i}/synapses.hdf
    echo
done

skeleton_path=/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/evals_synapse/skeletons

function update_gt {

    vol=$1
    cp ${skeleton_path}/${vol}_skeleton.json ${vol}/tree_geometry.json
    python extract_groundtruth2.py $vol
    python write_synapsefile.py ${vol}/ground_truth.csv ${vol}/synapses.hdf
 
}

update_gt pl2

python add_mask.py ml1/ml1.json
update_gt ml1

python add_mask.py ml0/ml0.json
update_gt ml0
