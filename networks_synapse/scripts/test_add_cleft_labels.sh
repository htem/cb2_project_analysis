
segway_dir=/n/groups/htem/Segmentation/shared-nondev
if [[ $PYTHONPATH != *${segway_dir}* ]]; then
    echo INFO: PYTHONPATH env does not have segway... adding it
    export PYTHONPATH=$PYTHONPATH:${segway_dir}
fi

# tiff_stack_path="/n/groups/htem/temcagt/datasets/cb2/segmentation/jeff/syn-tifs/cb2_ml0_syn_gt"
tiff_stack_path="/n/groups/htem/temcagt/datasets/cb2/segmentation/jeff/syn-tifs/cb2_ml0_syn_gt/flipped"
output_file='/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_gt/synapse_gt/ml0/ml0.zarr'
output_dataset="volumes/labels/cleft"
voxel_size_zyx='40 4 4'
# roi_offset='2000 2048 2048'
# roi_offset='2000 1392 1392'
roi_offset='3000 1392 1392'
# roi_offset='3000 1400 1400'
# roi_shape='4000 6144 6144'
roi_shape='4000 6016 6016'
y_tile_size=1504
x_tile_size=1504
# section_dir_name_format="cb2_ml0_syn_gt{:d}.tif"
section_dir_name_format="{:d}.0.tif"
# bad_sections='180 202'
# z_numbering_offset=-50
z_numbering_offset=-25

script=/n/groups/htem/Segmentation/shared-nondev/segway/tasks/make_zarr_from_tiff/task_make_zarr_from_tiff_single_file.py

python $script $tiff_stack_path $y_tile_size $x_tile_size $voxel_size_zyx $output_file $output_dataset --roi_offset $roi_offset --roi_shape $roi_shape --section_dir_name_format $section_dir_name_format --single_file_format 1 --z_numbering_offset $z_numbering_offset --no_launch_workers 0 --num_workers 4  --overwrite 2


# convert cleft labels to binary labels
output_dataset="volumes/labels/cleft"
output_dataset_binary="volumes/labels/cleft_255"
python convert_labels_to_255.py $output_file $output_dataset $output_dataset_binary

# add train/test masks
# use 80% of labels for train, 20% for testing
train_mask_offset='3000 1392 1392'
train_mask_shape='4000 4800 6016'
test_mask_offset='3000 6192 1392'
test_mask_shape='4000 1216 6016'
python add_mask3.py $output_file volumes/raw volumes/labels/cleft_mask_train --roi_offset ${train_mask_offset} --roi_shape ${train_mask_shape}
python add_mask3.py $output_file volumes/raw volumes/labels/cleft_mask_test --roi_offset ${test_mask_offset} --roi_shape ${test_mask_shape}


### CUTOUT1 201220

output_dataset="volumes/labels/cleft"
voxel_size_zyx='40 4 4'
# roi_offset='2000 1392 1392'
roi_offset='2000 1100 1100'
roi_shape='4000 6016 6016'
y_tile_size=1504
x_tile_size=1504
section_dir_name_format="{:d}.tif"
z_numbering_offset=-50
script=/n/groups/htem/Segmentation/shared-nondev/segway/tasks/make_zarr_from_tiff/task_make_zarr_from_tiff_single_file.py


vols='cutout1 cutout2 cutout5 cutout6 cutout7 ml1'
vols='ml1 cutout2 cutout5 cutout6 cutout7'
for vol in $vols; do
    tiff_stack_path="/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_gt/synapse_gt/cleft_wkw/${vol}/tiffs"
    output_file="/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_gt/synapse_gt/${vol}/${vol}.zarr"
    echo $vol
    echo $output_file
    python $script $tiff_stack_path $y_tile_size $x_tile_size $voxel_size_zyx $output_file $output_dataset --roi_offset $roi_offset --roi_shape $roi_shape --section_dir_name_format $section_dir_name_format --single_file_format 1 --z_numbering_offset $z_numbering_offset --no_launch_workers 0 --num_workers 1  --overwrite 2
done


vols='ml1 cutout1 cutout2 cutout5 cutout6 cutout7'
for vol in $vols; do
    # convert cleft labels to binary labels
    output_dataset="volumes/labels/cleft"
    output_dataset_binary="volumes/labels/cleft_255"
    output_file="/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_gt/synapse_gt/${vol}/${vol}.zarr"
    python convert_labels_to_255.py $output_file $output_dataset $output_dataset_binary
    # add train/test masks
    # use 80% of labels for train, 20% for testing
    train_mask_offset='2000 1100 1100'
    train_mask_shape='4000 4800 6016'
    test_mask_offset='2000 5900 1100'
    test_mask_shape='4000 1216 6016'
    python add_mask3.py $output_file volumes/raw volumes/labels/cleft_mask_train --roi_offset ${train_mask_offset} --roi_shape ${train_mask_shape}
    python add_mask3.py $output_file volumes/raw volumes/labels/cleft_mask_test --roi_offset ${test_mask_offset} --roi_shape ${test_mask_shape}
done


vols='ml1 cutout1 cutout2 cutout5 cutout6 cutout7'
for vol in $vols; do
    output_file="/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_gt/synapse_gt/${vol}/${vol}.zarr"
    # convert cleft labels to binary labels
    # output_dataset="volumes/labels/cleft"
    # output_dataset_binary="volumes/labels/cleft_255"
    # python convert_labels_to_255.py $output_file $output_dataset $output_dataset_binary
    # add train/test masks
    # use 80% of labels for train, 20% for testing
    train_mask_offset='2000 1100 1100'
    train_mask_shape='4000 4800 6016'
    test_mask_offset='2000 5900 1100'
    test_mask_shape='4000 1216 6016'
    python add_mask3.py $output_file volumes/raw volumes/labels/cleft_mask_train --roi_offset ${train_mask_offset} --roi_shape ${train_mask_shape}
    python add_mask3.py $output_file volumes/raw volumes/labels/cleft_mask_test --roi_offset ${test_mask_offset} --roi_shape ${test_mask_shape}
done


# add dummy masks for nosynapse vols
vols='nosynapse_cutout0 nosynapse_cutout1 nosynapse_cutout2 nosynapse_cutout3 nosynapse_cutout4 nosynapse_cutout5 nosynapse_cutout6'
vols='nosynapse_cutout4'
for vol in $vols; do
    output_file="/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_gt/synapse_gt/${vol}/${vol}.zarr"
    python add_mask3.py $output_file volumes/labels/labels_mask volumes/labels/cleft_255 --write_val 0
    python add_mask3.py $output_file volumes/labels/labels_mask volumes/labels/cleft_mask_train
done

