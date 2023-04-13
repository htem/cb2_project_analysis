
for folder in cutout*; do
   echo Syncing $folder
   rfolder=slowpoke1:/groups/funke/home/nguyent3/julia_synful/01_data/cb2/synapses_v02/$folder/
   rsync -aP $rfolder/*.json $rfolder/*.csv $rfolder/*.hdf $folder
   rsync -aP slowpoke1:/groups/funke/home/nguyent3/julia_synful/01_data/cb2/mask.hdf $folder
   rsync -aP slowpoke1:/groups/funke/home/nguyent3/julia_synful/01_data/cb2/mask_roi.json $folder
done

# for folder in nosynapse_cutout*; do
#    echo Syncing $folder
#    rsync -aP $folder/*.zarr slowpoke1:/groups/funke/home/nguyent3/synapse_gt/cb2/$folder
# done
