
for folder in cutout*; do
   echo Syncing $folder
   rsync -aP $folder/*.zarr $folder/ground_truth.csv $folder/*.json slowpoke1:/groups/funke/home/nguyent3/synapse_gt/cb2/$folder
done

for folder in nosynapse_cutout*; do
   echo Syncing $folder
   rsync -aP $folder/*.zarr slowpoke1:/groups/funke/home/nguyent3/synapse_gt/cb2/$folder
done
