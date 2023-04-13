# for folder in nosynapse_cutout*/*.json; do
#    echo Adding mask to $folder
#    python add_dummy_seg.py $folder
#    # rsync -aP $folder/* slowpoke1:/groups/funke/home/nguyent3/synapse_gt/vnc/$folder
# done

for json in nosynapse_cutout*/*cutout*.json; do
   echo Adding mask to $json
   python add_dummy_seg.py $json
done

# for folder in nosynapse_cutout*; do
# 	cp nosyn_cutout.hdf $folder/synapses.hdf
# done
