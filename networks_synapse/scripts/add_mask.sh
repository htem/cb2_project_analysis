# for folder in nosynapse_cutout*/*.json; do
#    echo Adding mask to $folder
#    python add_mask.py $folder
#    # rsync -aP $folder/* slowpoke1:/groups/funke/home/nguyent3/synapse_gt/vnc/$folder
# done

for json in cutout*/*cutout*.json; do
   echo Adding mask to $json
   python add_mask.py $json
done

# for folder in nosynapse_cutout*; do
# 	cp nosyn_cutout.hdf $folder/synapses.hdf
# done

python add_mask.py pl2/cb2_gt_pl2_v2.json


