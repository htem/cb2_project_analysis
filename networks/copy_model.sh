
oldsetup=$1
newsetup=$2
it=$3

mkdir -p $newsetup

cp $oldsetup/*.py $oldsetup/*.sh $oldsetup/*.json $oldsetup/*.meta $newsetup
rm $newsetup/unet_checkpoint*.meta

if [ ! -z "$it" ]; then
   cp $oldsetup/unet_checkpoint_$it.meta $oldsetup/unet_checkpoint_$it.index $oldsetup/unet_checkpoint_$it.data-00000-of-00001 $newsetup
   rm $newsetup/checkpoint
   echo "model_checkpoint_path: \"unet_checkpoint_$it\"" >> $newsetup/checkpoint
   echo "all_model_checkpoint_paths: \"unet_checkpoint_$it\"" >> $newsetup/checkpoint
fi

