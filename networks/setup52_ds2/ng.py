import daisy
import neuroglancer
import sys
# import numpy as np

# forwarded_port = 7777

neuroglancer.set_server_bind_address('0.0.0.0')

#path to raw
f = sys.argv[1]

#raw key
raw = daisy.open_ds(f, 'volumes/raw')

labels = daisy.open_ds(f, 'volumes/labels/neuron_ids')
gt_myelin_embedding = daisy.open_ds(f, 'volumes/gt_myelin_embedding')
myelin_embedding = daisy.open_ds(f, 'volumes/pred_myelin_embedding')
gt_affs = daisy.open_ds(f, 'volumes/gt_affinities')
affs = daisy.open_ds(f, 'volumes/pred_affinities')
labels_mask = daisy.open_ds(f, 'volumes/labels/mask')
unlabeled_mask = daisy.open_ds(f, 'volumes/labels/unlabeled')
affs_gradient = daisy.open_ds(f, 'volumes/affs_gradient')

#path to labels
# f='output.zarr'


def add(s, a, name, shader=None):

    if shader == 'rgb':
        shader="""void main() { emitRGB(vec3(toNormalized(getDataValue(0)), toNormalized(getDataValue(1)), toNormalized(getDataValue(2)))); }"""

    elif shader == '255':
        shader="""void main() { emitGrayscale(float(getDataValue().value)); }"""

    kwargs = {}
    if shader is not None:
        kwargs['shader'] = shader

    s.layers.append(
            name=name,
            layer=neuroglancer.LocalVolume(
                data=a.data,
                offset=a.roi.get_offset()[::-1],
                voxel_size=a.voxel_size[::-1]
            ),
            **kwargs)
    print(s.layers)

viewer = neuroglancer.Viewer()

with viewer.txn() as s:

    f, prepend = (f, "test")
    #add(s, daisy.open_ds(f, 'volumes/sparse_segmentation_0.5'), '%s_seg'%prepend)
    #add(s, daisy.open_ds(f, 'volumes/affs'), '%s_aff'%prepend)
    #add(s, daisy.open_ds(f, 'volumes/fragments'), '%s_frag'%prepend)
    #add(s, daisy.open_ds(f, 'volumes/segmentation_0.100'), '%s_seg_100'%prepend)
    #add(s, daisy.open_ds(f, 'volumes/segmentation_0.200'), '%s_seg_200'%prepend)
    #add(s, daisy.open_ds(f, 'volumes/segmentation_0.300'), '%s_seg_300'%prepend)
    #add(s, daisy.open_ds(f, 'volumes/segmentation_0.400'), '%s_seg_400'%prepend)
    # add(s, daisy.open_ds(f, 'volumes/sparse_segmentation_0.5'), '%s_seg_500'%prepend)

    add(s, raw, 'raw')
    add(s, labels, 'labels')
    add(s, gt_myelin_embedding, 'gt_myelin')
    add(s, myelin_embedding, 'pred_myelin')
    add(s, gt_affs, 'gt_affs', shader='rgb')
    add(s, affs, 'pred_affs', shader='rgb')
    add(s, affs_gradient, 'affs_gradient', shader='rgb')
    add(s, labels_mask, 'labels_mask', shader='rgb')
    add(s, unlabeled_mask, 'unlabeled_mask', shader='rgb')
    
print(viewer)
link = str(viewer)
print(link.replace("gandalf", "catmaid3.hms.harvard.edu"))
