import daisy
import neuroglancer
import sys
from funlib.show.neuroglancer import add_layer, ScalePyramid
# import numpy as np

# forwarded_port = 7777

neuroglancer.set_server_bind_address('0.0.0.0')

#path to raw
f = sys.argv[1]

#raw key
raw = daisy.open_ds(f, 'volumes/raw')

labels = daisy.open_ds(f, 'volumes/labels/neuron_ids')
gt_myelin = daisy.open_ds(f, 'volumes/gt_myelin')
myelin = daisy.open_ds(f, 'volumes/pred_myelin')
gt_embedding = daisy.open_ds(f, 'volumes/labels/gt_embedding')
affs = daisy.open_ds(f, 'volumes/pred_affinities')
labels_mask = daisy.open_ds(f, 'volumes/labels/mask')
unlabeled_mask = daisy.open_ds(f, 'volumes/labels/unlabeled')
affs_gradient = daisy.open_ds(f, 'volumes/affs_gradient')
gt_affinities = daisy.open_ds(f, 'volumes/gt_affinities')

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

    add_layer(s, raw, 'raw')
    add_layer(s, gt_affinities, 'gt_affinities', shader='rgb', visible=False)
    add_layer(s, gt_embedding, 'gt_embedding', shader='rgb', visible=False)
    add_layer(s, affs, 'pred_affs', shader='rgb', visible=True)
    add_layer(s, labels, 'labels', visible=False)
    add_layer(s, affs_gradient, 'affs_gradient', shader='rgb', visible=False)
    add_layer(s, labels_mask, 'labels_mask', shader='mask', visible=False)
    add_layer(s, unlabeled_mask, 'unlabeled_mask', shader='mask', visible=False)
    add_layer(s, myelin, 'pred_myelin', visible=False)
    add_layer(s, gt_myelin, 'gt_myelin', visible=False)
    
print(viewer)
link = str(viewer)
print(link.replace("gandalf", "catmaid3.hms.harvard.edu"))
print(link.replace("lee-htem-gpu0", "10.11.144.169"))
print(link.replace("lee-lab-gpu1", "10.11.144.167"))

