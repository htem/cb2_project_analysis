import daisy
import neuroglancer
import sys

sys.path.insert(0, '/n/groups/htem/Segmentation/tmn7/funlib.show.neuroglancer_v2')
from funlib.show.neuroglancer import add_layer
from funlib.show.neuroglancer.contrib import make_viewer

f = sys.argv[1]
raw = daisy.open_ds(f, 'raw')
labels = daisy.open_ds(f, 'labels')
gt_affs = daisy.open_ds(f, 'gt_affs')
affs = daisy.open_ds(f, 'pred_affs')
labels_mask = daisy.open_ds(f, 'labels_mask')
unlabeled_mask = daisy.open_ds(f, 'unlabeled_mask')
try:
    affs_gradient = daisy.open_ds(f, 'affs_gradient')
except:
    pass

viewer = make_viewer(port_range=(33400, 33500), token='snapshot')

with viewer.txn() as s:

    add_layer(s, raw, 'raw')
    add_layer(s, labels, 'labels')
    add_layer(s, gt_affs, 'gt_affs', shader='rgb', visible=False)
    add_layer(s, affs, 'pred_affs', shader='rgb', visible=True)
    try:
        add_layer(s, affs_gradient, 'affs_gradient', shader='rgb', visible=False)
    except:
        pass
    add_layer(s, labels_mask, 'labels_mask', shader='mask', visible=False)
    add_layer(s, unlabeled_mask, 'unlabeled_mask', shader='mask', visible=False)

    s.layout = 'xy'
    # s.navigation.zoomFactor=1.5
    s.projectionScale = 256
    s.crossSectionScale = .15

print(viewer)
link = str(viewer)
print(link.replace("gandalf", "catmaid3.hms.harvard.edu"))
print(link.replace("catmaid2", "catmaid2.hms.harvard.edu"))
print(link.replace("lee-htem-gpu0", "10.11.144.169"))
print(link.replace("lee-lab-gpu1", "10.11.144.167"))
