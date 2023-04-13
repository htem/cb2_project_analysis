import daisy
import sys
import logging
import numpy as np
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

'''
Changelog:
'''

if __name__ == "__main__":

    config_f = sys.argv[1]
    with open(config_f) as f:
        config = json.load(f)

    if "out_file" not in config:
        # script_name = os.path.basename(config_f)
        script_name = '.'.join(config_f.split(".")[0:-1])
        out_file = script_name + ".zarr"
        config["out_file"] = out_file

    file = config["out_file"]

    raw = daisy.open_ds(file, "volumes/raw")

    roi_offset = sys.argv[2]
    roi_shape = sys.argv[3]

    roi_offset = roi_offset.split(',')
    roi_offset = [int(i) for i in roi_offset]
    roi_shape = roi_shape.split(',')
    roi_shape = [int(i) for i in roi_shape]
    roi = daisy.Roi(roi_offset, roi_shape)

    assert raw.roi.contains(roi)

    postpend = ""
    if len(sys.argv) > 4:
        postpend = sys.argv[4]

    assert raw.voxel_size is not None
    assert roi is not None

    print("Writing volumes/train_masks with ROI ", roi)
    mask_ds = daisy.prepare_ds(
        file,
        "volumes/train_masks/" + postpend,
        roi,
        raw.voxel_size,
        np.uint8,
        compressor={'id': 'zlib', 'level': 5},
        delete=True
        )
    mask_ds[mask_ds.roi] = 1

    # creating dummy synapse_mask.json
    offset = daisy.Coordinate([500, 512, 512])
    roi = raw.roi.grow(-offset, -offset)
    out_path = os.path.join(file, "annotations", postpend)
    os.makedirs(out_path, exist_ok=True)
    json_f = os.path.join(out_path, "synapse_mask.json")
    print("Writing synapse_mask.json with ROI %s to %s" % (roi, json_f))
    with open(json_f, 'w') as f:
        json.dump({
            'offset': roi.get_offset(),
            'size': roi.get_shape()
        }, f)
