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

    roi = raw.roi
    try:
    	roi_context = daisy.Coordinate(tuple(config["CatmaidIn"]["synapse_roi_context_nm"]))
    except:
    	roi_context = daisy.Coordinate(tuple(config["CatmaidIn"]["roi_context_nm"]))
    roi = roi.grow(-roi_context, -roi_context)

    print("Writing volumes/segmentation with ROI ", roi)
    mask_ds = daisy.prepare_ds(
        file,
        "volumes/segmentation",
        roi,
        raw.voxel_size,
        np.uint64,
        compressor={'id': 'zlib', 'level': 5},
        # delete=True
        )
