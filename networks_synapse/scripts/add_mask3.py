import daisy
import sys
import logging
import numpy as np
import json
import os
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

'''
Changelog:
'''

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("out_file", type=str, help='Input hdf/zarr volume')
    ap.add_argument("ref_ds", type=str, help='')
    ap.add_argument("out_ds", type=str, help='')
    ap.add_argument(
        "--roi_offset", type=int, help='',
        nargs='+', default=None)
    ap.add_argument(
        "--roi_shape", type=int, help='',
        nargs='+', default=None)
    ap.add_argument(
        "--write_val", type=int, help='', default=1)
    config = ap.parse_args()

    roi_offset = config.roi_offset
    roi_shape = config.roi_shape

    ref_ds = daisy.open_ds(config.out_file, config.ref_ds)

    if config.roi_shape is not None or config.roi_offset is not None:
        assert roi_shape is not None and roi_offset is not None
        roi = daisy.Roi(config.roi_offset, config.roi_shape)
    else:
        roi = ref_ds.roi

    assert ref_ds.roi.contains(roi)
    assert ref_ds.voxel_size is not None
    assert roi is not None

    print(f"Writing {config.out_ds} with ROI ", roi)
    mask_ds = daisy.prepare_ds(
        config.out_file,
        config.out_ds,
        roi,
        ref_ds.voxel_size,
        np.uint8,
        compressor={'id': 'zlib', 'level': 3},
        delete=True
        )
    mask_ds[mask_ds.roi] = config.write_val
