import daisy
import sys
import logging
import numpy as np
# import gt_tools
import argparse

from funlib.segment.arrays import replace_values

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

'''
Changelog:
5/24/19:
- 

'''


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("out_file", type=str, help='Input hdf/zarr volume')
    ap.add_argument("in_ds", type=str, help='')
    ap.add_argument("out_ds", type=str, help='')
    # ap.add_argument("output_dir", type=str, help='Directory to save meshes to')
    # ap.add_argument(
    #     "--block_size", type=int, help='zyx in nm, should align to the superfragment block size',
    #     nargs='+')
    # ap.add_argument(
    #     "--roi_offset", type=int, help='',
    #     nargs='+', default=None)
    # ap.add_argument(
    #     "--roi_shape", type=int, help='',
    #     nargs='+', default=None)
    # ap.add_argument(
    #     "--context", type=int, help='',
    #     nargs='+', default=[0, 0, 0])
    # # ap.add_argument(
    # #     "--downsample", type=int, help='',
    # #     default=1)
    # ap.add_argument(
    #     "--downsample", type=int, help='',
    #     nargs='+', default=None)
    # ap.add_argument(
    #     "--hierarchical_path_size", type=int, help='',
    #     default=10000)


    # arg = sys.argv[1]

    # if arg.endswith(".zarr"):
    #     out_file = arg

    # else:
    #     config = gt_tools.load_config(sys.argv[1])
    #     out_file = config["out_file"]
    #     print(out_file)

    config = ap.parse_args()

    label_ds = daisy.open_ds(config.out_file, config.in_ds)

    out_ds = daisy.prepare_ds(
        config.out_file,
        config.out_ds,
        label_ds.roi,
        label_ds.voxel_size,
        np.uint8,
        compressor={'id': 'zlib', 'level': 3},
        delete=True
        )

    out_ndarray = np.ones(out_ds.shape, dtype=np.uint8)
    # out_ndarray *= 255

    labels_ndarray = label_ds.to_ndarray()
    segment_by_foreground = [0]
    new_mask_values = [0]
    replace_values(
        labels_ndarray,
        segment_by_foreground,
        new_mask_values,
        out_ndarray)

    out_ds[out_ds.roi] = out_ndarray
