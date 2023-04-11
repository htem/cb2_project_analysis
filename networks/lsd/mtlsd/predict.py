import daisy
import glob
import logging
import numpy as np
import os
import random
import sys
import torch
import zarr

from gunpowder import *
from gunpowder.ext import torch
from gunpowder.torch import *
from funlib.learn.torch.models import UNet, ConvPass

logging.basicConfig(level=logging.INFO)

class MtlsdModel(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up,
            num_fmaps_out,
            constant_upsample=True):

        super().__init__()

        self.unet = UNet(
                in_channels=in_channels,
                num_fmaps=num_fmaps,
                fmap_inc_factor=fmap_inc_factor,
                downsample_factors=downsample_factors,
                kernel_size_down=kernel_size_down,
                kernel_size_up=kernel_size_up,
                num_fmaps_out=num_fmaps_out,
                constant_upsample=constant_upsample)

        self.lsd_head = ConvPass(num_fmaps_out,10,[[1,1,1]],activation='Sigmoid')
        self.aff_head = ConvPass(num_fmaps_out,3,[[1,1,1]],activation='Sigmoid')

    def forward(self,input):

        z = self.unet(input)
        lsds = self.lsd_head(z)
        affs = self.aff_head(z)

        return lsds,affs

def predict(
    checkpoint,
    raw_file,
    raw_dataset,
    out_file,
    out_datasets):

    downsample = 2

    raw_fr = ArrayKey('RAW_FR')

    if downsample > 1:
        raw = ArrayKey('RAW')

    else:
        raw = ArrayKey('RAW_FR')

    pred_lsds = ArrayKey('PRED_LSDS')
    pred_affs = ArrayKey('PRED_AFFS')

    input_shape = Coordinate((40,196,196))
    output_shape = Coordinate((20,104,104))

    voxel_size = Coordinate((40,4*downsample,4*downsample))
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    context = (input_size - output_size) / 2

    scan_request = BatchRequest()

    scan_request.add(raw, input_size)
    scan_request.add(pred_lsds, output_size)
    scan_request.add(pred_affs, output_size)

    in_channels = 1
    num_fmaps = 12
    fmap_inc_factor = 5
    downsample_factors = [(1,2,2),(1,2,2),(2,2,2)]

    kernel_size_down = [
                [(3,)*3, (3,)*3],
                [(3,)*3, (3,)*3],
                [(3,)*3, (3,)*3],
                [(1,3,3), (1,3,3)]]

    kernel_size_up = [
                [(1,3,3), (1,3,3)],
                [(3,)*3, (3,)*3],
                [(3,)*3, (3,)*3]]

    model = MtlsdModel(
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up,
            num_fmaps_out=14)

    source = ZarrSource(
        raw_file,
            {
                raw_fr: raw_dataset
            },
            {
                raw_fr: ArraySpec(interpolatable=True)
            }
        )

    with build(source):
        total_input_roi = source.spec[raw_fr].roi
        total_output_roi = total_input_roi.grow(-context, -context)

    for ds in out_datasets:

        dims = 10 if 'lsds' in ds else 3

        daisy.prepare_ds(
                out_file,
                ds,
                daisy.Roi(
                    total_output_roi.get_offset(),
                    total_output_roi.get_shape()
                ),
                voxel_size,
                write_size=output_size,
                num_channels=dims,
                dtype=np.uint8)

    model.eval()

    predict = Predict(
        model=model,
        checkpoint=checkpoint,
        inputs = {
            'input': raw
        },
        outputs = {
            0: pred_lsds,
            1: pred_affs
        })

    scan = Scan(scan_request)

    write = ZarrWrite(
        dataset_names={
            pred_lsds: out_datasets[0],
            pred_affs: out_datasets[1]
            },
        output_filename=out_file)

    pipeline = source
    pipeline += Normalize(raw_fr)
    pipeline += Pad(raw_fr, None)

    pipeline += DownSample(raw_fr, (1, downsample, downsample), raw)

    pipeline += IntensityScaleShift(raw, 2,-1)

    pipeline += Unsqueeze([raw])
    pipeline += Stack(1)
    pipeline += predict

    pipeline += IntensityScaleShift(pred_lsds, 255, 0)
    pipeline += IntensityScaleShift(pred_affs, 255, 0)

    pipeline += Squeeze([raw])

    pipeline += Squeeze([raw, pred_lsds, pred_affs])

    pipeline += write

    pipeline += scan

    predict_request = BatchRequest()

    predict_request.add(raw, total_input_roi.get_end())
    predict_request.add(pred_lsds, total_output_roi.get_end())
    predict_request.add(pred_affs, total_output_roi.get_end())

    with build(pipeline):
        pipeline.request_batch(predict_request)

if __name__ == '__main__':

    checkpoint = 'model_checkpoint_40000'
    raw_file = sys.argv[1]
    raw_dataset = sys.argv[2]
    out_file = 'test.zarr'
    out_datasets = ['pred_lsds', 'pred_affs']

    predict(
            checkpoint,
            raw_file,
            raw_dataset,
            out_file,
            out_datasets)
