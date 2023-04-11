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

class ACRLSDModel(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up,
            constant_upsample=True):

        super().__init__()

        self.unet = UNet(
                in_channels=in_channels,
                num_fmaps=num_fmaps,
                fmap_inc_factor=fmap_inc_factor,
                downsample_factors=downsample_factors,
                kernel_size_down=kernel_size_down,
                kernel_size_up=kernel_size_up,
                constant_upsample=constant_upsample)

        self.affs_head = ConvPass(num_fmaps,3,[[1,1,1]],activation='Sigmoid')

    def forward(self,raw,lsds):

        z = self.unet(torch.cat((raw,lsds),dim=1))

        affs = self.affs_head(z)

        return affs


def predict(
    raw_file,
    raw_dataset,
    lsd_file,
    lsd_dataset,
    checkpoint,
    out_file,
    out_dataset):

    downsample = 2

    raw_fr = ArrayKey('RAW_FR')

    if downsample > 1:

        raw = ArrayKey('RAW')

    else:

        raw = ArrayKey('RAW_FR')

    pred_lsds = ArrayKey('PRED_LSDS')
    pred_affs = ArrayKey('PRED_AFFS')

    input_shape = Coordinate((40,208,208))
    output_shape = Coordinate((20,84,84))

    voxel_size = Coordinate((40,4*downsample,4*downsample))

    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    context = (input_size - output_size) / 2

    scan_request = BatchRequest()

    scan_request.add(raw, input_size)
    scan_request.add(pred_lsds, input_size)
    scan_request.add(pred_affs, output_size)

    in_channels = 11
    num_fmaps = 12
    fmap_inc_factor = 5
    downsample_factors = [(1,2,2),(1,2,2),(2,3,3)]

    kernel_size_down = [
                [(3,)*3, (3,)*3],
                [(3,)*3, (3,)*3],
                [(3,)*3, (3,)*3],
                [(1,3,3), (1,3,3)]]

    kernel_size_up = [
                [(1,3,3), (1,3,3)],
                [(3,)*3, (3,)*3],
                [(3,)*3, (3,)*3]]

    model = ACRLSDModel(
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up,
            constant_upsample=True)

    model.eval()

    raw_source = ZarrSource(
                raw_file,
                    {
                        raw_fr: raw_dataset
                    },
                    {
                        raw_fr: ArraySpec(interpolatable=True)
                    })

    lsd_source = ZarrSource(
                lsd_file,
                    {
                        pred_lsds: lsd_dataset
                    },
                    {
                        pred_lsds: ArraySpec(interpolatable=True)
                    }
            )

    with build(lsd_source):
        total_input_roi = lsd_source.spec[pred_lsds].roi
        total_output_roi = total_input_roi.grow(-context, -context)

    daisy.prepare_ds(
            out_file,
            out_dataset,
            daisy.Roi(
                total_output_roi.get_offset(),
                total_output_roi.get_shape()
            ),
            voxel_size,
            write_size=output_size,
            num_channels=3,
            dtype=np.uint8)

    source = (
            raw_source +
            Normalize(raw_fr) +
            Pad(raw_fr, size=None) +
            DownSample(raw_fr, (1, downsample, downsample), raw) +
            IntensityScaleShift(raw, 2, -1),

            lsd_source +
            Normalize(pred_lsds) +
            Pad(pred_lsds, size=None)
        )

    predict = Predict(
        model=model,
        checkpoint=checkpoint,
        inputs = {
            'raw': raw,
            'lsds': pred_lsds
        },
        outputs = {
            0: pred_affs
        })

    scan = Scan(scan_request)

    write = ZarrWrite(
        dataset_names={
            pred_affs: out_dataset
            },
        output_filename=out_file)

    pipeline = source
    pipeline += MergeProvider()

    pipeline += Unsqueeze([raw])
    pipeline += Stack(1)

    pipeline += predict

    pipeline += IntensityScaleShift(pred_affs, 255, 0)

    pipeline += Squeeze([pred_affs])

    pipeline += write

    pipeline += scan

    predict_request = BatchRequest()

    predict_request.add(raw, total_input_roi.get_end())
    predict_request.add(pred_lsds, total_input_roi.get_end())
    predict_request.add(pred_affs, total_output_roi.get_end())

    with build(pipeline):
        pipeline.request_batch(predict_request)


if __name__ == '__main__':

    raw_file = sys.argv[1]
    raw_dataset = sys.argv[2]
    lsd_file = sys.argv[3]
    lsd_dataset = sys.argv[4]
    checkpoint = 'model_checkpoint_5000'
    out_file = 'test.zarr'
    out_dataset = 'pred_affs'

    predict(
            raw_file,
            raw_dataset,
            lsd_file,
            lsd_dataset,
            checkpoint,
            out_file,
            out_dataset)
