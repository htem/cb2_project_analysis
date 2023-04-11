import json
import logging
import math
import numpy as np
import os
import sys
import torch

from funlib.learn.torch.models import UNet,ConvPass
from gunpowder import *
from gunpowder.ext import torch
from gunpowder.torch import *

sys.path.insert(0, '/n/groups/htem/Segmentation/networks/cb2_setups/lsd')

from duplicate_augment import DuplicateAugment
from add_affinities import AddAffinities
from defect_augment import DefectAugment
import make_sources
from make_sources import raw_fr, labels_fr, labels_mask_fr, unlabeled_mask_fr, raw, labels, labels_mask, unlabeled_mask

logging.basicConfig(level=logging.INFO)

torch.backends.cudnn.benchmark = True

# data_dir = '../../01_data'

# samples = [
#     'gl0_setup45.zarr',
# ]

batch_size = 1

neighborhood = np.array([
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
    [-2, 0, 0],
    [0, -3, 0],
    [0, 0, -3],
    [-3, 0, 0],
    [0, -9, 0],
    [0, 0, -9],
    [-4, 0, 0],
    [0, -27, 0],
    [0, 0, -27]
])

def calc_max_padding(
        output_size,
        voxel_size,
        neighborhood=None,
        sigma=None,
        mode='shrink'):

    if neighborhood is not None:

        if len(neighborhood) > 3:
            neighborhood = neighborhood[9:12]

        max_affinity = Coordinate(
                [np.abs(aff) for val in neighborhood \
                        for aff in val if aff != 0])

        method_padding = voxel_size * max_affinity

    if sigma:

        method_padding = Coordinate((sigma*3,)*3)

    diag = np.sqrt(output_size[1]**2 + output_size[2]**2)

    max_padding = Roi(
            (Coordinate(
                [i/2 for i in [output_size[0], diag, diag]]) +
                method_padding),
            (0,)*3).snap_to_grid(voxel_size,mode=mode)

    return max_padding.get_begin()


class WeightedMSELoss(torch.nn.MSELoss):

    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, affs_prediction, affs_target, affs_weights):

        scaled = (affs_weights * (affs_prediction - affs_target) ** 2)

        if len(torch.nonzero(scaled)) != 0:
            mask = torch.masked_select(scaled, torch.gt(affs_weights, 0))
            loss = torch.mean(mask)

        else:
            loss = torch.mean(scaled)

        return loss


def mknet():

    in_channels = 1
    num_fmaps = 12
    fmap_inc_factor = 5
    downsample_factors = [(1,2,2),(1,2,2),(2,2,2)]

    unet = UNet(
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            constant_upsample=True)

    model = torch.nn.Sequential(
            unet,
            ConvPass(num_fmaps, 12, [[1]*3], activation='Sigmoid'))

    return model

def train_until(max_iteration):

    xy_downsample = 2

    model = mknet()

    loss = WeightedMSELoss()

    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.5e-4,
            betas=(0.95,0.999))

    # raw_fr = ArrayKey('RAW_FR')
    # labels_fr = ArrayKey('GT_LABELS_FR')
    # labels_mask_fr = ArrayKey('LABELS_MASK_FR')
    # unlabeled_mask_fr = ArrayKey('unlabeled_mask_FR')

    # if xy_downsample > 1:

    #     raw = ArrayKey('RAW')
    #     labels = ArrayKey('GT_LABELS')
    #     labels_mask = ArrayKey('LABELS_MASK')
    #     unlabeled_mask = ArrayKey('unlabeled_mask')

    # else:

    #     raw = ArrayKey('RAW_FR')
    #     labels = ArrayKey('GT_LABELS_FR')
    #     labels_mask = ArrayKey('LABELS_MASK_FR')
    #     unlabeled_mask = ArrayKey('unlabeled_mask_FR')

    gt_affs = ArrayKey('GT_AFFS')
    affs_weights = ArrayKey('AFFS_WEIGHTS')
    affs_mask = ArrayKey('GT_AFFINITIES_MASK')
    pred_affs = ArrayKey('PRED_AFFS')

    input_shape = Coordinate((52,196,196))
    output_shape = Coordinate((20,104,104))

    voxel_size = Coordinate((40,4*xy_downsample,4*xy_downsample))
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    labels_padding = calc_max_padding(
            output_size,
            voxel_size,
            neighborhood=neighborhood)

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(unlabeled_mask, output_size)
    request.add(gt_affs, output_size)
    request.add(affs_weights, output_size)
    request.add(affs_mask, output_size)
    request.add(pred_affs, output_size)

    # data_sources = tuple(
    #         ZarrSource(
    #                 os.path.join(data_dir, sample),
    #                 {
    #                     raw_fr: 'volumes/raw',
    #                     labels_fr: 'volumes/labels/neuron_ids',
    #                     labels_mask_fr: 'volumes/labels/labels_mask2',
    #                     unlabeled_mask_fr: 'volumes/labels/unlabeled'
    #                 },
    #                 {
    #                     raw_fr: ArraySpec(interpolatable=True),
    #                     labels_fr: ArraySpec(interpolatable=False),
    #                     labels_mask_fr: ArraySpec(interpolatable=False),
    #                     unlabeled_mask_fr: ArraySpec(interpolatable=False)
    #                 }
    #             ) +
    #         Normalize(raw_fr) +
    #         Pad(raw_fr, None) +
    #         Pad(labels_fr, labels_padding) +
    #         Pad(labels_mask_fr, labels_padding) +
    #         Pad(unlabeled_mask_fr, labels_padding) +
    #         RandomLocation() +

    #         xy_downsample(raw_fr, (1, xy_downsample, xy_downsample), raw) +
    #         xy_downsample(labels_fr, (1, xy_downsample, xy_downsample), labels) +
    #         xy_downsample(labels_mask_fr, (1, xy_downsample, xy_downsample), labels_mask) +
    #         xy_downsample(unlabeled_mask_fr, (1, xy_downsample, xy_downsample), unlabeled_mask)
    #         for sample in samples
    #     )

    data_sources = make_sources.make_sources()

    train_pipeline = data_sources

    train_pipeline += RandomProvider()

    train_pipeline += ElasticAugment(
            control_point_spacing=[4,int(40/xy_downsample),int(40/xy_downsample)],
            jitter_sigma=[0,2,2],
            rotation_interval=[0,math.pi/2.0],
            prob_slip=0.05,
            prob_shift=0.05,
            max_misalign=int(28/xy_downsample),
            subsample=8)

    train_pipeline += SimpleAugment(transpose_only=[1, 2])

    train_pipeline += IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1, z_section_wise=True)

    train_pipeline += DuplicateAugment(
            label_key=labels,
            voxel_size=(40, 4*xy_downsample, 4*xy_downsample),
            max_consecutive_duplicate=5,
            # prob_duplicate=0.01,
            prob_duplicate=0.03,
            # prob_edge_duplicate=0.01,
            prob_edge_duplicate=0.01,
        )

    train_pipeline += GrowBoundary(
            labels,
            steps=1,
            only_xy=True)

    train_pipeline += AddAffinities(
            neighborhood,
            labels=labels,
            affinities=gt_affs,
            labels_mask=labels_mask,
            unlabeled=unlabeled_mask,
            unlabeled_z_fix=False,
            affinities_mask=affs_mask)

    train_pipeline += BalanceLabels(
            gt_affs,
            affs_weights,
            affs_mask)

    train_pipeline += DefectAugment(
            raw, prob_missing=0.005, max_consecutive_missing=3)

    train_pipeline += IntensityScaleShift(raw, 2,-1)

    train_pipeline += Unsqueeze([raw])
    train_pipeline += Stack(batch_size)

    train_pipeline += PreCache(
            cache_size=40,
            num_workers=11)

    train_pipeline += Train(
            model=model,
            loss=loss,
            optimizer=optimizer,
            inputs={
                'input': raw
            },
            loss_inputs={
                0: pred_affs,
                1: gt_affs,
                2: affs_weights
            },
            outputs={
                0: pred_affs
            },
            save_every=5000,
            log_dir='log')

    train_pipeline += Squeeze([raw])
    train_pipeline += Squeeze([raw, gt_affs, pred_affs])

    train_pipeline += IntensityScaleShift(raw, 0.5, 0.5)

    train_pipeline += Snapshot({
                raw: 'raw',
                labels: 'labels',
                gt_affs: 'gt_affs',
                pred_affs: 'pred_affs',
                affs_weights: 'affs_weights',
                labels_mask: 'labels_mask',
                unlabeled_mask: 'unlabeled_mask'
            },
            every=500,
            output_filename='batch_{iteration}.zarr')

    train_pipeline += PrintProfilingStats(every=100)

    with build(train_pipeline) as b:
        for i in range(max_iteration):
            print(i)
            b.request_batch(request)

if __name__ == '__main__':

    iterations = 300000
    train_until(iterations)
