from __future__ import print_function
import sys

from gunpowder import *
from gunpowder.tensorflow import *
from duplicate_augment import DuplicateAugment
from reject import Reject
import os
import math
import json
import tensorflow as tf
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

xy_downsample = 2

dense_data_path= '../data/dense/'
background_data_path= '../data/background/'
negative_data_path = '../data/negative/'
sparse_data_path = '../data/sparse/'
special_min_masks = {}

dense_samples = [
    dense_data_path + 'gl0_setup45.zarr',
    dense_data_path + 'gl1_setup50_v2_large.zarr',
    dense_data_path + 'ml0_setup45.zarr',
    dense_data_path + 'ml4_setup45.zarr',
    dense_data_path + 'ml5_setup45.zarr',
    dense_data_path + 'pl2_setup45.zarr',
]

# myelin samples only have myelin set to background
# other neurons are also labeled but not proofread
# necessary that they are labeled so that the network is not biased toward background labeling for other cells
background_samples = [
    background_data_path + 'cb2_myelin_cutout0_v2.zarr',
    background_data_path + 'cb2_myelin_cutout2_v2.zarr',
    background_data_path + 'cb2_myelin_cube_v2.zarr',
]

# SPARSE SAMPLES
# only a few neurons are labeled and proofread
# other neurons are set to 0 in unlabeled_mask
sparse_samples = [
    sparse_data_path + 'ml_branch_boundaries_cropped.zarr',  # mito
    sparse_data_path + 'mcp1.zarr',  # mito
]
# these ones has very sparse labeling so we decrease the min mask to make it work
special_min_masks[sparse_data_path + 'ml_branch_boundaries_cropped.zarr'] = 0.2
special_min_masks[sparse_data_path + 'mcp1.zarr'] = 0.2


# NEGATIVE SAMPLES
# usually dense labeled with all cells proofread
negative_samples = []
negative_samples = [
    negative_data_path + 'cb2_negative_cutout1.zarr',
    negative_data_path + 'cb2_negative_cutout2.zarr',
    negative_data_path + 'cb2_negative_cutout3.zarr',
    negative_data_path + 'cb2_negative_cutout4.zarr',
    ]

new_samples = []


new_samples = []


affs = ArrayKey('PREDICTED_AFFS')
gt_affs = ArrayKey('GT_AFFINITIES')
gt_affs_mask = ArrayKey('GT_AFFINITIES_MASK')
gt_affs_scale = ArrayKey('GT_AFFINITIES_SCALE')
affs_gradient = ArrayKey('AFFS_GRADIENT')

raw_fr = ArrayKey('RAW_FR')
labels_fr = ArrayKey('GT_LABELS_FR')
labels_mask_fr = ArrayKey('GT_LABELS_MASK_FR')
unlabeled_mask_fr = ArrayKey('GT_UNLABELED_MASK_FR')

if xy_downsample > 1:

    raw = ArrayKey('RAW')
    labels = ArrayKey('GT_LABELS')
    labels_mask = ArrayKey('GT_LABELS_MASK')
    unlabeled_mask = ArrayKey('GT_UNLABELED_MASK')

else:
    raw = ArrayKey('RAW_FR')
    labels = ArrayKey('GT_LABELS_FR')
    labels_mask = ArrayKey('GT_LABELS_MASK_FR')
    unlabeled_mask = ArrayKey('GT_UNLABELED_MASK_FR')

### CREATING GT WEIGHTS
total_samples = []
total_sample_weights = []
sample_mask_attrs = {}

def add_weights(samples, weight):
    global total_samples
    global total_sample_weights
    global sample_mask_attrs
    global special_min_masks

    global background_samples
    global sparse_samples
    global negative_samples
    global dense_samples

    for s in samples:

        assert os.path.exists(s), "Sample %s does not exist!" % s

        logical_and_mask = None
        logical_and_mask_val = 0
        mask = unlabeled_mask
        min_mask = 0.5

        if s in background_samples:
            mask = unlabeled_mask
            logical_and_mask = labels
            min_mask = 0.6

        if s in sparse_samples:
            assert s in special_min_masks
        if s in special_min_masks:
            min_mask = special_min_masks[s]

        total_samples.append(s)
        total_sample_weights.append(weight)
        sample_mask_attrs[s] = (
            mask, min_mask, logical_and_mask, logical_and_mask_val)

add_weights(dense_samples, 1)
add_weights(background_samples, 0.05)  # lower?
add_weights(negative_samples, 0.025)  # lower?
add_weights(sparse_samples, 0.075)  # 0.05 has bad performance at 100k, ok-ish at 200k
add_weights(new_samples, 5)

print("Samples used for training:")
for s, w in zip(total_samples, total_sample_weights):
    print("%s: %s" % (s, w))

neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]


def train_until(max_iteration, num_workers):

    if tf.train.latest_checkpoint('.'):
        trained_until = int(tf.train.latest_checkpoint('.').split('_')[-1])
    else:
        trained_until = 0
    if trained_until >= max_iteration:
        return

    try:
        with open('unet.json', 'r') as f:
            config = json.load(f)
    except:
        with open('config.json', 'r') as f:
            config = json.load(f)


    voxel_size = Coordinate((40, 4*xy_downsample, 4*xy_downsample))
    input_size = Coordinate(config['input_shape'])*voxel_size
    output_size = Coordinate(config['output_shape'])*voxel_size

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(unlabeled_mask, output_size)
    request.add(gt_affs, output_size)
    request.add(gt_affs_mask, output_size)
    request.add(gt_affs_scale, output_size)

    snapshot_request = BatchRequest({
        affs: request[gt_affs],
        affs_gradient: request[gt_affs]
    })

    def create_source(sample):

        src = ZarrSource(
                sample,
                datasets={
                    raw_fr: 'volumes/raw',
                    labels_fr: 'volumes/labels/neuron_ids',
                    labels_mask_fr: 'volumes/labels/labels_mask2',
                    unlabeled_mask_fr: 'volumes/labels/unlabeled',
                },
                array_specs={
                    raw_fr: ArraySpec(interpolatable=True),
                    labels_fr: ArraySpec(interpolatable=False),
                    labels_mask_fr: ArraySpec(interpolatable=False),
                    unlabeled_mask_fr: ArraySpec(interpolatable=False),
                }
            )

        src += Pad(raw_fr, None)
        src += Pad(labels_fr, Coordinate((1000, 256, 256)))
        src += Pad(labels_mask_fr, Coordinate((1000, 256, 256)))
        src += Pad(unlabeled_mask_fr, Coordinate((1000, 256, 256)))
        src += RandomLocation()

        if xy_downsample > 1:
            src += DownSample(unlabeled_mask_fr, (1, xy_downsample, xy_downsample), unlabeled_mask)
            src += DownSample(labels_fr, (1, xy_downsample, xy_downsample), labels)

        mask, min_mask, logical_and_mask, logical_and_mask_val = sample_mask_attrs[sample]

        src += Reject(
            mask=mask,
            min_masked=min_mask,
            reject_probability=0.99,
            logical_and_mask=logical_and_mask,
            logical_and_mask_val=logical_and_mask_val,
            )

        if xy_downsample > 1:
            src += DownSample(raw_fr, (1, xy_downsample, xy_downsample), raw)
            src += DownSample(labels_mask_fr, (1, xy_downsample, xy_downsample), labels_mask)

        src += Normalize(raw)

        return src

    data_sources = tuple(
        create_source(sample) for sample in total_samples
    )

    train_pipeline = (
        data_sources +
        RandomProvider(probabilities=total_sample_weights) +
        ElasticAugment(
            control_point_spacing=[4,40,40],
            jitter_sigma=[0,2,2],
            rotation_interval=[0,math.pi/2.0],
            prob_slip=0.05,
            prob_shift=0.05,
            max_misalign=int(28/xy_downsample),
            subsample=8,
            voxel_size=(40, 4*xy_downsample, 4*xy_downsample)
            ) +
        SimpleAugment(transpose_only=[1, 2]) +
        IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        DuplicateAugment(
            label_key=labels,
            voxel_size=(40, 4*xy_downsample, 4*xy_downsample),
            max_consecutive_duplicate=5,
            # prob_duplicate=0.01,
            prob_duplicate=0.03,
            # prob_edge_duplicate=0.01,
            prob_edge_duplicate=0.01,
        ) +
        GrowBoundary(labels, labels_mask, steps=1, only_xy=True) +
        AddAffinities(
            neighborhood,
            labels=labels,
            affinities=gt_affs,
            labels_mask=labels_mask,
            unlabelled=unlabeled_mask,
            unlabelled_z_fix=True,
            affinities_mask=gt_affs_mask) +
        BalanceLabels(
            gt_affs,
            gt_affs_scale,
            gt_affs_mask) +
        DefectAugment(raw, prob_missing=0.005, max_consecutive_missing=3) +
        IntensityScaleShift(raw, 2,-1) +
        PreCache(
            cache_size=40,
            num_workers=num_workers-1) +
        Train(
            './unet',
            optimizer=config['optimizer'],
            loss=config['loss'],
            inputs={
                config['raw']: raw,
                config['gt_affs']: gt_affs,
                config['loss_weights_affs']: gt_affs_scale,
            },
            outputs={
                config['affs']: affs
            },
            gradients={
                config['affs']: affs_gradient
            },
            summary=config['summary'],
            log_dir='log',
            save_every=5000) +
        IntensityScaleShift(raw, 0.5, 0.5) +
        Snapshot({
                raw: 'volumes/raw',
                labels: 'volumes/labels/neuron_ids',
                gt_affs: 'volumes/gt_affinities',
                affs: 'volumes/pred_affinities',
                labels_mask: 'volumes/labels/mask',
                unlabeled_mask: 'volumes/labels/unlabeled',
                affs_gradient: 'volumes/affs_gradient'
            },
            dataset_dtypes={
                labels: np.uint64
            },
            every=10000,
            # every=1,
            output_filename='batch_{iteration}.hdf',
            additional_request=snapshot_request) +
        PrintProfilingStats(every=100)
    )

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(max_iteration - trained_until):
            b.request_batch(request)
    print("Training finished")

if __name__ == "__main__":
    iteration = int(sys.argv[1])
    num_workers = int(sys.argv[2])
    train_until(iteration, num_workers)

