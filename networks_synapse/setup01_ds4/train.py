from __future__ import print_function

import json
import logging
import math
import os
# import pdb
# import sys
import json
import sys

import gunpowder as gp
import numpy as np
from synful.gunpowder import AddPartnerVectorMap, Hdf5PointsSource
from generate_network import mknet

from duplicate_augment import DuplicateAugment

total_samples = []
total_sample_weights = []

samples = [
    'ml1',
    'cutout1',
    'cutout2',
    'cutout5',
    'cutout6',
    'cutout7',
    ]

total_samples.extend(samples)
total_sample_weights = [float(1)/len(samples)] * len(samples)
samples_dir = '/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_gt/synapse_gt'


cb2samples_nosyn = [
    'nosynapse_cutout0',
    'nosynapse_cutout1',
    'nosynapse_cutout2',
    'nosynapse_cutout3',
    'nosynapse_cutout4',
    'nosynapse_cutout5',
    'nosynapse_cutout6',
]
sample_weights = [0.05/len(cb2samples_nosyn)] * len(cb2samples_nosyn)
total_samples += cb2samples_nosyn
total_sample_weights += sample_weights


def build_pipeline(parameter, augment=True):
    downsample_xy = parameter['downsample_xy']
    voxel_size = gp.Coordinate([40, 4 * parameter['downsample_xy'],
                                4 * parameter['downsample_xy']])

    # Checkpoint pointer.
    checkpoint = parameter['checkpoint']
    checkpoint_iteration = parameter['checkpoint_iteration']
    if checkpoint is not None:
        checkpoint = os.path.join('..', checkpoint,
                                  'train_net_checkpoint_%i' % checkpoint_iteration)

    # Array Specifications.
    raw_fr = gp.ArrayKey('RAW_FULL_RES')
    indicator_loss_weight = gp.ArrayKey('INDICATOR_LOSS_WEIGHT')
    pred_syn_indicator = gp.ArrayKey('PRED_SYN_INDICATOR')
    maxima = gp.ArrayKey('MAXIMA')
    labels_mask_fr = gp.ArrayKey('GT_LABELS_MASK_FULL_RES')
    gt_syn_indicator_fr = gp.ArrayKey('GT_SYN_INDICATOR_FULL_RES')

    if parameter['downsample_xy'] > 1:
        raw = gp.ArrayKey('RAW')
        labels_mask = gp.ArrayKey('GT_LABELS_MASK')
        gt_syn_indicator = gp.ArrayKey('GT_SYN_INDICATOR')
    else:
        raw = gp.ArrayKey('RAW_FULL_RES')
        labels_mask = gp.ArrayKey('GT_LABELS_MASK_FULL_RES')
        gt_syn_indicator = gp.ArrayKey('GT_SYN_INDICATOR_FULL_RES')

    grad_indicator = gp.ArrayKey('GRAD_INDICATOR')

    # # Points specifications
    # site = gp.PointsKey('PRESYN')
    # site_dummy = gp.PointsKey('POSTSYN')

    with open('train_net_config.json', 'r') as f:
        net_config = json.load(f)

    input_size = gp.Coordinate(net_config['input_shape']) * voxel_size
    output_size = gp.Coordinate(net_config['output_shape']) * voxel_size

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels_mask, output_size)
    request.add(gt_syn_indicator, output_size)
    request.add(indicator_loss_weight, output_size)

    snapshot_request = gp.BatchRequest({
        pred_syn_indicator: request[gt_syn_indicator],
        maxima: request[gt_syn_indicator],
        grad_indicator: request[gt_syn_indicator],
    })

    def create_source(sample):

        zarr_dir = os.path.join(samples_dir, f'{sample}/{sample}.zarr')
        src = (
            gp.ZarrSource(
                zarr_dir,
                datasets={
                    raw_fr: 'volumes/raw',
                    gt_syn_indicator_fr: 'volumes/labels/cleft_255',
                    labels_mask_fr: 'volumes/labels/cleft_mask_train',
                },
                array_specs={
                    raw_fr: gp.ArraySpec(interpolatable=True),
                    gt_syn_indicator_fr: gp.ArraySpec(interpolatable=False),
                    labels_mask_fr: gp.ArraySpec(interpolatable=False,
                                              dtype=np.uint8),
                }
            )
        )

        src += gp.Pad(raw_fr, None)
        src += gp.Pad(gt_syn_indicator_fr, gp.Coordinate((400, 256, 256)))
        src += gp.Pad(labels_mask_fr, gp.Coordinate((400, 256, 256)))

        if parameter['downsample_xy'] > 1:
            src += gp.DownSample(
                gt_syn_indicator_fr, (1, downsample_xy, downsample_xy), gt_syn_indicator)
            src += gp.DownSample(
                labels_mask_fr, (1, downsample_xy, downsample_xy), labels_mask)

        if sample in cb2samples_nosyn:
            src += gp.RandomLocation(
                p_nonempty=parameter['reject_probability'],
                # min_masked=0.01,
                mask=labels_mask
            )
        else:
            # ensure that each minibatch has at least 5% of the voxels labeled
            src += gp.RandomLocation(
                p_nonempty=parameter['reject_probability'],
                min_masked=0.01/6,
                mask=gt_syn_indicator
            )

        if parameter['downsample_xy'] > 1:
            src += gp.DownSample(
                raw_fr, (1, downsample_xy, downsample_xy), raw)

        src += gp.Normalize(raw)

        return src

    data_sources = tuple(
        create_source(sample) for sample in total_samples
    )

    pipeline = data_sources
    pipeline += gp.RandomProvider(probabilities=total_sample_weights)
    if augment:
        pipeline += gp.ElasticAugment(
            # control_point_spacing=[4, int(40/downsample_xy), int(40/downsample_xy)],
            # control_point_spacing=[8, int(80/downsample_xy), int(80/downsample_xy)],
            control_point_spacing=[12, int(120/downsample_xy), int(120/downsample_xy)],
            # control_point_spacing=[16, int(160/downsample_xy), int(160/downsample_xy)],
            jitter_sigma=[0, 2, 2],
            rotation_interval=[0, math.pi / 2.0],
            prob_slip=0.05,
            prob_shift=0.05,
            # max_misalign=int(28/parameter['downsample_xy']),
            max_misalign=int(12/parameter['downsample_xy']),
            # we don't want to simulate misalignment too much for synapse prediction...
            # too much -> hard to postprocess and connect up the prediction blobs
            subsample=4,
            voxel_size=(40, 4*parameter['downsample_xy'], 4*parameter['downsample_xy'])
            )
        pipeline += gp.SimpleAugment(transpose_only=[1, 2], mirror_only=[1, 2])
        pipeline += gp.IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1,
                                        z_section_wise=True)
        pipeline += DuplicateAugment(
            label_key=labels_mask,
            voxel_size=(40, 4*parameter['downsample_xy'], 4*parameter['downsample_xy']),
            max_consecutive_duplicate=5,
            prob_duplicate=0.03,
            prob_edge_duplicate=0,
        )
        pipeline += gp.DefectAugment(
            raw, prob_missing=0.005, max_consecutive_missing=3)

    pipeline += gp.IntensityScaleShift(raw, 2, -1)
    pipeline += gp.BalanceLabels(labels=gt_syn_indicator,
                                 scales=indicator_loss_weight,
                                 slab=(-1, -1, -1),
                                 mask=labels_mask,
                                 clipmin=parameter['cliprange'][0],
                                 clipmax=parameter['cliprange'][1])
    pipeline += gp.PreCache(
        cache_size=40,
        num_workers=15)
    pipeline += gp.tensorflow.Train(
        './train_net',
        optimizer=net_config['optimizer'],
        loss=net_config['loss'],
        summary=net_config['summary'],
        log_dir='./tensorboard/',
        save_every=20000,  # 10000
        # log_every=10000,
        inputs={
            net_config['raw']: raw,
            net_config['gt_syn_indicator']: gt_syn_indicator,
            net_config['indicator_weight']: indicator_loss_weight,  # Loss weights,
            net_config['gt_mask']: labels_mask,  # Mask for GT

        },
        outputs={
            net_config['pred_syn_indicator']: pred_syn_indicator,
            net_config['maxima']: maxima
        },
        gradients={
            net_config['pred_syn_indicator']: grad_indicator,
        },
    )
    # Visualize.
    pipeline += gp.IntensityScaleShift(raw, 0.5, 0.5)
    pipeline += gp.Snapshot({
        raw: 'volumes/raw',
        gt_syn_indicator: 'volumes/gt_syn_indicator',
        pred_syn_indicator: 'volumes/pred_syn_indicator',
        indicator_loss_weight: 'volumes/indicator_loss_weight',
        maxima: 'volumes/maxima',
        grad_indicator: 'volumes/post_indicator_gradients',
        labels_mask: 'volumes/labels_mask',
    },
        every=parameter['snapshot_freq'],
        output_filename='batch_{iteration}.hdf',
        compression_type='gzip',
        additional_request=snapshot_request)
    pipeline += gp.PrintProfilingStats(every=100)

    print("Starting training...")
    max_iteration = parameter['max_iteration']
    with gp.build(pipeline) as b:
        for i in range(max_iteration):
            b.request_batch(request)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # For Training.
    parameter = {
        'fmap_num': 4,  # Initial num feature maps.
        'fmap_inc_factor': 5,
        'num_repetitions_last_layer': 1,  # Default is 1.
        'activation': None,  # activation of last layer for indicator loss
        'checkpoint': None,
        'checkpoint_iteration': None,
        'unet_model': 'vanilla',  # dh_unet or vanilla
        # Training.
        'learning_rate': 0.5e-4,
        'clip_gradients': False,
        # Loss
        'loss_comb_type': 'sum',
        'd_transition': False,
        'd_transition_iteration': None,
        # when to transition from d_loss_weight1 to 2
        'd_loss_weight1': 'mask',  # possible: mask, maxima, full
        'd_loss_weight2': None,  # only relevant if transition is set to True
        'm_loss_type': 'cross_entropy',
        'm_loss_scale': 1.0,
        'd_loss_scale': 0.0,
        'downsample_xy': 4
    }
    # Gunpowder specific.
    # clipfactor = 5e-2
    clipfactor = 0.1
    parameter.update({
        'reject_probability': 0.9,
        'blob_radius': 16,
        # 'blob_radius': 28, # max
        'max_iteration': 800000, #was 300000
        'snapshot_freq': 10000, #was 10000
        'blob_mode': 'ball',
        'd_scale': 1,
        'd_blob_radius': 100,
        'cliprange': (clipfactor, 1 - clipfactor)
        # 'cliprange': [7e-4, 0.9993]
    })

    # Architecture.
    if parameter['downsample_xy'] == 1:
        parameter['input_size'] = (48, 430, 430)  # output size: 20, 162, 162
        parameter['downsample_factors'] = [[1, 3, 3], [1, 3, 3], [3, 3, 3]]
        test_net_mult_z = 60
        test_net_mult_xy = 20

    elif parameter['downsample_xy'] == 2:
        parameter['downsample_factors'] = [[1, 2, 2], [1, 2, 2], [3, 3, 3]]
        parameter['input_size'] = (48, 220, 220)  # 6, 96, 96
        test_net_mult_z = 60
        test_net_mult_xy = 20

    elif parameter['downsample_xy'] == 4:
        parameter['input_size'] = (60, 172, 172)
        parameter['downsample_factors'] = [[1, 2, 2], [1, 2, 2], [3, 2, 2]]
        test_net_mult_z = 28
        test_net_mult_xy = 60

    else:
        assert False, "Untested downsample_xy"

    if len(sys.argv) >= 2 and sys.argv[1] == 'mknet':
        parameter_test = parameter.copy()
        input_size = parameter['input_size']
        downsample_factors_zyx = np.prod(np.array(parameter['downsample_factors']), axis=0)
        parameter_test['input_size'] = (
            parameter['input_size'][0] + test_net_mult_z*downsample_factors_zyx[0],
            parameter['input_size'][1] + test_net_mult_xy*downsample_factors_zyx[1],
            parameter['input_size'][2] + test_net_mult_xy*downsample_factors_zyx[2],
            )
        parameter_test['input_size'] = tuple([int(k) for k in parameter_test['input_size']])

        output_size = mknet(parameter_test, name='test_net')
        parameter_test['output_size'] = output_size
        json.dump(parameter_test, open("parameter.json", 'w'))

        output_size = mknet(parameter, name='train_net')
        parameter['output_size'] = output_size
        # parameter['output_size'] = [6, 80, 80]
    else:

        if len(sys.argv) >= 2 and sys.argv[1] == 'test':
            parameter.update({
                'max_iteration': 101,
                'snapshot_freq': 10,
                })
        build_pipeline(parameter, augment=True)