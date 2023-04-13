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

# cb2samples_nosyn = [
#     'nosynapse_cutout0',
#     'nosynapse_cutout1',
#     'nosynapse_cutout2',
#     'nosynapse_cutout3',
#     'nosynapse_cutout4',
#     'nosynapse_cutout5',
#     'nosynapse_cutout6',
# ]
# sample_weights = [0.5/len(cb2samples_nosyn)] * len(cb2samples_nosyn)
# total_samples += cb2samples_nosyn
# total_sample_weights += sample_weights

cb2samples = [
    ('cutout1', 'cleft2pre_200816'),
    ('cutout2', 'cleft2pre_200816'),
    # ('cutout3', 'cleft2pre_200816'),
    ('cutout4', 'cleft2pre_200816'),
    ('cutout5', 'cleft2pre_200816'),
    # 'cutout6',
    ('cutout7', 'cleft2pre_200816'),
    # 'cutout8',
    # 'cutout9',
    ]
sample_weights = [1] * len(cb2samples)
total_samples += cb2samples
total_sample_weights += sample_weights

cb2samples += [
    ('pl2', 'cleft2pre_200816'),
    ('ml1', 'cleft2pre_200816'),
    ]
sample_weights += [2] * 2
total_samples += cb2samples
total_sample_weights += sample_weights

cb2_samples_dir = '/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_gt/synapse_gt'


def json_to_roi(jsonfile):
    with open(jsonfile, 'r') as f:
        config = json.load(f)
    offset = config['offset']
    shape = config['size']
    return gp.Roi(offset=offset, shape=shape)


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
    raw = gp.ArrayKey('RAW')
    raw_fr = gp.ArrayKey('RAW_FULL_RES')
    gt_postpre_vectors = gp.ArrayKey('GT_POSTPRE_VECTORS')
    gt_post_indicator = gp.ArrayKey('GT_POST_INDICATOR')
    post_loss_weight = gp.ArrayKey('POST_LOSS_WEIGHT')
    vectors_mask = gp.ArrayKey('VECTORS_MASK')
    pred_postpre_vectors = gp.ArrayKey('PRED_POSTPRE_VECTORS')
    pred_post_indicator = gp.ArrayKey('PRED_POST_INDICATOR')
    maxima = gp.ArrayKey('MAXIMA')
    labels_mask = gp.ArrayKey('GT_LABELS_MASK')
    labels_mask_fr = gp.ArrayKey('GT_LABELS_MASK_FULL_RES')

    grad_syn_indicator = gp.ArrayKey('GRAD_SYN_INDICATOR')
    grad_partner_vectors = gp.ArrayKey('GRAD_PARTNER_VECTORS')

    # Points specifications
    postsyn = gp.PointsKey('POSTSYN')
    presyn = gp.PointsKey('PRESYN')
    core_points = gp.PointsKey('CORE_POINTS')

    with open('train_net_config.json', 'r') as f:
        net_config = json.load(f)

    input_size = gp.Coordinate(net_config['input_shape']) * voxel_size
    output_size = gp.Coordinate(net_config['output_shape']) * voxel_size

    # Since points roi is enlarged by AddPartnerVectorMap, add a dummy point
    # request (core_points) to avoid point empty batches downstream of
    # AddPartnerVectorMap.
    trg_context = 120*8  # AddPartnerVectorMap context in nm - pre-post distance
    # trg_context = 0
    core_output_ratio = 0.7  # empiricallly found, account for augmentation
    core_size = [max(1, int(d * core_output_ratio)) for d in
                 net_config['output_shape']]
    core_size = gp.Coordinate(core_size) * voxel_size
    print(core_size, 'core size')

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels_mask, output_size)
    request.add(gt_postpre_vectors, output_size)
    request.add(gt_post_indicator, output_size)
    request.add(post_loss_weight, output_size)
    request.add(core_points, core_size)
    request.add(vectors_mask, output_size)

    snapshot_request = gp.BatchRequest({
        pred_post_indicator: request[gt_postpre_vectors],
        pred_postpre_vectors: request[gt_postpre_vectors],
        maxima: request[gt_postpre_vectors],
        grad_syn_indicator: request[gt_postpre_vectors],
        grad_partner_vectors: request[gt_postpre_vectors],
        vectors_mask: request[gt_postpre_vectors],
    })

    syn_rastersetting = gp.RasterizationSettings(
        radius=parameter['blob_radius'], mode=parameter['blob_mode'])

    def create_source(sample):

        gt_name = None
        sample0 = sample
        if isinstance(sample, tuple):
            sample, gt_name = sample

        if sample0 in cb2samples:
            data_dir = cb2_samples_dir + '/{}/'.format(sample)
            zarr_dir = os.path.join(data_dir, sample + '.zarr')
        # elif sample in cb2samples_new:
        #     data_dir = cb2_samples_dir + '/{}/'.format(sample)
        #     zarr_dir = os.path.join(data_dir, sample + '.zarr')
        elif sample0 in cb2samples_nosyn:
            n = sample.split('_')[1]
            data_dir = cb2_samples_dir + '/nosynapse_{}/'.format(n)
            zarr_dir = os.path.join(data_dir, 'cb2_negative_' + n + '.zarr')
        else:
            assert False

        train_mask = 'volumes/mask'
        if gt_name is not None:
            train_mask = f'volumes/train_masks/{gt_name}'

        synapse_mask = os.path.join(zarr_dir, 'mask_roi.json')
        if gt_name is not None:
            synapse_mask = os.path.join(
                zarr_dir, 'annotations',
                f'{gt_name}/synapse_mask.json')

        synapse_data = os.path.join(data_dir, 'synapses.hdf')
        if gt_name is not None:
            synapse_data = os.path.join(
                data_dir, 'annotations',
                f'{gt_name}/synapses.hdf')

        src = (
            Hdf5PointsSource(
                synapse_data,
                datasets={presyn: 'annotations',
                          postsyn: 'annotations'},
                rois={
                    presyn: json_to_roi(synapse_mask),
                    postsyn: json_to_roi(synapse_mask)
                }
            ),
            Hdf5PointsSource(
                synapse_data,
                datasets={core_points: 'annotations'},
                rois={
                    core_points: json_to_roi(synapse_mask),
                },
                kind='postsyn'
            ),
            gp.ZarrSource(
                os.path.join(zarr_dir),
                datasets={
                    raw_fr: 'volumes/raw',
                    labels_mask_fr: train_mask,
                },
                array_specs={
                    raw_fr: gp.ArraySpec(interpolatable=True),
                    labels_mask_fr: gp.ArraySpec(interpolatable=False,
                                              dtype=np.uint8)
                }
            )
        ) + gp.MergeProvider()

        src += gp.Pad(raw_fr, None)

        if parameter['downsample_xy'] > 1:
            src += gp.DownSample(
                labels_mask_fr, (1, downsample_xy, downsample_xy), labels_mask)

        # if sample0 in cb2samples:
        src += gp.RandomLocation(
            ensure_nonempty=core_points,
            p_nonempty=parameter['reject_probability'],
            min_masked=0.5,
            mask=labels_mask
        )
        # else:
        # for negative cutouts
        # src += gp.RandomLocation(
        #     mask=labels_mask
        # )

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
            [4, int(40/parameter['downsample_xy']), int(40/parameter['downsample_xy'])],
            [0, 2, 2],
            [0, math.pi / 2.0],
            prob_slip=0.05,
            prob_shift=0.05,
            # max_misalign=int(28/parameter['downsample_xy']),
            max_misalign=int(10/parameter['downsample_xy']),
            subsample=8,
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
            # prob_edge_duplicate=0.01,
        )
        pipeline += gp.DefectAugment(
            raw, prob_missing=0.005, max_consecutive_missing=3)

    pipeline += gp.IntensityScaleShift(raw, 2, -1)
    pipeline += gp.RasterizePoints(presyn, gt_post_indicator,
                                   gp.ArraySpec(voxel_size=voxel_size,
                                                dtype=np.int32),
                                   syn_rastersetting)
    spec = gp.ArraySpec(voxel_size=voxel_size)
    pipeline += AddPartnerVectorMap(
        src_points=presyn,
        trg_points=postsyn,
        array=gt_postpre_vectors,
        radius=parameter['d_blob_radius'],
        trg_context=trg_context,  # enlarge
        array_spec=spec,
        pointmask=vectors_mask
    )
    pipeline += gp.BalanceLabels(labels=gt_post_indicator,
                                 scales=post_loss_weight,
                                 slab=(-1, -1, -1),
                                 mask=labels_mask,
                                 clipmin=parameter['cliprange'][0],
                                 clipmax=parameter['cliprange'][1])
    if parameter['d_scale'] != 1:
        pipeline += gp.IntensityScaleShift(gt_postpre_vectors,
                                           scale=parameter['d_scale'], shift=0)
    pipeline += gp.PreCache(
        cache_size=40,
        num_workers=15)
    pipeline += gp.tensorflow.Train(
        './train_net',
        optimizer=net_config['optimizer'],
        loss=net_config['loss'],
        summary=net_config['summary'],
        log_dir='./tensorboard/',
        save_every=10000,  # 10000
        log_every=100,
        inputs={
            net_config['raw']: raw,
            net_config['gt_partner_vectors']: gt_postpre_vectors,
            net_config['gt_syn_indicator']: gt_post_indicator,
            net_config['vectors_mask']: vectors_mask,
            # Loss weights --> mask
            net_config['indicator_weight']: post_loss_weight,  # Loss weights,
            net_config['gt_mask']: labels_mask,  # Mask for GT

        },
        outputs={
            net_config['pred_partner_vectors']: pred_postpre_vectors,
            net_config['pred_syn_indicator']: pred_post_indicator,
            net_config['maxima']: maxima
        },
        gradients={
            net_config['pred_partner_vectors']: grad_partner_vectors,
            net_config['pred_syn_indicator']: grad_syn_indicator,
        },
    )
    # Visualize.
    pipeline += gp.IntensityScaleShift(raw, 0.5, 0.5)
    pipeline += gp.Snapshot({
        raw: 'volumes/raw',
        gt_post_indicator: 'volumes/gt_post_indicator',
        gt_postpre_vectors: 'volumes/gt_postpre_vectors',
        pred_postpre_vectors: 'volumes/pred_postpre_vectors',
        pred_post_indicator: 'volumes/pred_post_indicator',
        post_loss_weight: 'volumes/post_loss_weight',
        maxima: 'volumes/maxima',
        grad_syn_indicator: 'volumes/post_indicator_gradients',
        grad_partner_vectors: 'volumes/partner_vectors_gradients',
        vectors_mask: 'volumes/vectors_mask',
        labels_mask: 'volumes/labels_mask'
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
        # Architecture.
        # 'input_size': (42, 430, 430),
        # 'input_size': (42, 172, 172),
        'input_size': (48, 172, 172),
        # 'input_size': (52, 508, 508), # 1,2,2 ; 1,2,2 ; 2, 3, 3
        # 'input_size': (42, 592, 592), # 1,2,2 ; 1,2,2 ; 3, 2, 2
        # 'downsample_factors': [[1, 3, 3], [1, 3, 3], [3, 3, 3]],
        'downsample_factors': [[1, 2, 2], [1, 2, 2], [3, 2, 2]],
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
        'm_loss_scale': 0.0,
        'd_loss_scale': 1.0,
        'downsample_xy': 4
    }
    # Gunpowder specific.
    clipfactor = 0.05
    parameter.update({
        'reject_probability': 0.85,
        'blob_radius': 80,
        'max_iteration': 1600000,
        'snapshot_freq': 10000,
        'blob_mode': 'ball',
        'd_scale': 1,
        'd_blob_radius': 100,
        'cliprange': (clipfactor, 1 - clipfactor)
    })

    # build_pipeline(parameter, augment=True)
    # output_size = mknet(parameter, name='train_net')
    # parameter['output_size'] = output_size

    if len(sys.argv) >= 2 and sys.argv[1] == 'mknet':
        parameter_test = parameter.copy()
        z = 8
        # xy = 80  # OOM
        xy = 60
        parameter_test['input_size'] = (48 + z * 12, 172 + xy * 8, 172 + xy * 8)
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