import json

import numpy as np
import tensorflow as tf

from funlib.learn.tensorflow import models


def mknet(parameter, name):
    learning_rate = parameter['learning_rate']
    input_shape = parameter['input_size']
    indicator_activation = parameter['activation']
    fmap_inc_factor = parameter['fmap_inc_factor']
    downsample_factors = parameter['downsample_factors']
    fmap_num = parameter['fmap_num']
    unet_model = parameter['unet_model']
    num_heads = 2 if unet_model == 'dh_unet' else 1
    m_loss_scale = parameter['m_loss_scale']
    d_loss_scale = parameter['d_loss_scale']
    if m_loss_scale is None or d_loss_scale is None:
        assert m_loss_scale is None and d_loss_scale is None
        m_loss_scale = 1
        d_loss_scale = 1
    else:
        assert parameter['loss_comb_type'] == 'sum'
    if parameter['m_loss_type'] == 'cross_entropy':
        indicator_activation = None
    assert parameter['m_loss_type'] == 'cross_entropy' or \
           parameter[
               'm_loss_type'] == 'mean_squared_error', 'm_loss_type {} unknown'.format(
        parameter['m_loss_type'])
    assert unet_model == 'vanilla' or unet_model == 'dh_unet', 'unknown unetmodel {}'.format(
        unet_model)
    assert parameter[
               'd_loss_weight1'] == 'mask', '{} d_loss_weight not implemeted or not checked'.format(
        parameter['d_loss_weight1'])
    assert parameter['loss_comb_type'] == 'sum' or parameter[
        'loss_comb_type'] == 'product'

    tf.reset_default_graph()

    # d, h, w
    raw = tf.placeholder(tf.float32, shape=input_shape)

    # b=1, c=1, d, h, w
    raw_batched = tf.reshape(raw, (1, 1) + input_shape)

    # b=1, c=fmap_num, d, h, w
    outputs, fov, voxel_size = models.unet(raw_batched,
                                           fmap_num, fmap_inc_factor,
                                           downsample_factors,
                                           num_heads=num_heads,
                                           voxel_size=(40, 4*parameter['downsample_xy'], 4*parameter['downsample_xy']))
    if num_heads == 1:
        outputs = (outputs, outputs)
    print('unet has fov in nm: ', fov)

    # b=1, c=3, d, h, w
    partner_vectors_batched, fov = models.conv_pass(
        outputs[0],
        kernel_sizes=[1],
        num_fmaps=3,
        activation=None,  # Regression
        name='partner_vector')

    # b=1, c=1, d, h, w
    syn_indicator_batched, fov = models.conv_pass(
        outputs[1],
        kernel_sizes=[1],
        num_fmaps=1,
        activation=indicator_activation,
        name='syn_indicator')
    print('fov in nm: ', fov)

    # d, h, w
    output_shape = tuple(syn_indicator_batched.get_shape().as_list()[
                         2:])  # strip batch and channel dimension.
    syn_indicator_shape = output_shape

    # c=3, d, h, w
    partner_vectors_shape = (3,) + syn_indicator_shape

    # c=3, d, h, w
    pred_partner_vectors = tf.reshape(partner_vectors_batched,
                                      partner_vectors_shape)
    gt_partner_vectors = tf.placeholder(tf.float32, shape=partner_vectors_shape)
    vectors_mask = tf.placeholder(tf.float32,
                                  shape=syn_indicator_shape)  # d,h,w
    gt_mask = tf.placeholder(tf.bool, shape=syn_indicator_shape)  # d,h,w
    vectors_mask = tf.cast(vectors_mask, tf.bool)

    # d, h, w
    pred_syn_indicator = tf.reshape(syn_indicator_batched,
                                    syn_indicator_shape)  # squeeze batch dimension
    gt_syn_indicator = tf.placeholder(tf.float32, shape=syn_indicator_shape)
    indicator_weight = tf.placeholder(tf.float32, shape=syn_indicator_shape)

    maxima = tf.nn.pool(
        tf.reshape(pred_syn_indicator, (1,) + syn_indicator_shape + (1,)),
        [1, 10, 10],
        'MAX',
        'SAME',
        strides=[1, 1, 1],
        data_format='NDHWC'
    )

    maxima = tf.reshape(maxima, syn_indicator_shape)
    maxima = tf.equal(pred_syn_indicator, maxima, name='maxima')
    if parameter['d_loss_weight1'] == 'full':
        # full mode: dir vec magnitue encodes presence of synapse, also use FN for training, add gt_mask.
        maxima = tf.logical_and(maxima,
                                pred_syn_indicator > 0.001)  # Do not consider homogeneous background.
        maxima = tf.logical_and(maxima,
                                gt_mask)  # do not use gt_mask for training.
    elif parameter['d_loss_weight1'] == 'mask':
        # if only trained in synapse regions, only use TP for training.
        maxima = tf.logical_and(maxima, tf.cast(vectors_mask, tf.bool))
    else:
        raise NotImplementedError(
            'unclear what to do: reconsider setting of d_loss_weight1')

    partner_vectors_loss_weighted = tf.losses.mean_squared_error(
        gt_partner_vectors,
        pred_partner_vectors,
        tf.reshape(
            indicator_weight,
            (1,) + syn_indicator_shape),
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

    partner_vectors_loss_mask = tf.losses.mean_squared_error(
        gt_partner_vectors,
        pred_partner_vectors,
        tf.reshape(
            vectors_mask,
            (1,) + syn_indicator_shape),
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

    partner_vectors_loss_maxima = tf.losses.mean_squared_error(
        gt_partner_vectors,
        pred_partner_vectors,
        tf.reshape(
            maxima,
            (1,) + syn_indicator_shape),
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)  # consider TP, FP.

    if parameter['m_loss_type'] == 'mean_squared_error':
        syn_indicator_loss_weighted = tf.losses.mean_squared_error(
            gt_syn_indicator,
            pred_syn_indicator,
            indicator_weight,
            reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)  # --> Although the name is a little confusing, in fact,
        # this reduction method is actually an average over nonzero weights.
        if indicator_activation is None:
            pred_syn_indicator_out = tf.clip_by_value(pred_syn_indicator, 0,
                                                      1)  # clip values for downstream computation.
        else:
            pred_syn_indicator_out = pred_syn_indicator
    elif parameter['m_loss_type'] == 'cross_entropy':
        syn_indicator_loss_weighted = tf.losses.sigmoid_cross_entropy(
            gt_syn_indicator,
            pred_syn_indicator,
            indicator_weight,
            reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        pred_syn_indicator_out = tf.sigmoid(pred_syn_indicator)  # For output.

    d_loss_dic = {
        'mask': partner_vectors_loss_mask,
        'full': partner_vectors_loss_weighted,
        'maxima': partner_vectors_loss_maxima
    }

    iteration = tf.Variable(1.0, name='training_iteration', trainable=False)
    if parameter['loss_comb_type'] == 'sum':
        if parameter['d_transition']:
            alpha = tf.constant(1.0) / (
                    tf.constant(1.0) + tf.exp(
                tf.constant(0.01) *
                (-iteration + tf.constant(
                    float(parameter['d_transition_iteration'])))
            )
            )
            loss = syn_indicator_loss_weighted + (
                    d_loss_dic[parameter['d_loss_weight2']] * alpha +
                    d_loss_dic[parameter['d_loss_weight1']] * (1.0 - alpha)
            )
        else:
            loss = m_loss_scale * syn_indicator_loss_weighted + d_loss_scale * \
                   d_loss_dic[
                       parameter['d_loss_weight1']]
    elif parameter['loss_comb_type'] == 'product':
        loss = syn_indicator_loss_weighted * d_loss_dic[
            parameter['d_loss_weight1']]

    # Monitor in tensorboard.
    tp_fp_ratio = tf.losses.absolute_difference(gt_syn_indicator, tf.cast(
        pred_syn_indicator_out > 0.5, dtype=tf.uint8),
                                                weights=gt_syn_indicator,
                                                reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)  # Approximate TP.
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('loss_vectors', partner_vectors_loss_mask)
    tf.summary.scalar('loss_vectors_maxima', partner_vectors_loss_maxima)
    tf.summary.scalar('loss_vectors_weighted', syn_indicator_loss_weighted)
    tf.summary.scalar('loss_indicator', syn_indicator_loss_weighted)
    tf.summary.scalar('tp_fp_ratio', tp_fp_ratio)
    if parameter['d_transition']:
        tf.summary.scalar('alpha', alpha)

    summary = tf.summary.merge_all()

    # l=1, d, h, w
    print("input shape : %s" % (input_shape,))
    print("output shape: %s" % (output_shape,))

    # Train Ops.
    opt = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8
    )
    if parameter['clip_gradients']:
        gradients, variables = zip(*opt.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
        optimizer = opt.apply_gradients(zip(gradients, variables),
                                        global_step=iteration)
    else:
        gvs_ = opt.compute_gradients(loss)
        optimizer = opt.apply_gradients(gvs_, global_step=iteration)

    tf.train.export_meta_graph(filename=name + '.meta')

    names = {
        'raw': raw.name,
        'gt_partner_vectors': gt_partner_vectors.name,
        'pred_partner_vectors': pred_partner_vectors.name,
        'gt_syn_indicator': gt_syn_indicator.name,
        'pred_syn_indicator': pred_syn_indicator.name,
        'pred_syn_indicator_out': pred_syn_indicator_out.name,
        'indicator_weight': indicator_weight.name,
        'vectors_mask': vectors_mask.name,
        'gt_mask': gt_mask.name,
        'maxima': maxima.name,
        'loss': loss.name,
        'optimizer': optimizer.name,
        'summary': summary.name,
        'input_shape': input_shape,
        'output_shape': output_shape}

    names['outputs'] = {'pred_syn_indicator_out':
                            {"out_dims": 1, "out_dtype": "uint8"},
                        'pred_partner_vectors': {"out_dims": 3,
                                                 "out_dtype": "float32"}}
    if m_loss_scale == 0:
        names['outputs'].pop('pred_syn_indicator_out')
    if d_loss_scale == 0:
        names['outputs'].pop('pred_partner_vectors')

    with open(name + '_config.json', 'w') as f:
        json.dump(names, f)

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("Number of parameters:", total_parameters)
    print("Estimated size of parameters in GB:",
          float(total_parameters) * 8 / (1024 * 1024 * 1024))

    return output_shape