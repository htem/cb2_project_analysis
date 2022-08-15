import random
import copy
import logging
import sys
from collections import defaultdict
import itertools
from sim_lite import SimulationLite
import compress_pickle
import functools
import random_patterns
import time
import numpy as np
import os
import statistics
from random_patterns import generate_patterns, make_random_mask

SILENT_MODE = False
if "SILENT_MODE" in os.environ:
    SILENT_MODE = True

NO_DIM_SIM = False
np.set_printoptions(linewidth=180, edgeitems=30)

sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')

# from neurons import GranuleCell, MossyFiber, Simulation, generate_binary_patterns, add_noise_binary_patterns, add_noise_to_core_patterns, make_noisy_patterns_float
from neurons import Simulation
import analysis
from analysis import get_optimal_weights_change
# logging.basicConfig(level=logging.DEBUG)

def make_sim(
        input_graph,
        sim=None,
        per_bouton=False,
        ):
    if sim is None:
        # removed = input_graph.remove_empty_mfs()
        # print(f'Removed {len(removed)} mfs')
        sim = Simulation(
            input_graph=input_graph, per_bouton=per_bouton,
            )
    return sim

calibrations = defaultdict(lambda: None)
default_pattern_generator = random_patterns.generate_patterns

def calibrate_sim(
        sim,
        activation_level,
        pattern_generator=None,
        seed=1,
        ):
    global calibrations
    if calibrations[activation_level] is None:
        n_pattern = 512
        # n_pattern = 4096
        if pattern_generator is None:
            pattern_generator = default_pattern_generator
        patterns = pattern_generator(sim.num_mfs, count=n_pattern)
        sim.set_failure_rate(0, seed=seed)
        sim.evaluate(patterns, no_random=True,
            calibrate_activation_level=activation_level)
        calibrations[activation_level] = sim.get_activation_levels()

    random.seed(seed)
    np.random.seed(seed)

    try:
        sim.set_activation_levels(calibrations[activation_level])
    except:
        # calibrated values are missing some keys
        # rerunning from scratch
        calibrations[activation_level] = None
        return calibrate_sim(sim, activation_level, pattern_generator, seed)
    print(f'New calibrations: {sim.get_activation_levels()}')

grc_acts_bg = None

def calibrate_sim_with_bg_activity(
        sim,
        activation_level,
        pattern_generator=None,
        seed=None,
        ):
    global calibrations
    global grc_acts_bg
    if seed is not None:
        # seed = 5
        random.seed(seed)
        np.random.seed(seed)

    if calibrations[activation_level] is None:
        n_pattern = 512
        # n_pattern = 4096
        if pattern_generator is None:
            pattern_generator = default_pattern_generator
        patterns = pattern_generator(sim.num_mfs, count=n_pattern, seed=seed)
        sim.set_failure_rate(0, seed=seed)
        sim.evaluate(patterns, no_random=True,
            calibrate_activation_level=activation_level)
        calibrations[activation_level] = sim.get_activation_levels()

    try:
        sim.set_activation_levels(calibrations[activation_level])
    except:
        # calibrated values are missing some keys
        # rerunning from scratch
        calibrations[activation_level] = None
        return calibrate_sim(sim, activation_level, pattern_generator, seed)
    print(f'New calibrations: {sim.get_activation_levels()}')

    # get background act
    patterns = pattern_generator(sim.num_mfs, count=n_pattern)
    sim.evaluate(patterns, no_random=True)
    grc_acts_bg = copy.deepcopy(sim.get_grc_activities())

def analyze_dims(
        grc_acts,
        mf_acts,
        sim,
        activation_level,
        ref_output=None,
        print_output=True,
        ):

    mf_dim = 1
    mf_pop_corr = 1
    mf_act_lv = 1
    if not NO_DIM_SIM and mf_acts is not None:
        mf_dim, mf_pop_corr = analysis.get_dim_from_acts(mf_acts, ret_population_correlation=True)
        mf_act_lv = sum(mf_acts[0])/len(mf_acts[0])

    voi = analysis.get_average_metric2(grc_acts, ref_output, metric='voi')
    binary_similarity = analysis.get_average_metric2(grc_acts, ref_output, metric='binary_similarity')
    hamming_distance = analysis.get_average_hamming_distance2(grc_acts, ref_output)
    normalized_mse = analysis.get_normalized_mean_squared_distance(hamming_distance/len(grc_acts), f=activation_level)
    # dir_distance_01 = analysis.get_average_metric2(grc_acts, ref_output, metric='dir_distance_01')
    # dir_distance_10 = analysis.get_average_metric2(grc_acts, ref_output, metric='dir_distance_10')
    # print(mf_acts); asdf
    # print(ref_output)
    # print(grc_acts[0:20]); asdf
    # print(grc_acts[0:20])

    grc_dim = 1
    grc_pop_corr = 1
    if not NO_DIM_SIM:
        grc_dim, grc_pop_corr = analysis.get_dim_from_acts(grc_acts, ret_population_correlation=True)

    grc_mean, grc_stdev = analysis.get_output_deviation(grc_acts)
    grc_stdev_pct = grc_stdev / grc_mean

    # grc_act_lv = sum(grc_acts[0])/len(grc_acts[0])
    grc_act_lv = grc_mean/len(grc_acts[0])
    pct_grc = int(grc_dim*1000/sim.num_grcs)/10
    pct_mfs = int(grc_dim*1000/sim.num_mfs)/10
    pct_mf_dim = int(grc_dim*1000/mf_dim)/10
    if print_output:
        # print(f'voi: {voi}')
        # print(f'binary_similarity: {binary_similarity}')
        print(f'hamming_distance: {hamming_distance}')
        # print(f'normalized_mse: {normalized_mse}')
        print(f'grc_act_lv: {grc_act_lv}')
        if not NO_DIM_SIM:
            print(f'mf_act_lv: {mf_act_lv}')
            print(f'grc_pop_corr: {grc_pop_corr} ({1/grc_pop_corr})')
            print(f'mf_pop_corr: {mf_pop_corr} ({1/mf_pop_corr})')
            print(f'Dim MFs: {mf_dim}')
            print(f'Dim GRCs: {grc_dim}')
        print(f'grc_mean: {grc_mean}')
        print(f'grc_stdev: {grc_stdev}')
        print(f'grc_stdev_pct: {grc_stdev_pct}')
        # print(f'dir_distance_01: {dir_distance_01}')
        # print(f'dir_distance_10: {dir_distance_10}')
        # print(f'    = {pct_mfs}% of MFs')
        # print(f'    = {pct_mf_dim}% of MF dim')
        # print(f'    = {pct_grc}% of GrCs')
        print()
    res = {}
    res['voi'] = voi
    res['binary_similarity'] = binary_similarity
    res['hamming_distance'] = hamming_distance
    res['normalized_mse'] = normalized_mse
    res['grc_pop_corr'] = grc_pop_corr
    res['mf_dim'] = mf_dim
    res['mf_pop_corr'] = mf_pop_corr
    res['grc_dim'] = grc_dim
    res['pct_grc'] = pct_grc
    res['pct_mfs'] = pct_mfs
    res['pct_mf_dim'] = pct_mf_dim
    res['num_grcs'] = sim.num_grcs
    res['num_mfs'] = sim.num_mfs 
    res['grc_mean'] = grc_mean
    res['grc_stdev'] = grc_stdev
    res['grc_stdev_pct'] = grc_stdev_pct
    # res['dir_distance_01'] = dir_distance_01
    return res

def add_synapse_fail_rate(
        fail_rate,
        act,
        acts=None,
        ):
    if fail_rate is None or fail_rate == 1.0:
        return act, acts
    assert fail_rate >= 0 and fail_rate <= 1
    pattern_len = len(act)
    fail_mask = np.ones(pattern_len, dtype=np.bool_)
    fail_mask[0:int((1-fail_rate)*pattern_len+.5)] = 0
    np.random.shuffle(fail_mask)
    act &= fail_mask
    for i, o in enumerate(acts):
        np.random.shuffle(fail_mask)
        acts[i] = o & fail_mask
    return act, acts

def test_across_grc_pcts(
        sim,
        activation_level,
        grc_pcts,
        # noise_probs,
        # failure_rates,
        pattern_generator=None,
        noise_generator=None,
        seed=0,
        test_len=512,
        print_output=True,
        synapse_fail_rate=None,
        grc_scale=1,
        ):
    noise = 1.0
    if pattern_generator is None:
        pattern_generator = default_pattern_generator
    patterns = pattern_generator(sim.num_mfs, count=1, seed=seed)
    test_patterns = [patterns[0]]
    calibrate_sim(sim, activation_level,
        pattern_generator=pattern_generator,
        )
    redundant_patterns = noise_generator(test_patterns, prob=noise, n=test_len, seed=seed)
    ref_pattern = patterns[0]
    sim.evaluate([ref_pattern], no_random=True)
    ref_output = sim.get_grc_activities()[0]
    ref_output = copy.deepcopy(ref_output)
    sim.evaluate(redundant_patterns, no_random=True)
    # mf_acts = sim.get_mfs_activities()
    grc_acts = sim.get_grc_activities()
    all_res = {}
    for grc_pct in grc_pcts:
        print(f'grc_pct={grc_pct}')
        pattern_len = len(ref_output)
        grc_mask = np.zeros(pattern_len, dtype=np.bool_)
        grc_mask[0:int(grc_pct*pattern_len*grc_scale+.5)] = 1
        np.random.shuffle(grc_mask)
        grc_acts_sub = grc_acts[:, grc_mask]
        ref_output_sub = ref_output[grc_mask]
        add_synapse_fail_rate(synapse_fail_rate, act=ref_output_sub, acts=grc_acts_sub)
        all_res[grc_pct] = analyze_dims(
            ref_output=ref_output_sub,
            grc_acts=grc_acts_sub,
            mf_acts=None,
            sim=sim,
            activation_level=activation_level,
            print_output=print_output,
            )
    return all_res

def test_across_grc_pcts_selective(
        sim,
        activation_level,
        grc_pcts,
        # noise_probs,
        # failure_rates,
        pattern_generator=None,
        noise_generator=None,
        seed=0,
        test_len=512,
        print_output=True,
        synapse_fail_rate=None,
        grc_scale=1,
        grc_drop=1,
        selective=0,
        ):
    noise = 1.0
    if pattern_generator is None:
        pattern_generator = default_pattern_generator
    patterns = pattern_generator(sim.num_mfs, count=1, seed=seed)
    test_patterns = [patterns[0]]
    calibrate_sim(sim, activation_level,
        pattern_generator=pattern_generator,
        )
    redundant_patterns = noise_generator(test_patterns, prob=noise, n=test_len, seed=seed)
    ref_pattern = patterns[0]
    sim.evaluate([ref_pattern], no_random=True)
    ref_output = sim.get_grc_activities()[0]
    ref_output = copy.deepcopy(ref_output)
    sim.evaluate(redundant_patterns, no_random=True)
    # mf_acts = sim.get_mfs_activities()
    grc_acts = sim.get_grc_activities()
    pattern_len = len(ref_output)

    # estimate output toggles wrt the output to filter out noise
    # randomly assign 1/2 of patterns to a different output
    for i, p in enumerate(redundant_patterns):
        redundant_patterns[i] = [k for k in p]
        redundant_patterns[i][1] = random.randint(0, 1)

    # sample 4096 combinations and build histogram of SNR
    patterns0 = [p for p, v in zip(grc_acts, redundant_patterns)
                 if v[1] == 0]
    patterns1 = [p for p, v in zip(grc_acts, redundant_patterns)
                 if v[1] == 1]

    hist = defaultdict(int)
    for i in range(4096):
        p0 = random.choice(patterns0)
        p1 = random.choice(patterns1)
        for j, (b0, b1) in enumerate(zip(p0, p1)):
            if b1 > b0:
                hist[j] += 1
    hist_list = [(k, v) for k, v in hist.items()]
    hist_list.sort(key=lambda x: x[1], reverse=True)

    # hist_list_rev = copy.deepcopy(hist_list)
    # hist_list_rev.sort(key=lambda x: x[1], reverse=False)
    drop_grcs = int(sim.num_grcs*(1.0-grc_drop))
    # print(drop_grcs)
    drop_idx = [k for k, v in hist_list][drop_grcs:]
    drop_mask = np.zeros(pattern_len, dtype=np.bool_)
    drop_mask[drop_idx] = 1
    # print(drop_mask)
    drop_mask = np.logical_not(drop_mask)
    # print(drop_mask)
    # print(patterns0)
    # patterns0 = patterns0[:, drop_mask]
    # patterns1 = patterns1[:, drop_mask]
    for i, p in enumerate(patterns0):
        patterns0[i] = p[drop_mask]
    for i, p in enumerate(patterns1):
        patterns1[i] = p[drop_mask]
    ref_output = ref_output[drop_mask]
    pattern_len = len(ref_output)
    print(len(patterns0[0]))
    print(pattern_len)
    grc_acts = grc_acts[:, drop_mask]

    hist = defaultdict(int)
    for i in range(4096):
        p0 = random.choice(patterns0)
        p1 = random.choice(patterns1)
        for j, (b0, b1) in enumerate(zip(p0, p1)):
            if b1 > b0:
                hist[j] += 1
    hist_list = [(k, v) for k, v in hist.items()]
    hist_list.sort(key=lambda x: x[1], reverse=True)

    all_res = {}
    for grc_pct in grc_pcts:
        print(f'grc_pct={grc_pct}')
        grc_mask = np.zeros(pattern_len, dtype=np.bool_)

        if not selective:
            grc_mask[0:int(grc_pct*pattern_len*grc_scale+.5)] = 1
            np.random.shuffle(grc_mask)
        else:
            keep_grcs = int(int(grc_pct*pattern_len*grc_scale+.5))
            keep_idx = [k for k, v in hist_list][0:keep_grcs]
            grc_mask[keep_idx] = 1

        grc_acts_sub = grc_acts[:, grc_mask]
        ref_output_sub = ref_output[grc_mask]

        add_synapse_fail_rate(synapse_fail_rate, act=ref_output_sub, acts=grc_acts_sub)
        all_res[grc_pct] = analyze_dims(
            ref_output=ref_output_sub,
            grc_acts=grc_acts_sub,
            mf_acts=None,
            sim=sim,
            activation_level=activation_level,
            print_output=print_output,
            )
    return all_res

def test_across_synapse_fail_rates(
        sim,
        activation_level,
        synapse_fail_rates,
        # noise_probs,
        # failure_rates,
        pattern_generator=None,
        noise_generator=None,
        seed=0,
        test_len=512,
        print_output=True,
        # synapse_fail_rate=None,
        grc_scale=1,
        ):
    noise = 1.0
    if pattern_generator is None:
        pattern_generator = default_pattern_generator
    patterns = pattern_generator(sim.num_mfs, count=1, seed=seed)
    test_patterns = [patterns[0]]
    calibrate_sim(sim, activation_level,
        pattern_generator=pattern_generator,
        )
    redundant_patterns = noise_generator(test_patterns, prob=noise, n=test_len, seed=seed)
    ref_pattern = patterns[0]
    sim.evaluate([ref_pattern], no_random=True)
    ref_output = sim.get_grc_activities()[0]
    ref_output = copy.deepcopy(ref_output)
    sim.evaluate(redundant_patterns, no_random=True)
    # mf_acts = sim.get_mfs_activities()
    grc_acts = sim.get_grc_activities()
    all_res = {}
    for synapse_fail_rate in synapse_fail_rates:
        print(f'synapse_fail_rate={synapse_fail_rate}')
        pattern_len = len(ref_output)
        grc_mask = np.zeros(pattern_len, dtype=np.bool_)
        grc_mask[0:int(pattern_len*grc_scale+.5)] = 1
        np.random.shuffle(grc_mask)
        grc_acts_sub = grc_acts[:, grc_mask]
        ref_output_sub = ref_output[grc_mask]
        if synapse_fail_rate is not None:
            if synapse_fail_rate != 1.0:
                assert synapse_fail_rate >= 0 and synapse_fail_rate <= 1
                pattern_len = len(ref_output_sub)
                fail_mask = np.ones(pattern_len, dtype=np.bool_)
                fail_mask[0:int(synapse_fail_rate*pattern_len+.5)] = 0
                np.random.shuffle(fail_mask)
                ref_output_sub &= fail_mask
                for o in grc_acts_sub:
                    np.random.shuffle(fail_mask)
                    o &= fail_mask
        all_res[synapse_fail_rate] = analyze_dims(
            ref_output=ref_output_sub,
            grc_acts=grc_acts_sub,
            mf_acts=None,
            sim=sim,
            activation_level=activation_level,
            print_output=print_output,
            )
    return all_res



# def get_optimal_weights_same(act0, act1,
#         valence_dir='0',
#         seed=0):
#     weights = []
#     assert len(act0) == len(act1)
#     weight_len = len(act0)
#     num_zeroes = int(weight_len/2)
#     num_ones = num_zeroes
#     count_zeroes = 0
#     count_ones = 0
#     for a0, a1 in zip(act0, act1):
#         if a0 == a1:
#             if valence_dir == '0':
#                 if a0 == 1:
#                     weights.append(0)
#                     count_zeroes += 1
#                 else:
#                     weights.append(1)
#                     count_ones += 1
#             elif valence_dir == '1':
#                 if a0 == 1:
#                     weights.append(1)
#                     count_ones += 1
#                 else:
#                     weights.append(0)
#                     count_zeroes += 1
#             else:
#                 assert False
#         else:
#             weights.append(2)
#     assert len(act0) == len(weights)
#     print(f'sum(act0): {sum(act0)}')
#     print(f'sum(act1): {sum(act1)}')
#     print(f'sum(act0&act1): {sum(act0&act1)}')
#     print(f'count_ones: {count_ones}')
#     print(f'count_zeroes: {count_zeroes}')
#     print(f'len(weights): {len(weights)}')
#     asdf
#     num_ones -= count_ones
#     num_zeroes -= count_zeroes
#     prob_ones = num_ones / (num_ones+num_zeroes)
#     assert prob_ones <= 1.0
#     for i, w in enumerate(weights):
#         if w == 2:
#             weights[i] = random.random() < prob_ones
#     return weights

def get_optimal_weights_same(act0, act1,
        valence_dir='0',
        seed=0):
    weights = []
    assert len(act0) == len(act1)
    weight_len = len(act0)

    pass0_count = 0
    pass1_idx = []
    for i, (a0, a1) in enumerate(zip(act0, act1)):
        if a0 == a1:
            if a0 == 1:
                weights.append(0 if valence_dir == '0' else 1)
                pass0_count += 1
            else:
                weights.append(2)
                pass1_idx.append(i)
        else:
            weights.append(3)
    assert len(act0) == len(weights)
    assert pass0_count <= int(weight_len/2)  # if act_level >= .5, need another algorithm

    random.shuffle(pass1_idx)
    for i in range(pass0_count):
        idx = pass1_idx[i]
        assert weights[idx] == 2
        weights[idx] = 1 if valence_dir == '0' else 0

    # number of ones and zeros should be balanced now
    # randomly assign the rest of weights to either 0 or 1
    for i, w in enumerate(weights):
        if w == 2 or w == 3:
            weights[i] = 1 if random.random() < .5 else 0

    assert 2 not in weights
    assert 3 not in weights

    # print(f'pass0_count: {pass0_count}')
    # print(f'len(pass1_idx): {len(pass1_idx)}')
    # print(f'sum(act0): {sum(act0)}')
    # print(f'sum(act1): {sum(act1)}')
    # print(f'sum(act0&act1): {sum(act0&act1)}')
    # print(f'len(weights): {len(weights)}')
    # print(f'sum(weights): {sum(weights)}')
    # asdf
    return weights


def get_output_delta(act0, act1, weights):
    out0 = 0
    out1 = 0
    for a0, a1, w in zip(act0, act1, weights):
        if w:
            out0 += a0
            out1 += a1
    return out1 - out0

def get_output_with_weights(act0, weights):
    out0 = 0
    for a0, w in zip(act0, weights):
        if w:
            out0 += a0
    return out0

def get_sum_hist(grc_acts, weights, ref_sum0):
    hist_sum = defaultdict(int)
    hist_delta = defaultdict(int)
    for grc_act in grc_acts:
        out_sum = get_output_with_weights(grc_act, weights)
        hist_sum[out_sum] += 1
        d = out_sum - ref_sum0
        hist_delta[d] += 1
    return hist_sum, hist_delta


def test_consistency_across_variations(
        sim,
        activation_level,
        variation_sizes,
        make_weights_fn,
        noise_scaling=None,
        noise_level=None,
        # failure_rates,
        pattern_generator=None,
        variation_generator=None,
        noise_generator=None,
        seed=0,
        test_len=512,
        print_output=True,
        ):
    if pattern_generator is None:
        pattern_generator = default_pattern_generator

    assert noise_scaling is not None or noise_level is not None
    assert noise_scaling is None or noise_level is None

    patterns = pattern_generator(sim.num_mfs, count=1, seed=seed)
    test_pattern = patterns[0]
    calibrate_sim(sim, activation_level,
        pattern_generator=pattern_generator,
        )
    all_res = {}
    for variation_size in variation_sizes:
        print(f'variation_size={variation_size}')

        ref_pattern = patterns[0]

        redundant_patterns = variation_generator([test_pattern], prob=variation_size, n=2, seed=seed)
        redundant_pattern = redundant_patterns[1]

        noise_mask = (redundant_pattern[0]^test_pattern[0])
        noise_mask = 1-noise_mask
        # print(f'test_pattern : {test_pattern[0:60]}')
        # print(f'redundant_pattern: {redundant_pattern[0:60]}')
        # print(f'noise_mask    : {np.array(noise_mask)[0:60]}')
        noise_prob = noise_level
        if noise_scaling:
            noise_prob = variation_size*noise_scaling

        noisy_patterns = noise_generator(
            [redundant_pattern], prob=noise_prob, n=test_len, seed=seed,
            noise_mask=noise_mask,
            )
        noisy_ref_patterns = noise_generator(
            [ref_pattern], prob=noise_prob, n=int(test_len/4), seed=seed,
            noise_mask=noise_mask,
            )

        random_patterns = noise_generator(
            [ref_pattern], prob=1, n=int(test_len/8), seed=seed,
            # noise_mask=noise_mask,
            )

        random_masked_patterns = noise_generator(
            [ref_pattern], prob=1, n=int(test_len/8), seed=seed,
            noise_mask=noise_mask,
            )

        sim.evaluate([ref_pattern], no_random=True)
        ref_output = sim.get_grc_activities()[0]
        ref_output = copy.deepcopy(ref_output)
        sim.evaluate([redundant_pattern], no_random=True)
        grc_acts = sim.get_grc_activities()
        ref_output1 = grc_acts[0]
        weights = make_weights_fn(ref_output, ref_output1, seed=seed)
        ref_sum0 = get_output_with_weights(ref_output, weights)
        ref_sum1 = get_output_with_weights(ref_output1, weights)
        ref_delta = ref_sum1 - ref_sum0
        # print(f'ref_output : {ref_output[0:60]}')
        # print(f'ref_output1: {ref_output1[0:60]}')
        # print(f'weights    : {np.array(weights)[0:60]}')

        sim.evaluate(noisy_patterns, no_random=True)
        sum_hist, delta_hist = get_sum_hist(sim.get_grc_activities(), weights, ref_sum0)

        sim.evaluate(noisy_ref_patterns, no_random=True)
        noisy_ref_sum_hist, noisy_ref_delta_hist = get_sum_hist(sim.get_grc_activities(), weights, ref_sum0)

        sim.evaluate(random_patterns, no_random=True)
        random_sum_hist, _ = get_sum_hist(sim.get_grc_activities(), weights, ref_sum0)

        sim.evaluate(random_masked_patterns, no_random=True)
        random_masked_sum_hist, _ = get_sum_hist(sim.get_grc_activities(), weights, ref_sum0)


        if not SILENT_MODE:
            print(f'ref_sum0: {ref_sum0}')
            print(f'ref_sum1: {ref_sum1}')
            print(f'ref_delta: {ref_delta}')
            # for k in sorted(hist.keys()):
                # print(f'{k}: {hist[k]}')
            print('noisy_ref_delta_hist')
            for k in sorted(noisy_ref_delta_hist.keys()):
                print(f'{k}: {noisy_ref_delta_hist[k]}')
            print('delta_hist')
            for k in sorted(delta_hist.keys()):
                print(f'{k}: {delta_hist[k]}')
            print('random_sum_hist')
            for k in sorted(random_sum_hist.keys()):
                print(f'{k}: {random_sum_hist[k]}')
            print('random_masked_sum_hist')
            for k in sorted(random_masked_sum_hist.keys()):
                print(f'{k}: {random_masked_sum_hist[k]}')
        # print(hist_sum)
        res = {}
        res['ref_sum0'] = ref_sum0
        res['ref_sum1'] = ref_sum1
        res['ref_delta'] = ref_delta
        # res['hist_raw'] = hist_raw
        res['sum_hist'] = dict(sum_hist)
        res['noisy_ref_sum_hist'] = dict(noisy_ref_sum_hist)
        res['random_sum_hist'] = dict(random_sum_hist)
        res['random_masked_sum_hist'] = dict(random_masked_sum_hist)
        # res['hist'] = hist
        all_res[variation_size] = res
    return all_res


def test_across_noise(
        sim,
        activation_level,
        noise_probs,
        # failure_rates,
        pattern_generator=None,
        noise_generator=None,
        seed=0,
        test_len=512,
        print_output=True,
        grc_pct=None,
        synapse_fail_rate=None,
        grc_pct_learned=False,
        ):
    if pattern_generator is None:
        pattern_generator = default_pattern_generator
    patterns = pattern_generator(sim.num_mfs, count=1, seed=seed)
    test_patterns = [patterns[0]]
    calibrate_sim(sim, activation_level,
        pattern_generator=pattern_generator,
        )
    all_res = {}
    for noise in noise_probs:
        print(f'noise={noise}')
        redundant_patterns = noise_generator(test_patterns, prob=noise, n=test_len, seed=seed)
        ref_pattern = patterns[0]
        sim.evaluate([ref_pattern], no_random=True)
        ref_output = sim.get_grc_activities()[0]
        ref_output = copy.deepcopy(ref_output)
        sim.evaluate(redundant_patterns, no_random=True)
        mf_acts = sim.get_mfs_activities()
        grc_acts = sim.get_grc_activities()

        if grc_pct is not None:
            pattern_len = len(ref_output)
            grc_mask = np.zeros(pattern_len, dtype=np.bool_)
            keep_n = int(grc_pct*pattern_len+.5)
            if grc_pct_learned:
                grc_mask = get_learned_mask2(ref_output, grc_acts, grc_mask, grc_pct)
            else:
                # drop random grcs            
                grc_mask[0:keep_n] = 1
                np.random.shuffle(grc_mask)
            grc_acts = grc_acts[:, grc_mask]
            ref_output = ref_output[grc_mask]

        # print(ref_output)
        # print(grc_acts)
        ref_output, grc_acts = add_synapse_fail_rate(synapse_fail_rate, act=ref_output, acts=grc_acts)
        # print(ref_output); asdf
        # print(grc_acts); asdf

        all_res[noise] = analyze_dims(
            ref_output=ref_output,
            grc_acts=grc_acts,
            mf_acts=mf_acts,
            sim=sim,
            activation_level=activation_level,
            print_output=print_output,
            )
    return all_res

def test_snr_across_noise(
        sim,
        activation_level,
        loop_vals,
        pattern_generator=None,
        noise_generator=None,
        seed=None,
        print_output=True,
        # input_noise_pct=0,
        num_patterns=1,
        test_mult=10,
        noise_during_training=False,
        pattern_size_pct=100.0,
        unrelated_mfs_pct=0.0,
        synapse_pct=50.0,
        ):

    assert pattern_size_pct + unrelated_mfs_pct <= 100.0

    if pattern_generator is None:
        pattern_generator = default_pattern_generator
    calibrate_sim_with_bg_activity(sim, activation_level,
        pattern_generator=pattern_generator, seed=seed)

    # make train/test patterns and masks
    patterns0 = pattern_generator(sim.num_mfs, count=num_patterns, seed=seed)
    # make unrelated MF mask. this denotes where randomization always happens
    unrelated_mfs_mask = make_random_mask(sim.num_mfs, (unrelated_mfs_pct)/100)
    # make irrelevant mask to denote randomization
    pattern_masks0 = []
    for i in range(len(patterns0)):
        pattern_masks0.append(make_random_mask(
            sim.num_mfs, (100-pattern_size_pct)/100, additional_mask=unrelated_mfs_mask))

    rets = {}
    for input_noise_pct in loop_vals:
        print(f'\ninput_noise_pct: {input_noise_pct}')

        train_noise = input_noise_pct if noise_during_training else 0
        patterns_train = noise_generator(
            patterns0, input_noise_pct=train_noise, test_mult=test_mult,
            seed=seed, pattern_masks=pattern_masks0)
        patterns_test = noise_generator(
            patterns0, input_noise_pct=input_noise_pct, test_mult=test_mult,
            seed=None, pattern_masks=pattern_masks0)

        # run mf->grc
        sim.evaluate(patterns_train, no_random=True)
        grc_acts_train = copy.deepcopy(sim.get_grc_activities())
        sim.evaluate(patterns_test, no_random=True)
        grc_acts_test = copy.deepcopy(sim.get_grc_activities())

        # calculate optimal weights using training patterns
        # TODO: only for positive patterns for now
        grc_scores = [0]*sim.num_grcs
        for grc_act in grc_acts_train:
            for i, b in enumerate(grc_act):
                if b:
                    grc_scores[i] += 1

        synapse_weights = get_optimal_weights2(grc_scores, synapse_pct)

        # calculate signals for the test patterns
        signals = []
        for test_pattern in grc_acts_test:
            signals.append(calc_signal(test_pattern, synapse_weights))
        signal_size = sum(signals)/len(signals)
        signal_std = statistics.stdev(signals)

        # calculate bg mean and std
        bg_signals = []
        for bg_pattern in grc_acts_bg:
            bg_signals.append(calc_signal(bg_pattern, synapse_weights))
        bg_mean = sum(bg_signals)/len(bg_signals)
        bg_std = statistics.stdev(bg_signals)
        snr = (signal_size - bg_mean) / bg_std

        print(f'snr: {snr}')
        print(f'bg_mean: {bg_mean}')
        print(f'bg_std: {bg_std}')
        print(f'signal_size: {signal_size}')
        print(f'signal_std: {signal_std}')

        ret = {
            'snr': snr,
            'bg_mean': bg_mean,
            'bg_std': bg_std,
            'signal_size': signal_size,
            'signal_std': signal_std,
        }

        rets[input_noise_pct] = ret

    return rets

def test_across_failure(
        input_graph,
        activation_level,
        # noise_probs,
        failure_rates,
        seed=0,
        test_len=512,
        # core_noise=False,
        print_output=True,
        # scaled_noise=False,
        signal_mask=None,
        sim=None,
        ):
    random.seed(seed)
    if sim is None:
        asdf; # should be per bouton?
        removed = input_graph.remove_empty_mfs()
        print(f'Removed {len(removed)} mfs')
        sim = Simulation(
            input_graph=input_graph,
            )
    n_pattern = 1024*4  # 309
    # n_pattern = 512  # 309
    patterns = sim.generate_patterns(count=n_pattern)
    # print(patterns); asdf
    sim.set_failure_rate(0, seed=seed)
    sim.evaluate(patterns, no_random=True,
        calibrate_activation_level=activation_level)
    test_patterns = [patterns[0]]
    all_res = {}
    for failure_rate in failure_rates:
        print(f'failure_rate={failure_rate}')
        redundant_patterns = make_noisy_patterns_float(test_patterns, prob=1.0, n=test_len, seed=seed, signal_mask=signal_mask)
        # redundant_patterns = add_noise_to_core_patterns(test_patterns, prob=noise, n=2)
        # for p in redundant_patterns[0:2]:
            # print(p[0][0:20])
        # asdf
        # print(redundant_patterns)
        ref_pattern = patterns[0]
        sim.set_failure_rate(failure_rate, seed=seed)
        sim.evaluate([ref_pattern], no_random=True)
        ref_output = sim.get_grc_activities()[0]
        ref_output = copy.deepcopy(ref_output)

        sim.evaluate(redundant_patterns, no_random=True)
        mf_acts = sim.get_mfs_activities()
        # for act in mf_acts[0:2]:
        #     print(act[0:18])
        mf_dim, mf_pop_corr = analysis.get_dim_from_acts(mf_acts, ret_population_correlation=True)
        grc_acts = sim.get_grc_activities()
        # for act in grc_acts[0:2]:
        #     print(act[0:37])
        # print(f'ref_output:\n{ref_output[0:37]}')
        # print(f'ref_pattern:\n{ref_pattern[0][0:37]}')
        # print(f'redundant_patterns:\n{redundant_patterns[0][0][0:37]}')
        # print(f'test_patterns:\n{test_patterns[0][0][0:37]}')
        # asdf
        voi = analysis.get_average_metric2(grc_acts, ref_output, metric='voi')
        binary_similarity = analysis.get_average_metric2(grc_acts, ref_output, metric='binary_similarity')
        hamming_distance = analysis.get_average_hamming_distance2(grc_acts, ref_output)
        normalized_mse = analysis.get_normalized_mean_squared_distance(hamming_distance/len(grc_acts), f=activation_level)
        grc_dim, grc_pop_corr = analysis.get_dim_from_acts(grc_acts, ret_population_correlation=True)
        pct_grc = int(grc_dim*1000/sim.num_grcs)/10
        pct_mfs = int(grc_dim*1000/sim.num_mfs)/10
        pct_mf_dim = int(grc_dim*1000/mf_dim)/10
        if print_output:
            # print(f'voi: {voi}')
            # print(f'binary_similarity: {binary_similarity}')
            # print(f'hamming_distance: {hamming_distance}')
            # print(f'normalized_mse: {normalized_mse}')
            print(f'grc_pop_corr: {grc_pop_corr}')
            print(f'mf_pop_corr: {mf_pop_corr}')
            print(f'Dim MFs: {mf_dim}')
            print(f'Dim GRCs: {grc_dim}')
            print(f'    = {pct_mfs}% of MFs')
            print(f'    = {pct_mf_dim}% of MF dim')
            print(f'    = {pct_grc}% of GrCs')
        res = {}
        res['voi'] = voi
        res['binary_similarity'] = binary_similarity
        res['hamming_distance'] = hamming_distance
        res['normalized_mse'] = normalized_mse
        res['grc_pop_corr'] = grc_pop_corr
        res['mf_dim'] = mf_dim
        res['mf_pop_corr'] = mf_pop_corr
        res['grc_dim'] = grc_dim
        res['pct_grc'] = pct_grc
        res['pct_mfs'] = pct_mfs
        res['pct_mf_dim'] = pct_mf_dim
        res['num_grcs'] = sim.num_grcs
        res['num_mfs'] = sim.num_mfs
        all_res[failure_rate] = res
    return all_res


def get_learned_mask(ref_output, grc_acts, grc_mask, grc_pct):
    pattern_len = len(ref_output)
    keep_n = int(grc_pct*pattern_len+.5)
    per_grc_sum = np.sum(grc_acts, axis=0)
    per_grc_f = per_grc_sum / len(grc_acts)
    grc_fs = [(i, f) for i, f in enumerate(per_grc_f)]
    # remove non-active grcs
    grc_fs = [(i, f) for i, f in enumerate(per_grc_f) if (f > 0.01 and f < 0.99)]
    # get the center keep_n grcs
    grc_fs.sort(key=lambda x: x[1])
    # print(grc_fs)
    # for k, v in grc_fs:
    #     print(v)
    # asdf
    remove_n = max(0, len(grc_fs) - keep_n)
    grc_fs = grc_fs[int(remove_n/2):int(-remove_n/2)-1]
    # if keep_n >= len(grc_fs):
    #     keep_start = 0
    #     keep_end = len(grc_fs)
    # else:
    #     keep_start = int(keep_n/2+.5)
    #     keep_end = int(keep_n)
    for i, _ in grc_fs:
        grc_mask[i] = 1
    # for k, v in grc_fs:
    #     print(v)
    # print(grc_mask); asdf
    # print(sum(grc_mask))
    return grc_mask


def get_learned_mask2(ref_output, grc_acts, grc_mask, grc_pct):

    pattern_len = len(ref_output)
    keep_n = int(grc_pct*pattern_len+.5)
    change_count = defaultdict(int)
    for act in grc_acts:
        for i, (a, b), in enumerate(zip(act, ref_output)):
            if a != b:
                change_count[i] += 1
    change_count = [(k, v) for k, v in change_count.items()]
    change_count.sort(key=lambda x: x[1], reverse=True)
    # for k, v in change_count:
        # print(v)
    for k, v in change_count[0:keep_n]:
        grc_mask[k] = 1
    return grc_mask


def get_optimal_weights2(grc_scores, synapse_pct):
    '''Make synapse where grc_scores are highest'''
    scores = list(enumerate(grc_scores))
    random.shuffle(scores)  # to randomly break ties
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    # print(scores)
    num_synapse = int(synapse_pct/100*len(scores))
    # print(num_synapse); asdf
    assert num_synapse > 0
    weights = [0]*len(scores)
    # num_synapse = 10
    for i, s in scores:
        weights[i] = 1
        num_synapse -= 1
        if num_synapse == 0:
            break
    return weights


def calc_signal(pattern, weights):
    s = 0
    assert len(pattern) == len(weights)
    for p, w in zip(pattern, weights):
        if p and w:
            s += 1
    return s
