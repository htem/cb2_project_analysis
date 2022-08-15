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
from random_patterns import generate_patterns

SILENT_MODE = False
if "SILENT_MODE" in os.environ:
    SILENT_MODE = True


sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')

# from neurons import GranuleCell, MossyFiber, Simulation, generate_binary_patterns, add_noise_binary_patterns, add_noise_to_core_patterns, make_noisy_patterns_float
from neurons import Simulation
import analysis
# logging.basicConfig(level=logging.DEBUG)

def make_sim(
        input_graph,
        sim=None,
        ):
    if sim is None:
        removed = input_graph.remove_empty_mfs()
        print(f'Removed {len(removed)} mfs')
        sim = Simulation(
            input_graph=input_graph,
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

        # quick hack to make grcs with very small # of claws and high # of claws work
        if 2 not in calibrations[activation_level]:
            calibrations[activation_level][2] = calibrations[activation_level][3]-.5
            assert calibrations[activation_level][2] > 0
        if 7 not in calibrations[activation_level]:
            calibrations[activation_level][7] = calibrations[activation_level][6]+.5
            assert calibrations[activation_level][7] > 0
        # sim.print_grc_act_lv_scale(); asdf
    # print(f'Calibrations: {calibrations[activation_level]}')
    random.seed(seed)
    np.random.seed(seed)
    sim.set_activation_levels(calibrations[activation_level])
    print(f'New calibrations: {sim.get_activation_levels()}')

def analyze_dims(
        grc_acts,
        mf_acts,
        sim,
        activation_level,
        ref_output=None,
        print_output=True,
        ):

    mf_dim = 1
    mf_pop_corr = 0
    mf_act_lv = 0
    if mf_acts is not None:
        mf_dim, mf_pop_corr = analysis.get_dim_from_acts(mf_acts, ret_population_correlation=True)
        mf_act_lv = sum(mf_acts[0])/len(mf_acts[0])
    voi = analysis.get_average_metric2(grc_acts, ref_output, metric='voi')
    binary_similarity = analysis.get_average_metric2(grc_acts, ref_output, metric='binary_similarity')
    # print(grc_acts)
    # print(ref_output)
    hamming_distance = analysis.get_average_hamming_distance2(grc_acts, ref_output)
    # hamming_hist = analysis.get_hamming_distance_hist(grc_acts, ref_output)
    normalized_mse = analysis.get_normalized_mean_squared_distance(hamming_distance/len(grc_acts), f=activation_level)
    grc_dim, grc_pop_corr = analysis.get_dim_from_acts(grc_acts, ret_population_correlation=True)

    grc_act_lv = sum(grc_acts[0])/len(grc_acts[0])
    pct_grc = int(grc_dim*1000/sim.num_grcs)/10
    pct_mfs = int(grc_dim*1000/sim.num_mfs)/10
    pct_mf_dim = int(grc_dim*1000/mf_dim)/10
    if print_output:
        print(f'voi: {voi}')
        print(f'binary_similarity: {binary_similarity}')
        print(f'hamming_distance: {hamming_distance}')
        print(f'normalized_mse: {normalized_mse}')
        print(f'grc_act_lv: {grc_act_lv}')
        print(f'mf_act_lv: {mf_act_lv}')
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
    return res

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
        grc_mask[0:int(grc_pct*pattern_len)] = 1
        np.random.shuffle(grc_mask)

        grc_acts_sub = grc_acts[:, grc_mask]
        ref_output_sub = ref_output[grc_mask]

        all_res[grc_pct] = analyze_dims(
            ref_output=ref_output_sub,
            grc_acts=grc_acts_sub,
            mf_acts=None,
            sim=sim,
            activation_level=activation_level,
            print_output=print_output,
            )
    return all_res


def get_optimal_weights_change(act0, act1,
        valence_dir='01',
        irrelevant_bits='0',
        seed=0):
    weights = []
    assert len(act0) == len(act1)
    for a0, a1 in zip(act0, act1):
        if a0 < a1:
            weights.append(1 if valence_dir == '01' else 0)
        elif a0 > a1:
            weights.append(1 if valence_dir == '10' else 0)
        else:
            if irrelevant_bits == '0':
                weights.append(0)
            elif irrelevant_bits == '1':
                weights.append(1)
            elif irrelevant_bits == 'random':
                weights.append(random.randint(0, 1))
            elif irrelevant_bits == 'plus':
                # set weight where there is potential for even more difference in the valence_dir
                if valence_dir == '01':
                    weights.append(1 if a0 == 0 else 0)
                elif valence_dir == '10':
                    weights.append(1 if a0 == 1 else 0)
                else: assert 0
            else: assert 0

    assert len(act0) == len(weights)
    return weights


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

        ref_pattern = patterns[0]
        sim.evaluate([ref_pattern], no_random=True)
        ref_output = sim.get_grc_activities()[0]
        ref_output = copy.deepcopy(ref_output)
        sim.evaluate([redundant_pattern], no_random=True)
        # mf_acts = sim.get_mfs_activities()
        grc_acts = sim.get_grc_activities()
        ref_output1 = grc_acts[0]
        weights = make_weights_fn(ref_output, ref_output1, seed=seed)
        ref_sum0 = get_output_with_weights(ref_output, weights)
        ref_sum1 = get_output_with_weights(ref_output1, weights)
        ref_delta = ref_sum1 - ref_sum0
        # print(f'ref_output : {ref_output[0:60]}')
        # print(f'ref_output1: {ref_output1[0:60]}')
        # print(f'weights    : {np.array(weights)[0:60]}')
        # print(f'ref_sum0: {ref_sum0}')
        # print(f'ref_sum1: {ref_sum1}')
        # print(f'ref_delta: {ref_delta}')

        sim.evaluate(noisy_patterns, no_random=True)
        grc_acts = sim.get_grc_activities()
        hist = defaultdict(int)
        hist_raw = []
        hist_sum = defaultdict(int)
        for grc_act in grc_acts:
            # d = get_output_delta(ref_output, grc_act, weights)
            out_sum = get_output_with_weights(grc_act, weights)
            d = out_sum - ref_sum0
            hist[d] += 1
            hist_raw.append(d)
            hist_sum[out_sum] += 1
        if not SILENT_MODE:
            for k in sorted(hist.keys()):
                print(f'{k}: {hist[k]}')
        # print(hist_sum)
        res = {}
        res['ref_sum0'] = ref_sum0
        res['ref_sum1'] = ref_sum1
        res['ref_delta'] = ref_delta
        # res['hist_raw'] = hist_raw
        res['hist_sum2'] = dict(hist_sum)
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
        all_res[noise] = analyze_dims(
            ref_output=ref_output,
            grc_acts=grc_acts,
            mf_acts=mf_acts,
            sim=sim,
            activation_level=activation_level,
            print_output=print_output,
            )
    return all_res

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
