import random
import copy
import logging
import sys
from collections import defaultdict
import itertools
from sim_lite import SimulationLite
import compress_pickle

sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')

from neurons import GranuleCell, MossyFiber, Simulation, generate_binary_patterns, add_noise_binary_patterns, add_noise_to_core_patterns, make_noisy_patterns_float
import analysis
# logging.basicConfig(level=logging.DEBUG)

def test_across_noise(
        input_graph,
        activation_level,
        noise_probs,
        # failure_rates,
        seed=0,
        test_len=512,
        core_noise=False,
        print_output=True,
        scaled_noise=False,
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
    for noise in noise_probs:
        print(f'noise={noise}')
        if core_noise:
            redundant_patterns = add_noise_to_core_patterns(test_patterns, prob=noise, n=test_len, seed=seed)
        else:
            redundant_patterns = make_noisy_patterns_float(test_patterns, prob=noise, n=test_len, seed=seed, scaled_noise=scaled_noise, signal_mask=signal_mask)
        # redundant_patterns = add_noise_to_core_patterns(test_patterns, prob=noise, n=2)
        # for p in redundant_patterns[0:2]:
            # print(p[0][0:20])
        # asdf
        ref_pattern = patterns[0]
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
        all_res[noise] = res
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
