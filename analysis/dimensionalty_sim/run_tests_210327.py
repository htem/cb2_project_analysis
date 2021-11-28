import random
import copy
import logging
import sys
from collections import defaultdict
import itertools
from sim_lite import SimulationLite
import compress_pickle

sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')

from neurons import GranuleCell, MossyFiber, Simulation, generate_binary_patterns, add_noise_binary_patterns, add_noise_to_core_patterns
import analysis
# logging.basicConfig(level=logging.DEBUG)

def test_similarity_by_activation_level3(
        input_graph,
        noise_prob=None,
        activation_levels=None,
        seed=0,
        print_output=False,
        test_len=128,
        sim=None,
        ):
    random.seed(seed)
    if sim is None:
        input_graph = copy.deepcopy(input_graph)
        removed = input_graph.remove_empty_mfs()
        sim = Simulation(
            input_graph=input_graph,
            )
    n_pattern = 1024*4  # 309
    patterns = sim.generate_patterns(
        count=n_pattern,
        # type='gaussian',
        )
    sim.set_failure_rate(0, seed=seed)
    # if noise_probs is None:
    #     noise_probs = range(5, 100, 5)
    # if activation_levels is None:
    #     activation_levels = range(5, 100, 5)

    all_res = {}
    for calibrate_activation_level in activation_levels:
        calibrate_activation_level = calibrate_activation_level
        print(f'calibrate_activation_level: {calibrate_activation_level}')
        sim.evaluate(patterns, no_random=True,
                        calibrate_activation_level=calibrate_activation_level)
        ref_pattern = patterns[0]
        sim.evaluate([ref_pattern], no_random=True)
        ref_output = sim.get_grc_activities()[0]
        # sim.calibrate_grc_activation_level(calibrate_activation_level)
        redundant_patterns = sim.add_noise_patterns([ref_pattern], prob=noise_prob, n=test_len)
        sim.evaluate(redundant_patterns, no_random=True)
        acts = sim.get_grc_activities()
        voi = analysis.get_average_metric2(acts, ref_output, metric='voi')
        binary_similarity = analysis.get_average_metric2(acts, ref_output, metric='binary_similarity')
        hamming_distance = analysis.get_average_hamming_distance2(acts, ref_output)
        normalized_mse = analysis.get_normalized_mean_squared_distance(hamming_distance/len(acts), f=calibrate_activation_level)
        mf_dim, mf_pop_corr = analysis.get_dim_from_acts(sim.get_mfs_activities(), ret_population_correlation=True)
        grc_dim, grc_pop_corr = analysis.get_dim_from_acts(sim.get_grc_activities(), ret_population_correlation=True)
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
        all_res[calibrate_activation_level] = res
    return all_res

def test_similarity_by_activation_level2(
        input_graph,
        noise_prob=None,
        activation_levels=None,
        seed=0,
        print_output=False,
        test_len=128,
        ):
    input_graph = copy.deepcopy(input_graph)
    removed = input_graph.remove_empty_mfs()
    random.seed(seed)
    sim = Simulation(
        input_graph=input_graph,
        )
    n_pattern = 1024*4  # 309
    patterns = sim.generate_patterns(
        count=n_pattern,
        # type='gaussian',
        )

    # if noise_probs is None:
    #     noise_probs = range(5, 100, 5)
    # if activation_levels is None:
    #     activation_levels = range(5, 100, 5)

    all_res = {}
    for calibrate_activation_level in activation_levels:
        calibrate_activation_level = calibrate_activation_level
        print(f'calibrate_activation_level: {calibrate_activation_level}')
        sim.evaluate(patterns, no_random=True,
                        calibrate_activation_level=calibrate_activation_level)
        ref_pattern = patterns[0]
        sim.evaluate([ref_pattern], no_random=True)
        ref_output = sim.get_grc_activities()
        # sim.calibrate_grc_activation_level(calibrate_activation_level)
        redundant_patterns = sim.add_noise_patterns([ref_pattern], prob=noise_prob, n=test_len)
        sim.evaluate(redundant_patterns, no_random=True)
        acts = sim.get_grc_activities()
        voi = analysis.get_average_metric(acts, ref_output, metric='voi')
        binary_similarity = analysis.get_average_metric(acts, ref_output, metric='binary_similarity')
        hamming_distance = analysis.get_average_hamming_distance(acts, ref_output)
        normalized_mse = analysis.get_normalized_mean_squared_distance(hamming_distance/len(acts), f=calibrate_activation_level)
        mf_dim, mf_pop_corr = analysis.get_dim_from_acts(sim.get_mfs_activities(), ret_population_correlation=True)
        grc_dim, grc_pop_corr = analysis.get_dim_from_acts(sim.get_grc_activities(), ret_population_correlation=True)
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
        all_res[calibrate_activation_level] = res
    return all_res


def test_similarity_by_noise(
        input_graph,
        noise_probs=None,
        activation_level=None,
        seed=0,
        print_output=False,
        test_len=128,
        ):
    input_graph = copy.deepcopy(input_graph)
    removed = input_graph.remove_empty_mfs()
    random.seed(seed)
    sim = Simulation(
        input_graph=input_graph,
        )
    n_pattern = 1024*4  # 309
    patterns = sim.generate_patterns(
        count=n_pattern,
        # type='gaussian',
        )

    if noise_probs is None:
        noise_probs = range(5, 100, 5)

    all_res = {}
    print(f'activation_level: {activation_level}')
    sim.evaluate(patterns, no_random=True,
                    calibrate_activation_level=activation_level)
    ref_pattern = patterns[0]
    sim.evaluate([ref_pattern], no_random=True)
    ref_output = sim.get_grc_activities()
    for noise_prob in noise_probs:
        print(noise_prob)
        redundant_patterns = sim.add_noise_patterns([ref_pattern], prob=noise_prob, n=test_len)
        sim.evaluate(redundant_patterns, no_random=True)
        acts = sim.get_grc_activities()
        voi = analysis.get_average_metric(acts, ref_output, metric='voi')
        binary_similarity = analysis.get_average_metric(acts, ref_output, metric='binary_similarity')
        hamming_distance = analysis.get_average_hamming_distance(acts, ref_output)
        normalized_mse = analysis.get_normalized_mean_squared_distance(hamming_distance/len(acts), f=activation_level)
        mf_dim, mf_pop_corr = analysis.get_dim_from_acts(sim.get_mfs_activities(), ret_population_correlation=True)
        grc_dim, grc_pop_corr = analysis.get_dim_from_acts(sim.get_grc_activities(), ret_population_correlation=True)
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
        all_res[noise_prob] = res
    return all_res



def add_random_failures(acts, failure_rate):
    indices = [k for k in range(len(acts[0]))]
    random.shuffle(indices)
    fail_idx = indices[0:int(failure_rate*len(indices))]
    failed_acts = copy.deepcopy(acts)
    for a in failed_acts:
        for i in fail_idx:
            a[i] = 0
    return failed_acts

def test_stability_of_core_inputs_across_noise(
        input_graph,
        activation_level,
        noise_probs,
        # failure_rates,
        seed=0,
        test_len=512,
        core_noise=True,
        print_output=True,
        scaled_noise=False,
        sim=None,
        ):
    random.seed(seed)
    if sim is None:
        input_graph = copy.deepcopy(input_graph)
        removed = input_graph.remove_empty_mfs()
        sim = Simulation(
            input_graph=input_graph,
            )
    n_pattern = 1024*4  # 309
    # n_pattern = 512  # 309
    patterns = sim.generate_patterns(count=n_pattern)
    sim.set_failure_rate(0, seed=seed)
    # print(patterns); asdf
    sim.evaluate(patterns, no_random=True,
        calibrate_activation_level=activation_level)
    test_patterns = [patterns[0]]
    all_res = {}
    for noise in noise_probs:
        print(f'noise={noise}')
        if core_noise:
            redundant_patterns = add_noise_to_core_patterns(test_patterns, prob=noise, n=test_len, seed=seed)
        else:
            redundant_patterns = sim.add_noise_patterns(test_patterns, prob=noise, n=test_len, seed=seed, scaled_noise=scaled_noise)
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

def test_dim_similar_input_with_failure(
        input_graph,
        activation_level,
        noise_probs,
        failure_rates,
        seed=0,
        ):
    removed = input_graph.remove_empty_mfs()
    print(f'Removed {len(removed)} mfs')
    random.seed(seed)
    sim = Simulation(
        input_graph=input_graph,
        )
    n_pattern = 1024*4  # 309
    patterns = sim.generate_patterns(count=n_pattern)
    # print(patterns); asdf
    sim.evaluate(patterns, no_random=True,
        calibrate_activation_level=activation_level)
    test_patterns = [patterns[0]]
    for noise in noise_probs:
        print(f'noise={noise}')
        redundant_patterns = sim.add_noise_patterns(test_patterns, prob=noise, n=512)
        sim.evaluate(redundant_patterns, no_random=True)
        acts = sim.get_mfs_activities()
        # for act in acts[0:10]:
        #     print(act[0:20])
        mf_dim = analysis.get_dim_from_acts(acts)
        print(f'Dim MFs: {mf_dim}')
        grc_acts = sim.get_grc_activities()
        # for act in grc_acts[0:10]:
        #     print(act[0:20])
        grc_dim = analysis.get_dim_from_acts(grc_acts)
        print(f'Dim GRCs: {grc_dim}')
        pct_grc = int(grc_dim*1000/sim.num_grcs)/10
        pct_mfs = int(grc_dim*1000/sim.num_mfs)/10
        pct_mf_dim = int(grc_dim*1000/mf_dim)/10
        print(f'    = {pct_mfs}% of MFs')
        print(f'    = {pct_mf_dim}% of MF dim')
        print(f'    = {pct_grc}% of GrCs')
        for failure_rate in failure_rates:
            print(f'unreliability={failure_rate}')
            grc_acts1 = add_random_failures(grc_acts, failure_rate)
            grc_dim1 = analysis.get_dim_from_acts(grc_acts1)
            print(f'    = {grc_dim1:.2f}, {grc_dim1/grc_dim*100:.2f}% of original dimensionality')

def measure_hamming_similarity(vec0, vec1):
    s = 0
    for i, j in zip(vec0, vec1):
        if i == j:
            s += 1
    return s / len(vec0)

def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)

import scipy
import scipy.stats

def measure_correlation_by_sharing(
        input_graph, act_patterns, save_fname=None):
    simlite_graph = SimulationLite(input_graph)
    pos = 0
    grcs_claws = []
    mf_to_grcs = defaultdict(set)
    for grc_id, dendrite_count in enumerate(simlite_graph.dendrite_counts):
        claws = []
        for j in range(dendrite_count):
            mf_id = simlite_graph.dendrite_mf_map[pos]
            pos += 1
            claws.append(mf_id)
            mf_to_grcs[mf_id].add(grc_id)
        grcs_claws.append(set(claws))
    nshares = defaultdict(int)
    nshare_pairs = defaultdict(list)
    counted = set()
    for mf_id, grcs in mf_to_grcs.items():
        for pair in itertools.combinations(grcs, 2):
            if pair in counted:
                continue
            nshare = len(grcs_claws[pair[0]] & grcs_claws[pair[1]])
            nshares[nshare] += 1
            if nshare > 0:
                nshare_pairs[nshare].append(pair)
            counted.add(pair)
            counted.add((pair[1], pair[0]))
    for n in sorted(nshares.keys()):
        print(f'{n}: {nshares[n]/len(simlite_graph.dendrite_counts)}')

    nshare_raw_data_similarity = defaultdict(list)
    nshare_raw_data_corr = defaultdict(list)

    nshare = 0
    # for pair in random_combination(range(len(grcs_claws)), 2):
    n_samples = len(grcs_claws)*10
    combinations = itertools.combinations(range(len(grcs_claws)), 2)
    sim_hist = defaultdict(int)
    cor_hist = defaultdict(int)
    for pair in random_combination(combinations, n_samples):
        # print(pair)
        nshare = len(grcs_claws[pair[0]] & grcs_claws[pair[1]])
        if nshare == 0:
            vec0 = act_patterns[:, pair[0]]
            vec1 = act_patterns[:, pair[1]]
            similarity = measure_hamming_similarity(vec0, vec1)
            corr = scipy.stats.spearmanr(vec0, vec1)[0]
            nshare_raw_data_similarity[nshare].append(similarity)
            nshare_raw_data_corr[nshare].append(corr)
            sim_hist[int(similarity*100)] += 1
    for n in sorted(sim_hist.keys()):
        print(f'{n} {sim_hist[n]}')
    print()

    for nshare in [1, 2, 3]:
        print(f'measuring {nshare}-share, {nshares[nshare]} pairs')
        sim_hist = defaultdict(int)
        for pair in nshare_pairs[nshare]:
            vec0 = act_patterns[:, pair[0]]
            vec1 = act_patterns[:, pair[1]]
            similarity = measure_hamming_similarity(vec0, vec1)
            nshare_raw_data_similarity[nshare].append(similarity)
            corr = scipy.stats.spearmanr(vec0, vec1)[0]
            nshare_raw_data_corr[nshare].append(corr)
            sim_hist[int(similarity*100)] += 1
        for n in sorted(sim_hist.keys()):
            print(f'{n} {sim_hist[n]}')
        # for n in sorted(sim_hist.keys()):
        #     print(n, end=',')
        # print()
        # for n in sorted(sim_hist.keys()):
        #     print(sim_hist[n], end=',')
        print()

    if save_fname:
        nshare_raw_data_similarity = dict(nshare_raw_data_similarity)
        print(f'Saving to {save_fname}')
        compress_pickle.dump(
            (nshare_raw_data_similarity, nshare_raw_data_corr)
            , save_fname)


def test_dim_similar_input_and_correlation_by_sharing(
        input_graph,
        noise_probs,
        activation_level=.3,
        seed=0,
        save_fname=None,
        ):
    removed = input_graph.remove_empty_mfs()
    print(f'Removed {len(removed)} mfs')
    random.seed(seed)
    sim = Simulation(
        input_graph=input_graph,
        )
    n_pattern = 1024*4  # 309
    patterns = sim.generate_patterns(count=n_pattern)
    # print(patterns); asdf
    sim.evaluate(patterns, no_random=True,
        calibrate_activation_level=activation_level)
    test_patterns = [patterns[0]]
    for noise in noise_probs:
        print(f'noise={noise}')
        redundant_patterns = sim.add_noise_patterns(test_patterns, prob=noise, n=512)
        sim.evaluate(redundant_patterns, no_random=True)
        acts = sim.get_mfs_activities()
        # for act in acts[0:10]:
        #     print(act[0:20])
        mf_dim = analysis.get_dim_from_acts(acts)
        print(f'Dim MFs: {mf_dim}')
        acts = sim.get_grc_activities()
        # for act in acts[0:10]:
        #     print(act[0:20])
        dim = analysis.get_dim_from_acts(acts)
        print(f'Dim GRCs: {dim}')
        pct_grc = int(dim*1000/sim.num_grcs)/10
        pct_mfs = int(dim*1000/sim.num_mfs)/10
        pct_mf_dim = int(dim*1000/mf_dim)/10
        print(f'    = {pct_mfs}% of MFs')
        print(f'    = {pct_mf_dim}% of MF dim')
        print(f'    = {pct_grc}% of GrCs')
        measure_correlation_by_sharing(sim, acts, save_fname)

def test_dim_similar_input(
        input_graph,
        noise_probs,
        seed=0,
        ):
    removed = input_graph.remove_empty_mfs()
    print(f'Removed {len(removed)} mfs')
    random.seed(seed)
    sim = Simulation(
        input_graph=input_graph,
        )
    n_pattern = 1024*4  # 309
    patterns = sim.generate_patterns(count=n_pattern)
    # print(patterns); asdf
    sim.evaluate(patterns, no_random=True,
        calibrate_activation_level=.3)
    test_patterns = [patterns[0]]
    for noise in noise_probs:
        print(f'noise={noise}')
        redundant_patterns = sim.add_noise_patterns(test_patterns, prob=noise, n=512)
        sim.evaluate(redundant_patterns, no_random=True)
        acts = sim.get_mfs_activities()
        # for act in acts[0:10]:
        #     print(act[0:20])
        mf_dim = analysis.get_dim_from_acts(acts)
        print(f'Dim MFs: {mf_dim}')
        acts = sim.get_grc_activities()
        # for act in acts[0:10]:
        #     print(act[0:20])
        dim = analysis.get_dim_from_acts(acts)
        print(f'Dim GRCs: {dim}')
        pct_grc = int(dim*1000/sim.num_grcs)/10
        pct_mfs = int(dim*1000/sim.num_mfs)/10
        pct_mf_dim = int(dim*1000/mf_dim)/10
        print(f'    = {pct_mfs}% of MFs')
        print(f'    = {pct_mf_dim}% of MF dim')
        print(f'    = {pct_grc}% of GrCs')

def test_replicated_input(
        n_mfs, n_grcs,
        noise_probs,
        seed=0,
        ):
    random.seed(seed)
    n_pattern = 1024*4  # 309
    for noise in noise_probs:
        print(f'noise={noise}')
        test_patterns = generate_binary_patterns(n_mfs, count=1, f=.3)
        # print(test_patterns[0][0])
        redundant_patterns = add_noise_binary_patterns(test_patterns[0][0], prob=noise, n=512, f=.3)
        # print(redundant_patterns)
        # asdf
        dim = analysis.get_dim_from_acts(redundant_patterns)
        print(f'Dim GRCs: {dim}')
