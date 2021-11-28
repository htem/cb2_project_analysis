import random
import copy
import logging
import sys

sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')

from neurons import GranuleCell, MossyFiber, Simulation
import analysis
# logging.basicConfig(level=logging.DEBUG)

def test_similarity_by_activation_level(
        input_graph,
        noise_probs=None,
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

    if noise_probs is None:
        noise_probs = range(5, 100, 5)
    if activation_levels is None:
        activation_levels = range(5, 100, 5)

    all_res = {}
    for calibrate_activation_level in activation_levels:
        ress = {}
        calibrate_activation_level = calibrate_activation_level/100
        print(f'calibrate_activation_level: {calibrate_activation_level}')
        sim.evaluate(patterns, no_random=True,
                        calibrate_activation_level=calibrate_activation_level)
        ref_pattern = patterns[0]
        sim.evaluate([ref_pattern], no_random=True)
        ref_output = sim.get_grc_activities()
        # sim.calibrate_grc_activation_level(calibrate_activation_level)
        for noise_prob in noise_probs:
            noise_prob = noise_prob / 100
            print(noise_prob)
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
            ress[noise_prob] = res
        all_res[calibrate_activation_level] = ress
    return all_res

