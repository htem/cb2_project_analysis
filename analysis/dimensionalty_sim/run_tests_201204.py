import random
import copy
import logging
import sys
import time

sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')

from neurons import GranuleCell, MossyFiber, Simulation
import analysis
# logging.basicConfig(level=logging.DEBUG)

def test_voi_similar_input_201204(
        input_graph,
        noise_prob=.1,
        # noise_prob=.5,
        # noise_prob=1,
        # calibrate_activation_level=.5,
        calibrate_activation_level=.1,
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
        calibrate_activation_level=.1)
    ref_pattern = patterns[0]
    sim.evaluate([ref_pattern], no_random=True)
    ref_output = sim.get_grc_activities()
    # print(f'ref_pattern: {ref_pattern}')
    # print(f'ref_output: {ref_output}')
    # redundant_patterns = sim.add_noise_patterns([ref_pattern], prob=.05, n=1024*4)
    # redundant_patterns = sim.add_noise_patterns([ref_pattern], prob=noise_prob, n=1024)
    redundant_patterns = sim.add_noise_patterns([ref_pattern], prob=noise_prob, n=128)
    # redundant_patterns = sim.add_noise_patterns([ref_pattern], prob=noise_prob, n=2)
    # redundant_patterns = sim.add_noise_patterns([ref_pattern], prob=.2, n=1024)
    # redundant_patterns = sim.add_noise_patterns([ref_pattern], prob=1, n=2)
    sim.evaluate(redundant_patterns, no_random=True)
    # mf_dim = analysis.get_dim_from_acts(sim.get_mfs_activities())
    # print(f'Dim MFs: {mf_dim}')
    acts = sim.get_grc_activities()
    voi = analysis.get_average_metric(acts, ref_output, metric='voi')
    print(f'VOI: {voi}')
    ref_similarity = analysis.get_average_metric(acts, ref_output, metric='ref_similarity')
    print(f'ref_similarity: {ref_similarity}')
    dist = analysis.get_average_hamming_distance(acts, ref_output)
    print(f'Hamming distance: {dist}')
    pct_grc = int(dist*1000/sim.num_grcs)/10
    print(f'    = {pct_grc}% of GrCs')
    dist = analysis.get_activation_similarity(acts, ref_output)
    dist = int(dist*1000)/10
    print(f'Activation Similarity: {dist}%')

def test_hamming_distance_similar_input_201204(
        input_graph,
        # noise_prob=.1,
        noise_prob=.5,
        # noise_prob=1,
        # calibrate_activation_level=.5,
        calibrate_activation_level=.1,
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
        calibrate_activation_level=.1)
    ref_pattern = patterns[0]
    sim.evaluate([ref_pattern], no_random=True)
    ref_output = sim.get_grc_activities()
    # print(f'ref_pattern: {ref_pattern}')
    # print(f'ref_output: {ref_output}')
    # redundant_patterns = sim.add_noise_patterns([ref_pattern], prob=.05, n=1024*4)
    # redundant_patterns = sim.add_noise_patterns([ref_pattern], prob=noise_prob, n=1024)
    redundant_patterns = sim.add_noise_patterns([ref_pattern], prob=noise_prob, n=128)
    # redundant_patterns = sim.add_noise_patterns([ref_pattern], prob=noise_prob, n=2)
    # redundant_patterns = sim.add_noise_patterns([ref_pattern], prob=.2, n=1024)
    # redundant_patterns = sim.add_noise_patterns([ref_pattern], prob=1, n=2)
    sim.evaluate(redundant_patterns, no_random=True)
    # mf_dim = analysis.get_dim_from_acts(sim.get_mfs_activities())
    # print(f'Dim MFs: {mf_dim}')
    acts = sim.get_grc_activities()
    dist = analysis.get_average_hamming_distance(acts, ref_output)
    print(f'Hamming distance: {dist}')
    pct_grc = int(dist*1000/sim.num_grcs)/10
    print(f'    = {pct_grc}% of GrCs')
    dist = analysis.get_activation_similarity(acts, ref_output)
    dist = int(dist*1000)/10
    print(f'Activation Similarity: {dist}%')

    # dim = analysis.get_dim_from_acts(sim.get_grc_activities())
    # # print(f'Dim GRCs: {dim}')
    # pct_mfs = int(dim*1000/sim.num_mfs)/10
    # pct_mf_dim = int(dim*1000/mf_dim)/10
    # print(f'    = {pct_mfs}% of MFs')
    # print(f'    = {pct_mf_dim}% of MF dim')

def test_dim_similar_input(input_graph):
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
        calibrate_activation_level=.1)
    test_patterns = [patterns[0]]
    # redundant_patterns = sim.add_noise_patterns(test_patterns, prob=.05, n=1024*4)
    redundant_patterns = sim.add_noise_patterns(test_patterns, prob=.1, n=1024*4)
    # redundant_patterns = sim.add_noise_patterns(test_patterns, prob=.2, n=1024)
    # redundant_patterns = sim.add_noise_patterns(test_patterns, prob=1, n=2)
    sim.evaluate(redundant_patterns, no_random=True)
    mf_dim = analysis.get_dim_from_acts(sim.get_mfs_activities())
    print(f'Dim MFs: {mf_dim}')
    dim = analysis.get_dim_from_acts(sim.get_grc_activities())
    print(f'Dim GRCs: {dim}')
    pct_grc = int(dim*1000/sim.num_grcs)/10
    pct_mfs = int(dim*1000/sim.num_mfs)/10
    pct_mf_dim = int(dim*1000/mf_dim)/10
    print(f'    = {pct_mfs}% of MFs')
    print(f'    = {pct_mf_dim}% of MF dim')
    print(f'    = {pct_grc}% of GrCs')


def run_tests_201204(
        input_graph,
        calibrate_activation_level=.1,
        ):
    removed = input_graph.remove_empty_mfs()
    print(f'Removed {len(removed)} mfs')
    random.seed(seed)
    sim = Simulation(
        input_graph=input_graph,
        )
    n_pattern = 1024*4  # 309
    patterns = sim.generate_patterns(
        count=n_pattern,
        # type='gaussian',
        )
    print(f'len(patterns): {len(patterns)}')
    sim.evaluate(patterns, no_random=True, calibrate_activation_level=calibrate_activation_level)
    sim.evaluate(patterns, no_random=True)
    mf_dim = analysis.get_dim_from_acts(sim.get_mfs_activities())
    print(f'Dim MFs: {mf_dim}')
    dim = analysis.get_dim_from_acts(sim.get_grc_activities())
    print(f'Dim GRCs: {dim}')
    pct_grc = int(dim*1000/sim.num_grcs)/10
    pct_mfs = int(dim*1000/sim.num_mfs)/10
    pct_mf_dim = int(dim*1000/mf_dim)/10
    print(f'    = {pct_mfs}% of MFs')
    print(f'    = {pct_mf_dim}% of MF dim')
    print(f'    = {pct_grc}% of GrCs')


def test_dim_201205(
        input_graph,
        calibrate_activation_level=.1,
        print_output=False,
        seed=0,
        n_pattern=4096,
        ):
    input_graph = copy.deepcopy(input_graph)
    removed = input_graph.remove_empty_mfs()
    random.seed(seed)
    sim = Simulation(
        input_graph=input_graph,
        )
    t0 = time.time()
    patterns = sim.generate_patterns(
        count=n_pattern,
        # type='gaussian',
        )
    print(f'{time.time()-t0}s to generate patterns')
    t0 = time.time()
    sim.evaluate(patterns, no_random=True, calibrate_activation_level=calibrate_activation_level)
    print(f'{time.time()-t0}s to calibrate')
    t0 = time.time()
    sim.evaluate(patterns, no_random=True)
    print(f'{time.time()-t0}s to evaluate')
    t0 = time.time()
    res = {}
    mf_dim = analysis.get_dim_from_acts(sim.get_mfs_activities())
    grc_dim = analysis.get_dim_from_acts(sim.get_grc_activities())
    pct_grc = int(grc_dim*1000/sim.num_grcs)/10
    pct_mfs = int(grc_dim*1000/sim.num_mfs)/10
    pct_mf_dim = int(grc_dim*1000/mf_dim)/10
    print(f'{time.time()-t0}s to calculate dims')
    if print_output:
        # print(f'Removed {len(removed)} mfs')
        # print(f'len(patterns): {len(patterns)}')
        print(f'Dim MFs: {mf_dim}')
        print(f'Dim GRCs: {grc_dim}')
        print(f'    = {pct_mfs}% of MFs')
        print(f'    = {pct_mf_dim}% of MF dim')
        print(f'    = {pct_grc}% of GrCs')
    res['mf_dim'] = mf_dim
    res['grc_dim'] = grc_dim
    res['pct_grc'] = pct_grc
    res['pct_mfs'] = pct_mfs
    res['pct_mf_dim'] = pct_mf_dim
    res['num_grcs'] = sim.num_grcs
    res['num_mfs'] = sim.num_mfs
    return res


def test_dim_by_activation_level_201205(
        input_graph,
        activation_levels=None,
        print_output=True,
        seed=0,
        # seed=1,
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
    sim.evaluate(patterns, no_random=True)
    if activation_levels is None:
        activation_levels = range(5, 100, 5)
    ress = {}
    mf_dim = None
    for calibrate_activation_level in activation_levels:
        calibrate_activation_level = calibrate_activation_level/100
        sim.calibrate_grc_activation_level(calibrate_activation_level)
        print(calibrate_activation_level)
        sim.evaluate(patterns, no_random=True)
        res = {}
        if mf_dim is None:
            mf_dim = analysis.get_dim_from_acts(sim.get_mfs_activities())
        grc_dim = analysis.get_dim_from_acts(sim.get_grc_activities())
        pct_grc = int(grc_dim*1000/sim.num_grcs)/10
        pct_mfs = int(grc_dim*1000/sim.num_mfs)/10
        pct_mf_dim = int(grc_dim*1000/mf_dim)/10
        if print_output:
            # print(f'Removed {len(removed)} mfs')
            # print(f'len(patterns): {len(patterns)}')
            print(f'Dim MFs: {mf_dim}')
            print(f'Dim GRCs: {grc_dim}')
            print(f'    = {pct_mfs}% of MFs')
            print(f'    = {pct_mf_dim}% of MF dim')
            print(f'    = {pct_grc}% of GrCs')
        res['mf_dim'] = mf_dim
        res['grc_dim'] = grc_dim
        res['pct_grc'] = pct_grc
        res['pct_mfs'] = pct_mfs
        res['pct_mf_dim'] = pct_mf_dim
        res['num_grcs'] = sim.num_grcs
        res['num_mfs'] = sim.num_mfs
        ress[calibrate_activation_level] = res
    return ress


def test_dim_degrade_201205(
        input_graph,
        calibrate_activation_level=.1,
        degrades=None,
        seed=0,
        print_output=True,
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
    sim.evaluate(patterns, no_random=True, calibrate_activation_level=calibrate_activation_level)
    sim.evaluate(patterns, no_random=True)
    mf_dim = analysis.get_dim_from_acts(sim.get_mfs_activities())
    grc_acts_0 = sim.get_grc_activities()
    grc_dim_max = analysis.get_dim_from_acts(grc_acts_0)

    if degrades is None:
        degrades = range(1, 100)
        degrades = range(5, 100, 5)
    ress = {}
    for grc_degrade in degrades:
        grc_degrade = grc_degrade/100
        res = {}

        grc_acts = copy.deepcopy(grc_acts_0)
        random.seed(seed)
        for i, individual_grc_act in enumerate(grc_acts):
            if random.random() < grc_degrade:
                grc_acts[i] = [0 for k in individual_grc_act]

        grc_dim = analysis.get_dim_from_acts(grc_acts)
        pct_grc = int(grc_dim*1000/sim.num_grcs)/10
        pct_mfs = int(grc_dim*1000/sim.num_mfs)/10
        pct_mf_dim = int(grc_dim*1000/mf_dim)/10
        if print_output:
            # print(f'Removed {len(removed)} mfs')
            # print(f'len(patterns): {len(patterns)}')
            print(f'Dim MFs: {mf_dim}')
            print(f'Dim GRCs: {grc_dim}')
            print(f'    = {pct_mfs}% of MFs')
            print(f'    = {pct_mf_dim}% of MF dim')
            print(f'    = {pct_grc}% of GrCs')
        res['mf_dim'] = mf_dim
        res['grc_dim'] = grc_dim
        res['pct_grc'] = pct_grc
        res['pct_mfs'] = pct_mfs
        res['pct_mf_dim'] = pct_mf_dim
        res['num_grcs'] = sim.num_grcs
        res['num_mfs'] = sim.num_mfs
        res['grc_dim_max'] = grc_dim_max
        ress[grc_degrade] = res
    return ress


def test_similarity_by_activation_level_201205(
        input_graph,
        noise_probs=None,
        activation_levels=None,
        seed=0,
        print_output=True,
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
            redundant_patterns = sim.add_noise_patterns([ref_pattern], prob=noise_prob, n=128)
            sim.evaluate(redundant_patterns, no_random=True)
            acts = sim.get_grc_activities()
            voi = analysis.get_average_metric(acts, ref_output, metric='voi')
            binary_similarity = analysis.get_average_metric(acts, ref_output, metric='binary_similarity')
            hamming_distance = analysis.get_average_hamming_distance(acts, ref_output)
            normalized_mse = analysis.get_normalized_mean_squared_distance(hamming_distance/len(acts), f=calibrate_activation_level)
            # mf_dim = analysis.get_dim_from_acts(sim.get_mfs_activities())
            grc_dim, grc_pop_corr = analysis.get_dim_from_acts(sim.get_grc_activities(), ret_population_correlation=True)
            pct_grc = int(grc_dim*1000/sim.num_grcs)/10
            pct_mfs = int(grc_dim*1000/sim.num_mfs)/10
            # pct_mf_dim = int(grc_dim*1000/mf_dim)/10
            if print_output:
                print(f'voi: {voi}')
                print(f'binary_similarity: {binary_similarity}')
                print(f'hamming_distance: {hamming_distance}')
                print(f'normalized_mse: {normalized_mse}')
                print(f'grc_pop_corr: {grc_pop_corr}')
                print(f'Dim GRCs: {grc_dim}')
                print(f'    = {pct_mfs}% of MFs')
                # print(f'    = {pct_mf_dim}% of MF dim')
                print(f'    = {pct_grc}% of GrCs')
            res = {}
            res['voi'] = voi
            res['binary_similarity'] = binary_similarity
            res['hamming_distance'] = hamming_distance
            res['normalized_mse'] = normalized_mse
            res['grc_pop_corr'] = grc_pop_corr
            # res['mf_dim'] = mf_dim
            res['grc_dim'] = grc_dim
            res['pct_grc'] = pct_grc
            res['pct_mfs'] = pct_mfs
            # res['pct_mf_dim'] = pct_mf_dim
            res['num_grcs'] = sim.num_grcs
            res['num_mfs'] = sim.num_mfs
            ress[noise_prob] = res
        all_res[calibrate_activation_level] = ress
    return all_res


def test_similarity_201205(
        input_graph,
        noise_probs=None,
        calibrate_activation_level=.1,
        seed=0,
        print_output=True,
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
    sim.evaluate(patterns, no_random=True,
                    calibrate_activation_level=calibrate_activation_level)
    ref_pattern = patterns[0]
    sim.evaluate([ref_pattern], no_random=True)
    ref_output = sim.get_grc_activities()

    all_res = {}
    if noise_probs is None:
        noise_probs = range(1, 100)
        # noise_probs = range(5, 100, 5)
    for noise_prob in noise_probs:
        noise_prob = noise_prob / 100
        print(noise_prob)
        redundant_patterns = sim.add_noise_patterns([ref_pattern], prob=noise_prob, n=128)
        sim.evaluate(redundant_patterns, no_random=True)
        acts = sim.get_grc_activities()
        voi = analysis.get_average_metric(acts, ref_output, metric='voi')
        binary_similarity = analysis.get_average_metric(acts, ref_output, metric='binary_similarity')
        hamming_distance = analysis.get_average_hamming_distance(acts, ref_output)
        normalized_mse = analysis.get_normalized_mean_squared_distance(hamming_distance)
        # mf_dim = analysis.get_dim_from_acts(sim.get_mfs_activities())
        grc_dim = analysis.get_dim_from_acts(sim.get_grc_activities())
        pct_grc = int(grc_dim*1000/sim.num_grcs)/10
        pct_mfs = int(grc_dim*1000/sim.num_mfs)/10
        # pct_mf_dim = int(grc_dim*1000/mf_dim)/10
        if print_output:
            print(f'voi: {voi}')
            print(f'binary_similarity: {binary_similarity}')
            print(f'hamming_distance: {hamming_distance}')
            print(f'Dim GRCs: {grc_dim}')
            print(f'    = {pct_mfs}% of MFs')
            # print(f'    = {pct_mf_dim}% of MF dim')
            print(f'    = {pct_grc}% of GrCs')
        res = {}
        res['voi'] = voi
        res['binary_similarity'] = binary_similarity
        res['hamming_distance'] = hamming_distance
        res['normalized_mse'] = normalized_mse
        # res['mf_dim'] = mf_dim
        res['grc_dim'] = grc_dim
        res['pct_grc'] = pct_grc
        res['pct_mfs'] = pct_mfs
        # res['pct_mf_dim'] = pct_mf_dim
        res['num_grcs'] = sim.num_grcs
        res['num_mfs'] = sim.num_mfs
        all_res[noise_prob] = res
    return all_res



def test_similarity_degrade_201205(
        input_graph,
        calibrate_activation_level=.1,
        degrades=None,
        noise_prob=1.0,
        seed=0,
        print_output=True,
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
    sim.evaluate(patterns, no_random=True,
                    calibrate_activation_level=calibrate_activation_level)
    ref_pattern = patterns[0]
    sim.evaluate([ref_pattern], no_random=True)
    ref_output_0 = sim.get_grc_activities()
    noise_prob = noise_prob
    redundant_patterns = sim.add_noise_patterns([ref_pattern], prob=noise_prob, n=128)
    sim.evaluate(redundant_patterns, no_random=True)
    grc_acts_0 = sim.get_grc_activities()
    max_voi = analysis.get_average_metric(grc_acts_0, ref_output_0, metric='voi')
    max_binary_similarity = analysis.get_average_metric(grc_acts_0, ref_output_0, metric='binary_similarity')
    max_hamming_distance = analysis.get_average_hamming_distance(grc_acts_0, ref_output_0)
    normalized_mse = analysis.get_normalized_mean_squared_distance(hamming_distance)
    max_grc_dim = analysis.get_dim_from_acts(grc_acts_0)

    all_res = {}
    if degrades is None:
        # degrades = range(1, 100)
        degrades = range(5, 100, 5)
    for grc_degrade in degrades:
        grc_degrade = grc_degrade / 100
        print(grc_degrade)

        grc_acts = copy.deepcopy(grc_acts_0)
        ref_output = copy.deepcopy(ref_output_0)
        random.seed(seed)
        for i, individual_grc_act in enumerate(grc_acts):
            if random.random() < grc_degrade:
                grc_acts[i] = [0 for k in individual_grc_act]
                ref_output[i] = [0 for k in ref_output[i]]

        voi = analysis.get_average_metric(grc_acts, ref_output, metric='voi')
        binary_similarity = analysis.get_average_metric(grc_acts, ref_output, metric='binary_similarity')
        hamming_distance = analysis.get_average_hamming_distance(grc_acts, ref_output)
        normalized_mse = analysis.get_normalized_mean_squared_distance(hamming_distance)
        grc_dim = analysis.get_dim_from_acts(grc_acts)
        pct_grc = int(grc_dim*1000/sim.num_grcs)/10
        pct_mfs = int(grc_dim*1000/sim.num_mfs)/10
        # pct_mf_dim = int(grc_dim*1000/mf_dim)/10
        if print_output:
            print(f'voi: {voi}')
            print(f'binary_similarity: {binary_similarity}')
            print(f'hamming_distance: {hamming_distance}')
            print(f'Dim GRCs: {grc_dim}')
            print(f'    = {pct_mfs}% of MFs')
            # print(f'    = {pct_mf_dim}% of MF dim')
            print(f'    = {pct_grc}% of GrCs')
        res = {}
        res['voi'] = voi
        res['binary_similarity'] = binary_similarity
        res['hamming_distance'] = hamming_distance
        res['normalized_mse'] = normalized_mse
        # res['mf_dim'] = mf_dim
        res['grc_dim'] = grc_dim
        res['pct_grc'] = pct_grc
        res['pct_mfs'] = pct_mfs
        res['max_voi'] = max_voi
        res['max_binary_similarity'] = max_binary_similarity
        res['max_hamming_distance'] = max_hamming_distance
        res['max_grc_dim'] = max_grc_dim
        # res['pct_mf_dim'] = pct_mf_dim
        res['num_grcs'] = sim.num_grcs
        res['num_mfs'] = sim.num_mfs
        all_res[grc_degrade] = res
    return all_res
