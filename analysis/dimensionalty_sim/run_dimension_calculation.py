import random
import copy
import logging
import sys

sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')

from neurons import GranuleCell, MossyFiber, Simulation
import analysis
# logging.basicConfig(level=logging.DEBUG)

def test_dim_similar_input(input_graph):
    removed = input_graph.remove_empty_mfs()
    print(f'Removed {len(removed)} mfs')
    random.seed(0)
    sim = Simulation(
        input_graph=input_graph,
        )
    n_pattern = 1024*4  # 309
    patterns = sim.generate_patterns(count=n_pattern)
    # print(patterns); asdf
    sim.evaluate(patterns, no_random=True,
        calibrate_activation_level=.1)
    test_patterns = [patterns[0]]
    redundant_patterns = sim.add_noise_patterns(test_patterns, prob=.1, n=1024)
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


def run_tests_201204(input_graph):
    removed = input_graph.remove_empty_mfs()
    print(f'Removed {len(removed)} mfs')
    random.seed(0)
    sim = Simulation(
        input_graph=input_graph,
        )
    n_pattern = 1024*4  # 309
    patterns = sim.generate_patterns(
        count=n_pattern,
        # type='gaussian',
        )
    print(f'len(patterns): {len(patterns)}')
    sim.evaluate(patterns, no_random=True, calibrate_activation_level=.1)
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
