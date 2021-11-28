import random
import copy
import logging
import sys
import compress_pickle
import argparse
import time
import numpy as np

# from run_tests_201204 import *
from run_tests_201210 import *
from spatial_model2 import SpatialModel
from global_random_model import GlobalRandomModel
from neurons import GranuleCell, MossyFiber, Simulation
from sim_lite import SimulationLite
# from sim_lite_test import SimulationLite

from shuffle import shuffle

ap = argparse.ArgumentParser()
# ap.add_argument("--n_random", type=int, help='', default=5)
# ap.add_argument("--pattern_len", type=int, help='', default=4096)
ap.add_argument("--activation_level", type=float, help='', default=.3)
ap.add_argument("--noise_probs", type=float, help='',
                        nargs='+', default=[5, 50, 95])
ap.add_argument("--model", type=str, help='', default='data')
ap.add_argument("--n_mfs", type=int, help='', default=5000)
ap.add_argument("--n_grcs", type=int, help='', default=10000)
ap.add_argument("--seed", type=int, help='', default=0)
# ap.add_argument("--actual_n_grcs", type=int, help='', default=1211)
ap.add_argument("--dump_full", type=int, help='', default=0)
config = ap.parse_args()

fname = f'models/global_random_{config.n_grcs}_{config.n_mfs}_f_{int(config.activation_level*1000)}_seed_{config.seed}'

g = GlobalRandomModel(
    n_grcs=config.n_grcs,
    n_mfs=config.n_mfs,
    # n_dendrites=4.5,
    # n_mfs=861,
    # n_mfs_actual=861,
    seed=config.seed,
    )

sim = Simulation(
    input_graph=g,
    )

n_pattern = 1024*4
# n_pattern = 1024
patterns = sim.generate_patterns(
    count=n_pattern,
    # type='gaussian',
    )

sim.evaluate(patterns, no_random=True,
             calibrate_activation_level=config.activation_level)

sim_lite = SimulationLite(sim)

compress_pickle.dump(
    sim_lite, fname+'_calibrated.gz'
    )

if config.dump_full:
    print(fname)
    sim.reset()
    compress_pickle.dump(
        sim, fname+'_calibrated_full.gz'
        )

out_array_ref = np.empty(config.n_grcs, dtype=np.uint8)
out_array = np.empty(config.n_grcs, dtype=np.uint8)
# out_array_ref = [None] * config.n_grcs
# out_array = [None] * config.n_grcs
# print(patterns[0])
t0 = time.time()
ret_ref = sim.encode(patterns[0][0], out_array=out_array_ref)
print(f'encode time: {time.time()-t0}')
# print(ret)

np_pattern = np.array(patterns[0][0], dtype=np.float32)

t0 = time.time()
# ret = sim_lite.encode(patterns[0][0], out_array=out_array)
ret = sim_lite.encode(np_pattern, out_array=out_array, use_cython=True)
print(f'cython encode time: {time.time()-t0}')
t0 = time.time()
# ret = sim_lite.encode(patterns[0][0], out_array=out_array)

error_count = 0
for i in range(len(ret)):
    if ret[i] != ret_ref[i]:
        error_count += 1
print(f'num mismatches: {error_count}')

ret = sim_lite.encode(np_pattern, out_array=out_array, use_cython=True, normalize_f=config.activation_level)
print(f'cython with norm: {time.time()-t0}')
# print(ret)

# test_similarity_by_activation_level(
#     g,
#     print_output=True,
#     test_len=4096,
#     activation_levels=config.activation_levels,
#     noise_probs=config.noise_probs,
#     seed=0,
#     )

from collections import defaultdict
import itertools
def count_redundancy(g):
    pos = 0
    grcs_claws = []
    mf_to_grcs = defaultdict(set)
    for grc_id, dendrite_count in enumerate(g.dendrite_counts):
        claws = []
        for j in range(dendrite_count):
            mf_id = g.dendrite_mf_map[pos]
            pos += 1
            claws.append(mf_id)
            mf_to_grcs[mf_id].add(grc_id)
        grcs_claws.append(set(claws))
    nshares = defaultdict(int)
    for mf_id, grcs in mf_to_grcs.items():
        for pair in itertools.combinations(grcs, 2):
            nshare = len(grcs_claws[pair[0]] & grcs_claws[pair[1]])
            nshares[nshare] += 1
    for n in sorted(nshares.keys()):
        print(f'{n}: {nshares[n]/len(g.dendrite_counts)}')

count_redundancy(sim_lite)




