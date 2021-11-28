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
from scaleup_model import ScaleUpModel
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
# ap.add_argument("--model", type=str, help='', default='small')
ap.add_argument("--n_mfs", type=int, help='', default=500)
ap.add_argument("--n_grcs", type=int, help='', default=1200)
ap.add_argument("--seed", type=int, help='', default=0)
ap.add_argument("--overwrite", type=int, help='', default=0)
ap.add_argument("--dump_full", type=int, help='', default=0)
# ap.add_argument("--actual_n_grcs", type=int, help='', default=1211)
config = ap.parse_args()

# if config.model == "small":
#     config.n_grcs = 1500
#     config.n_mfs = 500
# elif config.model == "big":
#     config.n_grcs = 2000
#     config.n_mfs = 700


fname_precalibrated = f'models/scaleup4_{config.n_grcs}_{config.n_mfs}_seed_{config.seed}_precalibrated.gz'
fname = f'models/scaleup4_{config.n_grcs}_{config.n_mfs}_f_{int(config.activation_level*1000)}_seed_{config.seed}'

input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_210407_xlim_90_140_zlim_4.0_40.0.gz')

# if config.model == "small":
#     input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_210407_xlim_110.0_130.0_zlim_3.0_40.0_rep_3_wrapz.gz')
#     config.n_grcs = 1500
#     config.n_mfs = 500
# elif config.model == "big":
#     input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_210407_xlim_100.0_130.0_zlim_5.0_38.0_rep_3_wrapz.gz')
#     config.n_grcs = 2000
#     config.n_mfs = 700

random.seed(config.seed)

g = ScaleUpModel(
    n_grcs=config.n_grcs,
    n_mfs=config.n_mfs,
    input_graph=input_graph,
    seed=config.seed,
    sort_mfs_by_z=True,
    )
# compress_pickle.dump(g, fname_precalibrated)

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

if config.dump_full:
    print(fname)
    sim.reset()
    compress_pickle.dump(
        sim, fname+'_calibrated_full.gz'
        )

sim_lite = SimulationLite(sim)

print(fname)
compress_pickle.dump(
    sim_lite, fname+'_calibrated.gz'
    )


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




