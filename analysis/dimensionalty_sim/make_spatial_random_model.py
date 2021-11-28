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
ap.add_argument("--n_mfs", type=int, help='', default=15000)
ap.add_argument("--n_grcs", type=int, help='', default=120000)
ap.add_argument("--seed", type=int, help='', default=0)
# ap.add_argument("--actual_n_grcs", type=int, help='', default=1211)
config = ap.parse_args()

fname = f'models/spatial_random_{config.n_grcs}_{config.n_mfs}_f_{int(config.activation_level*1000)}_seed_{config.seed}'

mf_size_dist = compress_pickle.load(
    '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/'
    'rosette_size_db_210116.gz'
    )
grc_dendrite_count_dist = compress_pickle.load(
    '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/'
    'grc_dendrite_count_dist_db_210225.gz'
    )
grc_dendrite_len_dist = compress_pickle.load(
    '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/'
    'grc_dendrite_len_dist_db_201109.gz'
    )
input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114_xlim_320_560_zlim_10_40.gz')

random.seed(config.seed)

g = SpatialModel(
    n_grcs=120000,
    actual_n_grcs=config.n_grcs,
    n_mfs=config.n_mfs,
    n_boutons=34440,
    size_xyz=(160, 80, 4416),
    dendrite_count_dist=[4, 4, 4, 4, 4, 4, 5, 5, 5, 5],
    dendrite_len_dist=[17, 17, 17, 17],
    mf_size_dist=[15, 15, 15, 15],
    x_expansion=80,
    box_size=80,
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
