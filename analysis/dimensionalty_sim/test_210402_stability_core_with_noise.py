import random
import copy
import logging
import sys
import compress_pickle
import argparse

# from run_tests_201204 import *
from run_tests_210327 import *
from scaleup_model import ScaleUpModel
from global_random_model import GlobalRandomModel

from shuffle import shuffle

ap = argparse.ArgumentParser()
# ap.add_argument("--n_random", type=int, help='', default=5)
ap.add_argument("--pattern_len", type=int, help='', default=4096)
ap.add_argument("--activation_level", type=float, help='', default=.3)
ap.add_argument("--noise_probs", type=float, help='',
                        nargs='+',
                        # default=[.05, .5, .9])
                        # default=[k/10 for k in range(0, 11)])
                        default=[.8, .85, .9, .95])
ap.add_argument("--model", type=str, help='', default='data')
ap.add_argument("--n_mfs", type=int, help='', default=1000)
ap.add_argument("--n_grcs", type=int, help='', default=5000)
ap.add_argument("--seed", type=int, help='', default=0)
ap.add_argument("--test_replicated_input", type=int, help='', default=0)
config = ap.parse_args()

# mf_size_dist = compress_pickle.load(
#     '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/'
#     'rosette_size_db_210116.gz'
#     )
# grc_dendrite_count_dist = compress_pickle.load(
#     '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/'
#     'grc_dendrite_count_dist_db_210225.gz'
#     )
# grc_dendrite_len_dist = compress_pickle.load(
#     '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/'
#     'grc_dendrite_len_dist_db_201109.gz'
#     )
input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114_xlim_320_560_zlim_10_40.gz')

random.seed(config.seed)

if config.model == 'scaleup':
    g = ScaleUpModel(
        n_grcs=config.n_grcs,
        n_mfs=config.n_mfs,
        input_graph=input_graph,
        seed=config.seed,
        )
elif config.model == 'global_random':
    g = GlobalRandomModel(
        n_grcs=config.n_grcs,
        n_mfs=config.n_mfs,
        # n_mfs=861,
        # n_mfs_actual=861,
        seed=config.seed,
        )
elif config.model == 'naive_random_17':
    input_graph.randomize_graph_by_grc(
        single_connection_per_pair=True,
        constant_grc_degree=4,
        constant_dendrite_length=17000,
        mf_dist_margin=4000,
        )
    g = input_graph
elif config.model == 'naive_random_21':
    input_graph.randomize_graph_by_grc(
        single_connection_per_pair=True,
        constant_grc_degree=4,
        constant_dendrite_length=21700,
        mf_dist_margin=4000,
        )
    g = input_graph
# else:
#     g = shuffle(input_graph, model=config.model,
#         constant_dendrite_length=17*1000,
#         seed=0)

# if '--naive_random' in sys.argv:
#     input_graph.randomize_graph_by_grc(
#         single_connection_per_pair=True,
#         constant_grc_degree=4,
#         # constant_dendrite_length=20000,
#         # constant_dendrite_length=,
#         always_pick_closest_rosette=True,
#         )


# run_tests_201204(input_graph)
# test_dim_similar_input(input_graph)
# test_hamming_distance_similar_input_201204(input_graph)
# test_dim_201205(input_graph, print_output=True)

# if config.test_replicated_input:
#     test_replicated_input(config.n_mfs, config.n_grcs, config.noise_probs, config.seed)
# else:
test_stability_of_core_inputs_across_noise(
    g,
    # print_output=True,
    # test_len=4096,
    # activation_levels=config.activation_levels,
    activation_level=config.activation_level,
    noise_probs=config.noise_probs,
    # failure_rates=[.1, .2, .3, .4, .5],
    seed=config.seed,
    )
