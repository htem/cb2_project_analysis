import random
import copy
import logging
import sys
import compress_pickle
import argparse

# from run_tests_201204 import *
from run_tests_201210 import *
from spatial_model2 import SpatialModel
from global_random_model import GlobalRandomModel

from shuffle import shuffle

ap = argparse.ArgumentParser()
ap.add_argument("--n_random", type=int, help='', default=5)
ap.add_argument("--pattern_len", type=int, help='', default=4096)
ap.add_argument("--activation_levels", type=float, help='', nargs='+', default=[20])
ap.add_argument("--noise_probs", type=float, help='',
                        nargs='+', default=[5, 50, 95])
ap.add_argument("--model", type=str, help='', default='data')
ap.add_argument("--n_mfs", type=int, help='', default=7000)
ap.add_argument("--actual_n_grcs", type=int, help='', default=1211)
config = ap.parse_args()


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

random.seed(0)

if config.model == 'spatial':
    g = SpatialModel(
        n_grcs=100000,
        actual_n_grcs=config.actual_n_grcs,
        n_mfs=10000,
        n_boutons=28700,
        size_xyz=(160, 80, 3680),
        dendrite_count_dist=grc_dendrite_count_dist,
        dendrite_len_dist=grc_dendrite_len_dist,
        mf_size_dist=mf_size_dist,
        x_expansion=80,
        box_size=80,
        )
elif config.model == 'spatial_random':
    g = SpatialModel(
        n_grcs=100000,
        actual_n_grcs=config.actual_n_grcs,
        n_mfs=10000,
        n_boutons=28700,
        size_xyz=(160, 80, 3680),
        dendrite_count_dist=[4, 4, 4, 4, 4, 4, 5, 5, 5, 5],
        dendrite_len_dist=[17, 17, 17, 17],
        mf_size_dist=[15, 15, 15, 15],
        x_expansion=80,
        # x_expansion=0,
        box_size=80,
        )
elif config.model == 'global_random':
    g = GlobalRandomModel(
        n_grcs=1211,
        n_mfs=config.n_mfs,
        # n_mfs=861,
        # n_mfs_actual=861,
        seed=0,
        )
else:
    g = shuffle(input_graph, model=config.model,
        constant_dendrite_length=17*1000,
        seed=0)

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

test_similarity_by_activation_level(
    g,
    print_output=True,
    test_len=4096,
    activation_levels=config.activation_levels,
    noise_probs=config.noise_probs,
    seed=0,
    )

