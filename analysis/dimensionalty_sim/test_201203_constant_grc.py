import random
import copy
import logging
import sys

from run_tests_201204 import *

'''Load data'''
import compress_pickle
# fname = '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114.gz'
input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114_xlim_320_560_zlim_10_40.gz')
# input_graph = compress_pickle.load(fname)

num_claws = 4
if '--num_claws' in sys.argv:
    num_claws = int(sys.argv[2])

input_graph.randomize_graph_by_grc(
        mf_dist_margin=4000,
        single_connection_per_pair=True,
        constant_grc_degree=num_claws,
        # constant_dendrite_length=15000,
        # always_pick_closest_rosette=True,
        preserve_mf_degree=True,
    # preserve_mf_degree=True,
    # approximate_in_degree=True,
    # local_lengths=True,
    )

# run_tests_201204(input_graph)
test_dim_similar_input(input_graph)

