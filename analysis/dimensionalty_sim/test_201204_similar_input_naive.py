import random
import copy
import logging
import sys

from run_tests_201204 import test_dim_similar_input, run_tests_201204

'''Load data'''
import compress_pickle
# fname = '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114.gz'
# input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114_restricted_z.gz')
input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114_xlim_320_560_zlim_10_40.gz')
# input_graph = compress_pickle.load(fname)

input_graph.randomize_graph_by_grc(
    # mf_dist_margin=mf_dist_margin,
    single_connection_per_pair=True,
    constant_grc_degree=4,
    # constant_dendrite_length=20000,
    # constant_dendrite_length=,
    always_pick_closest_rosette=True,
    # preserve_mf_degree=True,
    # approximate_in_degree=True,
    # local_lengths=True,
    )

# run_tests_201204(input_graph)
test_dim_similar_input(input_graph)

