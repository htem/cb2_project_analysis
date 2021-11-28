import random
import copy
import logging
import sys

from run_tests_201204 import *

'''Load data'''
import compress_pickle
# fname = '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114.gz'
# input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114_restricted_z.gz')
input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114_xlim_320_560_zlim_10_40.gz')
# input_graph = compress_pickle.load(fname)

# run_tests_201204(input_graph)
test_hamming_distance_similar_input_201204(input_graph)

input_graph.randomize_graph(
    random_model=True,
    constant_grc_degree=num_claws,
    )

# run_tests_201204(input_graph)
# test_hamming_distance_similar_input_201204(input_graph)
test_dim_similar_input(input_graph)


