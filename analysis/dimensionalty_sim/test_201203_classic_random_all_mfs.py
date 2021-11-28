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

num_claws = 4
if '--num_claws' in sys.argv:
    num_claws = int(sys.argv[2])

input_graph.randomize_graph(
    random_model="all_mfs",
    constant_grc_degree=num_claws,
    )

run_tests_201204(input_graph)

