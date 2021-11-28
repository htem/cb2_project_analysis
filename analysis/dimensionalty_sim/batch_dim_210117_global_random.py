
import argparse
import random
import copy
import logging
import sys
import os
import compress_pickle

script_n = os.path.basename(__file__).split('.')[0]

# from run_tests_201210 import *
from run_tests_201204 import *


from shuffle import shuffle
from global_random_model2 import GlobalRandomModel

ap = argparse.ArgumentParser()
ap.add_argument("--pattern_len", type=int, help='', default=4096)
config = ap.parse_args()

global_random = GlobalRandomModel(
    n_grcs=1211,
    n_mfs=7000,
    seed=0,
    )

# '''Load data'''
# import compress_pickle
# # fname = '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114.gz'
# # input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114_restricted_z.gz')
# input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114_xlim_320_560_zlim_10_40.gz')
# # input_graph = compress_pickle.load(fname)

res = test_dim_201205(
        global_random, print_output=True,
        # test_len=config.pattern_len,
        # activation_levels=range(1, 100),
        # noise_probs=range(1, 100),
        )

compress_pickle.dump((
    res,
    ), f"{script_n}_global_random_{config.pattern_len}.gz")
