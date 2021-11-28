import argparse
import random
import copy
import logging
import sys
import os

script_n = os.path.basename(__file__).split('.')[0]

from run_tests_201210 import *

from shuffle import shuffle



ap = argparse.ArgumentParser()
ap.add_argument("--pattern_len", type=int, help='', default=4096)
ap.add_argument("--models", type=str, help='',
        nargs='+', default=[
        'expanded_random_30',
        'expanded_random_50',
        'data',
        # 'classic_random',
        'naive_random',
        ])
config = ap.parse_args()

'''Load data'''
import compress_pickle
input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114_xlim_320_560_zlim_10_40.gz')

def test(g):
    return test_similarity_by_activation_level(
        g,
        # print_output=True,
        test_len=config.pattern_len,
        # activation_levels=range(1, 100),
        # noise_probs=range(1, 100),
        )

for model in config.models:
    print(f'Running {model}')
    if model == 'naive_random':
        g = shuffle(input_graph, model=model,
            mf_dist_margin=4*1000,
            constant_dendrite_length=17*1000,
            )
    else:
        g = shuffle(input_graph, model=model)
    res = test(g)
    compress_pickle.dump((
        res,
        ), f"data/{script_n}_{model}_{config.pattern_len}.gz")
