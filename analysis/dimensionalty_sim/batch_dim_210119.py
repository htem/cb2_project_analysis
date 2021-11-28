import argparse
import random
import copy
import logging
import sys
import os

script_n = os.path.basename(__file__).split('.')[0]

from run_tests_201204 import *
# from run_tests_201210 import *

from shuffle import shuffle
from global_random_model2 import GlobalRandomModel



ap = argparse.ArgumentParser()
ap.add_argument("--n_random", type=int, help='', default=5)
ap.add_argument("--n_pattern", type=int, help='', default=4096)
ap.add_argument("--act_lv", type=int, help='', default=10)
# ap.add_argument("--pattern_len", type=int, help='', default=4096)
# ap.add_argument("--activation_levels", type=float, help='', nargs='+', default=[20])
# ap.add_argument("--noise_probs", type=float, help='', nargs='+', default=None)
ap.add_argument("--models", type=str, help='',
        nargs='+', default=[
        # 'data',
        # 'expanded_random_30',
        # 'global_random',
        # 'naive_random_15_2',
        'naive_random_17_2',
        ])
config = ap.parse_args()

# noise_probs = config.noise_probs
# if noise_probs is None:
#     noise_probs = [k/10 for k in range(25, 1025, 25)]
# print(noise_probs)

'''Load data'''
import compress_pickle
input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114_xlim_320_560_zlim_10_40.gz')

def test(g, seed):
    return test_dim_201205(
        g,
        print_output=True,
        n_pattern=config.n_pattern,
        # activation_levels=config.activation_levels,
        # noise_probs=noise_probs,
        calibrate_activation_level=config.act_lv/100,
        seed=seed,
        )

for model in config.models:
    print(f'Running {model}')
    ress = []
    for n in range(config.n_random):
        print(f'Pass {n}')
        if model == 'naive_random':
            g = shuffle(input_graph, model=model,
                constant_dendrite_length=17*1000,
                seed=n,
                )
        elif model == 'global_random':
            g = GlobalRandomModel(
                n_grcs=1211,
                n_mfs=7000,
                seed=n,
                )
        else:
            g = shuffle(input_graph, model=model,
                seed=n)
        res = test(g, seed=n)
        ress.append(res)
    compress_pickle.dump((
        ress,
        ), f"data/{script_n}_{model}_len_{config.n_pattern}_{config.act_lv}_{config.n_random}.gz")
