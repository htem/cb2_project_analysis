import argparse
import random
import copy
import logging
import sys
import os

script_n = os.path.basename(__file__).split('.')[0]

from run_tests_210327 import *

from shuffle_210329 import shuffle
from global_random_model2 import GlobalRandomModel

'''
python3 batch_similarity_by_activation_level_210329.py --mode act --model random
python3 batch_similarity_by_activation_level_210329.py --mode act --model global_random
python3 batch_similarity_by_activation_level_210329.py --mode act --model data
python3 batch_similarity_by_activation_level_210329.py --mode act --model naive_random_15
python3 batch_similarity_by_activation_level_210329.py --mode act --model naive_random_17
python3 batch_similarity_by_activation_level_210329.py --mode act --model naive_random_21
'''

os.makedirs(script_n, exist_ok=True)


ap = argparse.ArgumentParser()
ap.add_argument("--n_random", type=int, help='', default=10)
ap.add_argument("--pattern_len", type=int, help='', default=512)
# ap.add_argument("--activation_levels", type=float, help='', nargs='+', default=None)
ap.add_argument("--activation_level", type=float, help='', default=0.3)
ap.add_argument("--noise_probs", type=float, help='', nargs='+', default=None)
# ap.add_argument("--noise_prob", type=float, help='', default=1)
ap.add_argument("--model", type=str, help='', default=None)
config = ap.parse_args()

# acts = config.activation_levels
# if acts is None:
#     acts = [k/100 for k in range(5, 100, 5)]
#     acts.insert(0, 0.01)
#     acts.append(0.99)
#     print(f'acts: {acts}')

noise_probs = config.noise_probs
if noise_probs is None:
    noise_probs = [k/1000 for k in range(25, 1025, 25)]
print(noise_probs)

'''Load data'''
import compress_pickle
input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114_xlim_320_560_zlim_10_40.gz')

def test(g, seed):
    return test_stability_of_core_inputs_across_noise(
        g,
        print_output=True,
        test_len=config.pattern_len,
        # activation_levels=acts,
        activation_level=config.activation_level,
        noise_probs=noise_probs,
        scaled_noise=True,
        core_noise=False,
        # noise_prob=config.noise_prob,
        seed=seed,
        )

model = config.model
assert model is not None
print(f'Running {model}')
ress = []
for n in range(config.n_random):
    print(f'Pass {n}')
    input_graph0 = copy.deepcopy(input_graph)
    g = shuffle(input_graph0, model=model,
        seed=n)
    res = test(g, seed=n)
    ress.append(res)
    compress_pickle.dump((
        ress,
        ), f"{script_n}/{script_n}_{model}_{config.activation_level}_{config.pattern_len}_{config.n_random}.gz")
