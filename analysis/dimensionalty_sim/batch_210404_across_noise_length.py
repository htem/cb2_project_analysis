import argparse
import random
import copy
import logging
import sys
import os

script_n = os.path.basename(__file__).split('.')[0]

from run_tests_210327 import *

from shuffle_210404 import shuffle
from global_random_model2 import GlobalRandomModel

'''
python batch_210404_across_noise_length.py --model expanded_random --dendrite_len 16
'''

os.makedirs(script_n, exist_ok=True)


ap = argparse.ArgumentParser()
ap.add_argument("--n_random", type=int, help='', default=10)
ap.add_argument("--pattern_len", type=int, help='', default=512)
# ap.add_argument("--activation_levels", type=float, help='', nargs='+', default=None)
ap.add_argument("--activation_level", type=float, help='', default=0.3)
ap.add_argument("--noise_probs", type=float, help='', nargs='+', default=None)
# ap.add_argument("--noise_prob", type=float, help='', default=1)
ap.add_argument("--model", type=str, help='', default='naive_random')
ap.add_argument("--scaled_noise", type=int, help='', default=0)
ap.add_argument("--core_noise", type=int, help='', default=0)
ap.add_argument("--n_grcs", type=int, help='', default=None)
ap.add_argument("--n_mfs", type=int, help='', default=None)
ap.add_argument("--dendrite_len", type=int, help='', default=15)
config = ap.parse_args()

# acts = config.activation_levels
# if acts is None:
#     acts = [k/100 for k in range(5, 100, 5)]
#     acts.insert(0, 0.01)
#     acts.append(0.99)
#     print(f'acts: {acts}')

noise_probs = config.noise_probs
if noise_probs is None:
    # noise_probs = [k/1000 for k in range(25, 1025, 25)]
    noise_probs = [k/1000 for k in range(50, 1050, 50)]
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
        scaled_noise=config.scaled_noise,
        core_noise=config.core_noise,
        # noise_prob=config.noise_prob,
        seed=seed,
        )

model = config.model
model_save_name = model + f'_{config.dendrite_len}'
assert model is not None
print(f'Running {model}')
ress = []
for n in range(config.n_random):
    print(f'Pass {n}')
    input_graph0 = copy.deepcopy(input_graph)
    g = shuffle(input_graph0, model=model,
        seed=n,
        n_grcs=config.n_grcs,
        n_mfs=config.n_mfs,
        constant_dendrite_length=config.dendrite_len*1000,
        )
    res = test(g, seed=n)
    ress.append(res)
    compress_pickle.dump((
        ress,
        ), f"{script_n}/{script_n}_{model_save_name}_{config.activation_level}_{config.pattern_len}_{config.n_random}.gz")
