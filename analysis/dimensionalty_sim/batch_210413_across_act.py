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
'''

os.makedirs(script_n, exist_ok=True)


ap = argparse.ArgumentParser()
ap.add_argument("--n_random", type=int, help='', default=10)
ap.add_argument("--pattern_len", type=int, help='', default=512)
ap.add_argument("--activation_levels", type=float, help='', nargs='+', default=None)
# ap.add_argument("--activation_level", type=float, help='', default=0.3)
# ap.add_argument("--noise_probs", type=float, help='', nargs='+', default=None)
ap.add_argument("--noise_prob", type=float, help='', default=1)
ap.add_argument("--model", type=str, help='', default='data')
ap.add_argument("--scaled_noise", type=int, help='', default=0)
ap.add_argument("--core_noise", type=int, help='', default=0)
ap.add_argument("--n_grcs", type=int, help='', default=2400)
ap.add_argument("--n_mfs", type=int, help='', default=400)
config = ap.parse_args()

acts = config.activation_levels
if acts is None:
    acts = [k/100 for k in range(5, 100, 5)]
    acts.insert(0, 0.01)
    acts.append(0.99)
    print(f'acts: {acts}')

n_grcs = config.n_grcs
n_mfs = config.n_mfs
model = config.model
assert model is not None

# noise_probs = config.noise_probs
# if noise_probs is None:
#     # noise_probs = [k/1000 for k in range(25, 1025, 25)]
#     noise_probs = [k/1000 for k in range(50, 1050, 50)]
# print(noise_probs)

'''Load data'''
import compress_pickle
if config.model == 'scaleup4':
    input_graph = compress_pickle.load(f'models/scaleup4_{n_grcs}_{n_mfs}_f_300_seed_0_calibrated_full.gz')
elif config.model == 'naive_random4':
    input_graph = compress_pickle.load(f'models/naive_random4_{n_grcs}_{n_mfs}_f_300_seed_0_calibrated_full.gz')
elif config.model == 'random':
    input_graph = compress_pickle.load(f'models/global_random_{n_grcs}_{n_mfs}_f_300_seed_0_calibrated_full.gz')
else:
    assert False

def test(g, seed):
    return test_similarity_by_activation_level3(
        g,
        print_output=True,
        test_len=config.pattern_len,
        # activation_level=config.activation_level,
        activation_levels=acts,
        noise_prob=config.noise_prob,
        # noise_probs=noise_probs,
        # scaled_noise=config.scaled_noise,
        # core_noise=config.core_noise,
        seed=seed,
        sim=input_graph,
        )

model = config.model
assert model is not None
print(f'Running {model}')
ress = []
for n in range(config.n_random):
    print(f'Pass {n}')
    res = test(input_graph, seed=n)
    ress.append(res)
    compress_pickle.dump((
        ress,
        ), f"{script_n}/{script_n}_{model}_{config.n_grcs}_{config.n_mfs}_noise_{config.noise_prob}_{config.pattern_len}_{config.n_random}.gz")
