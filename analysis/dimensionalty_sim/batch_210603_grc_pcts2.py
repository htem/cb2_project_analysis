import argparse
import random
import copy
import logging
import sys
import os

from random_patterns import generate_patterns, add_noise_to_patterns

script_n = os.path.basename(__file__).split('.')[0]

from run_tests_210603 import *

# from shuffle_210404 import shuffle
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
# ap.add_argument("--n_random", type=int, help='', default=1)
ap.add_argument("--n_random", type=int, help='', default=20)
ap.add_argument("--pattern_len", type=int, help='', default=512)
# ap.add_argument("--activation_levels", type=float, help='', nargs='+', default=None)
ap.add_argument("--activation_level", type=float, help='', default=0.3)
ap.add_argument("--grc_pcts", type=float, help='', nargs='+', default=None)
# ap.add_argument("--noise_prob", type=float, help='', default=1)
ap.add_argument("--model", type=str, help='', default='global_random')
ap.add_argument("--scaled_noise", type=int, help='', default=0)
ap.add_argument("--core_noise", type=int, help='', default=0)
ap.add_argument("--n_grcs", type=int, help='', default=2800)
ap.add_argument("--n_mfs", type=int, help='', default=400)
ap.add_argument("--pattern_type", type=str, help='', default='binary')
config = ap.parse_args()

# acts = config.activation_levels
# if acts is None:
#     acts = [k/100 for k in range(5, 100, 5)]
#     acts.insert(0, 0.01)
#     acts.append(0.99)
#     print(f'acts: {acts}')

grc_pcts = config.grc_pcts
if grc_pcts is None:
    grc_pcts = [k/1000 for k in range(25, 1025, 25)]
    # grc_pcts = [k/1000 for k in range(50, 1050, 50)]
    # grc_pcts = [k/1000 for k in range(1000, 1050, 50)]
print(grc_pcts)

n_grcs = config.n_grcs
n_mfs = config.n_mfs
model = config.model
assert model is not None

input_graph = GlobalRandomModel(
    n_grcs=n_grcs,
    n_mfs=n_mfs,
    # n_mfs=861,
    # n_mfs_actual=861,
    seed=0,
    )

# model = Simulation(model)


# '''Load data'''
# import compress_pickle
# if config.model == 'scaleup4':
#     input_graph = compress_pickle.load(f'models/scaleup4_{n_grcs}_{n_mfs}_f_300_seed_0_calibrated_full.gz')
# elif config.model == 'naive_random4':
#     input_graph = compress_pickle.load(f'models/naive_random4_{n_grcs}_{n_mfs}_f_300_seed_0_calibrated_full.gz')
# elif config.model == 'random':
#     input_graph = compress_pickle.load(f'models/global_random_{n_grcs}_{n_mfs}_f_300_seed_0_calibrated_full.gz')
# else:
#     assert False

# pattern_type = 'uniform'
pattern_type = config.pattern_type

pattern_generator = functools.partial(generate_patterns, type=pattern_type)
noise_generator = functools.partial(add_noise_to_patterns, type=pattern_type)

def test(input_graph, seed):

    sim = make_sim(input_graph)
    if pattern_type == 'binary':
        sim.set_binary_mode()
        pass

    return test_across_grc_pcts(
        sim,
        pattern_generator=pattern_generator,
        noise_generator=noise_generator,
        print_output=True,
        test_len=config.pattern_len,
        activation_level=config.activation_level,
        # noise_probs=noise_probs,
        grc_pcts=grc_pcts,
        seed=seed,
        )

print(f'Running {model}')
ress = []
for n in range(config.n_random):
    print(f'Pass {n}')
    res = test(input_graph, seed=n)
    ress.append(res)
    compress_pickle.dump((
        ress,
        ), f"{script_n}/{script_n}_{model}_{pattern_type}_{config.n_grcs}_{config.n_mfs}_{config.activation_level}_{config.pattern_len}_{config.n_random}.gz")
