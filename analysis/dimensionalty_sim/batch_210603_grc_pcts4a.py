import argparse
import random
import copy
import logging
import sys
import os

from random_patterns import generate_patterns, add_noise_to_patterns

script_n = os.path.basename(__file__).split('.')[0]

from run_tests_210603 import *
from make_graph_210611a import make_graph

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
ap.add_argument("--model", type=str, help='', default='local_random')
ap.add_argument("--scaled_noise", type=int, help='', default=0)
ap.add_argument("--core_noise", type=int, help='', default=0)
ap.add_argument("--n_grcs", type=int, help='', default=1789)
ap.add_argument("--n_mfs", type=int, help='', default=444)
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
    # grc_pcts = [k/1000 for k in range(25, 1025, 25)]
    grc_pcts = [k/1000 for k in range(10, 1010, 10)]
    # grc_pcts = [k/1000 for k in range(50, 1050, 50)]
    # grc_pcts = [k/1000 for k in range(1000, 1050, 50)]
print(grc_pcts)

n_grcs = config.n_grcs
n_mfs = config.n_mfs
model = config.model
assert model is not None


# model = Simulation(model)
pattern_type = config.pattern_type

pattern_generator = functools.partial(generate_patterns, type=pattern_type)
noise_generator = functools.partial(add_noise_to_patterns, type=pattern_type)

def test(model, seed):

    g, model_desc = make_graph(model, n_grcs, n_mfs, seed=seed)

    sim = make_sim(g)
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
    res = test(model, seed=n)
    ress.append(res)
    compress_pickle.dump((
        ress,
        ), f"{script_n}/{script_n}_{model}_{pattern_type}_{config.n_grcs}_{config.n_mfs}_{config.activation_level}_{config.pattern_len}_{config.n_random}.gz")
