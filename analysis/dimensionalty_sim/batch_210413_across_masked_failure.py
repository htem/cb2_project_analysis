import argparse
import random
import copy
import logging
import sys
import os

script_n = os.path.basename(__file__).split('.')[0]

# from run_tests_210327 import *
from run_tests_210413 import *

from shuffle_210404 import shuffle
from global_random_model2 import GlobalRandomModel

'''
'''

os.makedirs(script_n, exist_ok=True)


ap = argparse.ArgumentParser()
ap.add_argument("--n_random", type=int, help='', default=10)
ap.add_argument("--pattern_len", type=int, help='', default=512)
# ap.add_argument("--activation_levels", type=float, help='', nargs='+', default=None)
ap.add_argument("--activation_level", type=float, help='', default=0.3)
ap.add_argument("--failure_rates", type=float, help='', nargs='+', default=None)
# ap.add_argument("--noise_prob", type=float, help='', default=1)
ap.add_argument("--model", type=str, help='', default=None)
ap.add_argument("--scaled_noise", type=int, help='', default=0)
ap.add_argument("--core_noise", type=int, help='', default=0)
ap.add_argument("--n_grcs", type=int, help='', default=2400)
ap.add_argument("--n_mfs", type=int, help='', default=400)
ap.add_argument("--signal_ratio", type=float, help='', default=.5)
ap.add_argument("--signal_type", type=str, help='', default=None)
config = ap.parse_args()

# acts = config.activation_levels
# if acts is None:
#     acts = [k/100 for k in range(5, 100, 5)]
#     acts.insert(0, 0.01)
#     acts.append(0.99)
#     print(f'acts: {acts}')

failure_rates = config.failure_rates
if failure_rates is None:
    # failure_rates = [k/1000 for k in range(25, 1025, 25)]
    failure_rates = [k/1000 for k in range(50, 1000, 50)]
print(failure_rates)

n_grcs = config.n_grcs
n_mfs = config.n_mfs
model = config.model
assert model is not None

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

# compute signal mask
def get_top_mf_signal_mask(sim, signal_ratio, bottom=False):
    count = defaultdict(int)
    for grc in sim.grcs:
        for c in grc.claws:
            count[c] += 1
    count = list(count.items())
    count.sort(key=lambda x: x[1], reverse=not bottom)
    print(count)
    mask = [0]*len(sim.mfs)
    mask_len = int(signal_ratio*len(sim.mfs))
    for mfid in count[0:mask_len+1]:
        # print(mfid[0])
        mask[mfid[0]] = 1
    print(mask)
    return mask

def get_random_mf_signal_mask(sim, signal_ratio):
    mask = [0]*len(sim.mfs)
    mask_len = int(signal_ratio*len(sim.mfs))
    for i in range(mask_len):
        mask[i] = 1
    random.shuffle(mask)
    print(mask)
    return mask

if config.signal_type == 'top':
    signal_mask = get_top_mf_signal_mask(input_graph, config.signal_ratio)
elif config.signal_type == 'bottom':
    signal_mask = get_top_mf_signal_mask(input_graph, config.signal_ratio, bottom=True)
elif config.signal_type == 'random':
    signal_mask = get_random_mf_signal_mask(input_graph, config.signal_ratio)
# print(signal_mask)
else:
    assert False

def test(g, seed):
    return test_across_failure(
        g,
        print_output=True,
        test_len=config.pattern_len,
        # activation_levels=acts,
        activation_level=config.activation_level,
        failure_rates=failure_rates,
        # scaled_noise=config.scaled_noise,
        # core_noise=config.core_noise,
        # noise_prob=config.noise_prob,
        seed=seed,
        signal_mask=signal_mask,
        sim=input_graph,
        )

print(f'Running {model}')
ress = []
for n in range(config.n_random):
    print(f'Pass {n}')
    res = test(input_graph, seed=n)
    ress.append(res)
    compress_pickle.dump((
        ress,
        # ), f"{script_n}/{script_n}_{model}_{n_grcs}_{n_mfs}_scaled_noise_{config.scaled_noise}_core_noise_{config.core_noise}_{config.activation_level}_{config.pattern_len}_{config.n_random}.gz")
        ), f"{script_n}/{script_n}_{model}_{n_grcs}_{n_mfs}_signal_ratio_{config.signal_ratio}_signal_type_{config.signal_type}_{config.activation_level}_{config.pattern_len}_{config.n_random}.gz")
