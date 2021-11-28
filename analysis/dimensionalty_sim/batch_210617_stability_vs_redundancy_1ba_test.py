import argparse
import os
from batch_210621 import run_config

script_n = os.path.basename(__file__).split('.')[0]

os.makedirs(script_n, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--n_random", type=int, help='', default=40)
ap.add_argument("--pattern_len", type=int, help='', default=128*2)
ap.add_argument("--activation_level", type=float, help='', default=0.3)
ap.add_argument("--noise_level", type=float, help='', default=0.05)
ap.add_argument("--variation_sizes", type=float, help='', nargs='+', default=None)
ap.add_argument("--model", type=str, help='', default='global_random')
ap.add_argument("--n_grcs", type=int, help='', default=1459)
ap.add_argument("--n_mfs", type=int, help='', default=488)
ap.add_argument("--redundant_factor", type=float, help='', default=1)
ap.add_argument("--n_share", type=int, help='', default=2)
ap.add_argument("--pattern_type", type=str, help='', default='binary')
ap.add_argument("--valence_dir", type=str, help='', default='01')
ap.add_argument("--irrelevant_bits", type=str, help='', default='0')
ap.add_argument("--invert_noise_mask", type=int, help='', default=0)
ap.add_argument("--weight_type", type=str, help='', default='change')
ap.add_argument("--top_mf_mask", type=float, help='', default=None)
config = ap.parse_args()

if config.variation_sizes is None:
    # variation_sizes = [.1]
    # variation_sizes = [k/1000 for k in range(10, 210, 10)]
    config.variation_sizes = [.01, .02, .03, .04, .05, .06, .07, .08, .09, 
                       .10, .12, .14, .16, .18, .20]
print(config.variation_sizes)

run_config(config, script_n)
