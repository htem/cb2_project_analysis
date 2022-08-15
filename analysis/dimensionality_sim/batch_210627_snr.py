import argparse
import functools
import os
from batch_210622 import run_config
import compress_pickle
import make_graph_210618
import run_tests_210617

script_n = os.path.basename(__file__).split('.')[0]

os.makedirs(script_n, exist_ok=True)


'''
'''

ap = argparse.ArgumentParser()
# ap.add_argument("--n_random", type=int, help='', default=1)
ap.add_argument("--n_random", type=int, help='', default=40)
ap.add_argument("--pattern_len", type=int, help='', default=128*4)
ap.add_argument("--activation_level", type=float, help='', default=0.3)
# ap.add_argument("--noise_level", type=float, help='', default=0.05)
ap.add_argument("--variation_sizes", type=float, help='', nargs='+', default=None)
ap.add_argument("--model", type=str, help='', default='observed')
ap.add_argument("--n_grcs", type=int, help='', default=1847)
ap.add_argument("--n_mfs", type=int, help='', default=497)
# ap.add_argument("--redundant_factor", type=float, help='', default=1)
# ap.add_argument("--n_share", type=int, help='', default=2)
ap.add_argument("--pattern_type", type=str, help='', default='binary')
# ap.add_argument("--valence_dir", type=str, help='', default='01')
# ap.add_argument("--irrelevant_bits", type=str, help='', default='0')
# ap.add_argument("--invert_noise_mask", type=int, help='', default=0)
# ap.add_argument("--weight_type", type=str, help='', default='change')
ap.add_argument("--mfs_z_margin", type=int, help='', default=0)
# ap.add_argument("--top_mf_mask", type=float, help='', default=None)
# ap.add_argument("--mf_mask_limit", type=float, help='', default=None)
ap.add_argument("--grc_pct", type=float, help='', default=1.0)
ap.add_argument("--grc_pct_learned", type=int, help='', default=0)
ap.add_argument("--sub_feature_mode", type=int, help='', default=1)
# ap.add_argument("--synapse_failure", type=float, help='', default=1.0)
config = ap.parse_args()

run_tests_210617.NO_DIM_SIM = True

if config.variation_sizes is None:
    # begin = .1
    # end = .9
    # step = .2
    begin = .05
    end = 1.0
    step = .05
    # step = .5
    # step = .05
    mult=10000
    step_int = int(step*mult)
    config.variation_sizes = [k/mult for k in range(int(begin*mult), int((end+step)*mult), step_int)]

print(config.variation_sizes)

test_fn = functools.partial(run_tests_210617.test_across_noise,
                            grc_pct=config.grc_pct,
                            # synapse_fail_rate=config.synapse_failure,
                            noise_probs=config.variation_sizes,
                            grc_pct_learned=config.grc_pct_learned,
                            )

random_variation_kwargs = {
    'small_feature_mode': config.sub_feature_mode,
}

save_args = (f"zmargin_{config.mfs_z_margin}_"
             f"scale_{config.grc_pct}_"
             # f"fail_{config.synapse_failure}_"
             f"learned_{config.grc_pct_learned}_"
             f"sub_{config.sub_feature_mode}_"
            )

run_config(config, script_n, save_args=save_args, test_fn=test_fn,
            random_variation_kwargs=random_variation_kwargs,)
