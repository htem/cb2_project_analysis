import argparse
import os
import functools

from batch_210622 import run_config
import compress_pickle
import make_graph_210618
import run_tests_210617

script_n = os.path.basename(__file__).split('.')[0]

os.makedirs(script_n, exist_ok=True)


'''
'''

ap = argparse.ArgumentParser()
ap.add_argument("--n_random", type=int, help='', default=1)
# ap.add_argument("--n_random", type=int, help='', default=40)
ap.add_argument("--pattern_len", type=int, help='', default=128*4)
ap.add_argument("--activation_level", type=float, help='', default=0.3)
# ap.add_argument("--noise_level", type=float, help='', default=0.05)
ap.add_argument("--variation_sizes", type=float, help='', nargs='+', default=None)
ap.add_argument("--model", type=str, help='', default='observed')
ap.add_argument("--n_grcs", type=int, help='', default=1847)
ap.add_argument("--n_mfs", type=int, help='', default=497)
# ap.add_argument("--n_grcs", type=int, help='', default=1063)
# ap.add_argument("--n_mfs", type=int, help='', default=458)
# ap.add_argument("--n_grcs", type=int, help='', default=628)
# ap.add_argument("--n_mfs", type=int, help='', default=433)
# ap.add_argument("--n_grcs", type=int, help='', default=1847)
# ap.add_argument("--n_mfs", type=int, help='', default=497)
# ap.add_argument("--redundant_factor", type=float, help='', default=1)
# ap.add_argument("--n_share", type=int, help='', default=2)
ap.add_argument("--pattern_type", type=str, help='', default='binary')
# ap.add_argument("--valence_dir", type=str, help='', default='01')
# ap.add_argument("--irrelevant_bits", type=str, help='', default='0')
# ap.add_argument("--invert_noise_mask", type=int, help='', default=0)
# ap.add_argument("--weight_type", type=str, help='', default='change')
# ap.add_argument("--mfs_z_margin", type=int, help='', default=10000)
# ap.add_argument("--top_mf_mask", type=float, help='', default=None)
# ap.add_argument("--mf_mask_limit", type=float, help='', default=None)
ap.add_argument("--grc_pcts", type=float, help='', default=None)
ap.add_argument("--synapse_failure", type=float, help='', default=1.0)
ap.add_argument("--grc_scale", type=float, help='', default=1.0)

run_tests_210617.NO_DIM_SIM = True

config = ap.parse_args()

make_graph_210618.set_input_graph(
    '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/gen_db/mf_grc/input_graph_210611_grc_center_z_100_2_xlim_360000_600000_x_margin_20.gz'
    )

if config.grc_pcts is None:
    step = .05
    mult=1000
    step_int = int(step*mult)
    config.grc_pcts = [k/mult for k in range(step_int, int((1+step)*mult), step_int)]

print(config.variation_sizes)

test_fn = functools.partial(run_tests_210617.test_across_grc_pcts,
                            grc_pcts=config.grc_pcts,
                            grc_scale=config.grc_scale,
                            synapse_fail_rate=config.synapse_failure,
                            )

# save_args = (f"fail_{config.synapse_failure}_zmargin_{config.mfs_z_margin}")
save_args = (f"scale_{config.grc_scale}_fail_{config.synapse_failure}_")

run_config(config, script_n, test_fn=test_fn, save_args=save_args)
