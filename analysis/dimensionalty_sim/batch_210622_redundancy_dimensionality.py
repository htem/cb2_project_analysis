import argparse
import os
from batch_210622 import run_config
import compress_pickle
import make_graph_210618

script_n = os.path.basename(__file__).split('.')[0]

os.makedirs(script_n, exist_ok=True)

ap = argparse.ArgumentParser()
# ap.add_argument("--n_random", type=int, help='', default=1)
ap.add_argument("--n_random", type=int, help='', default=40)
ap.add_argument("--pattern_len", type=int, help='', default=128*4)
ap.add_argument("--activation_level", type=float, help='', default=0.3)
# ap.add_argument("--noise_level", type=float, help='', default=0.05)
ap.add_argument("--variation_sizes", type=float, help='', nargs='+', default=None)
ap.add_argument("--model", type=str, help='', default='local_random')
ap.add_argument("--n_grcs", type=int, help='', default=1459)
ap.add_argument("--n_mfs", type=int, help='', default=488)
# ap.add_argument("--n_grcs", type=int, help='', default=1063)
# ap.add_argument("--n_mfs", type=int, help='', default=458)
# ap.add_argument("--n_grcs", type=int, help='', default=628)
# ap.add_argument("--n_mfs", type=int, help='', default=433)
# ap.add_argument("--n_grcs", type=int, help='', default=1847)
# ap.add_argument("--n_mfs", type=int, help='', default=497)
ap.add_argument("--redundant_factor", type=float, help='', default=1)
ap.add_argument("--n_share", type=int, help='', default=2)
ap.add_argument("--pattern_type", type=str, help='', default='binary')
# ap.add_argument("--valence_dir", type=str, help='', default='01')
# ap.add_argument("--irrelevant_bits", type=str, help='', default='0')
# ap.add_argument("--invert_noise_mask", type=int, help='', default=0)
# ap.add_argument("--weight_type", type=str, help='', default='change')
# ap.add_argument("--top_mf_mask", type=float, help='', default=None)
config = ap.parse_args()

# make_graph_210618.set_input_graph(
#     compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/gen_db/mf_grc/input_graph_210611_grc_center_z_100_2_xlim_360000_600000_x_margin_20_z_margin_10.gz')
#     )

if config.variation_sizes is None:
    step = .1
    mult=1000
    step_int = int(step*mult)
    config.variation_sizes = [k/mult for k in range(step_int, int((1+step)*mult), step_int)]
print(config.variation_sizes)

run_config(config, script_n)
