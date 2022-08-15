import argparse
import functools
import os
from config_220719a_dim import run_config
import compress_pickle
import run_tests_210617 as run_tests

script_n = os.path.basename(__file__).split('.')[0]

os.makedirs(script_n, exist_ok=True)

ap = argparse.ArgumentParser()
# ap.add_argument("--n_random", type=int, help='', default=40)
ap.add_argument("--seed", type=int, help='', default=0)
ap.add_argument("--pattern_len", type=int, help='', default=128*4)
ap.add_argument("--activation_level", type=float, help='', default=0.3)
ap.add_argument("--variation_sizes", type=float, help='', nargs='+', default=None)
ap.add_argument("--model", type=str, help='', default='local_random_expanded2')
ap.add_argument("--n_grcs", type=int, help='', default=2541)
ap.add_argument("--n_mfs", type=int, help='', default=771)
# ap.add_argument("--dendrite_length", type=int, help='', default=21000)
ap.add_argument("--pattern_type", type=str, help='', default='binary')

ap.add_argument("--overgrow_factor", type=float, help='', default=1.2)
ap.add_argument("--clumpy_mfs", type=int, help='', default=0)
ap.add_argument("--increase_sharing", type=int, help='', default=0)

config = ap.parse_args()

if config.variation_sizes is None:
    begin = .025
    end = 1.0
    # end = .025
    step = .025
    mult = 10000
    step_int = int(step*mult)
    config.variation_sizes = [k/mult for k in range(int(begin*mult), int((end+step)*mult), step_int)]

print(config.variation_sizes)

test_fn = functools.partial(run_tests.test_across_noise,
                            # grc_pct=config.grc_pct,
                            # synapse_fail_rate=config.synapse_failure,
                            noise_probs=config.variation_sizes,
                            # grc_pct_learned=config.grc_pct_learned,
                            )

# random_variation_kwargs = {
#     'small_feature_mode': config.sub_feature_mode,
# }

save_args = (
            # f"zmargin_{config.mfs_z_margin}_"
             # f"scale_{config.grc_pct}_"
             # f"fail_{config.synapse_failure}_"
             # f"learned_{config.grc_pct_learned}_"
             # f"sub_{config.sub_feature_mode}_"
            )

save_args = ''

model_desc, res =run_config(
            config, script_n, save_args=save_args, test_fn=test_fn,
            seed=config.seed,
            # random_variation_kwargs=random_variation_kwargs,
            )

dirout = (f"{script_n}/{script_n}_{model_desc}_"
          f"{config.pattern_type}_{config.n_grcs}_{config.n_mfs}_"
          f"{save_args}"
          f"{config.activation_level}_{config.pattern_len}")
fout_name = dirout+f"/{config.seed}.gz"

os.makedirs(dirout, exist_ok=True)
assert model_desc is not None
compress_pickle.dump((
    res,
    ), fout_name)

# model_desc, res = test(model, seed=n)
# # ress.append(res)
# assert model_desc is not None
# compress_pickle.dump((
#     res,
#     ), f"{script_n}/{script_n}_{model_desc}_"
#        f"{pattern_type}_{config.n_grcs}_{config.n_mfs}_"
#        f"{save_args}"
#        f"{config.activation_level}_{config.pattern_len}_{n}.gz")
# print()