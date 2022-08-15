import argparse
import functools
import os
from config_220724_snr import run_config
import compress_pickle
import run_tests_220725_snr as run_tests

script_n = os.path.basename(__file__).split('.')[0]

os.makedirs(script_n, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, help='', default=0)
ap.add_argument("--activation_level", type=float, help='', default=0.3)
ap.add_argument("--model", type=str, help='', default='local_random_expanded2')
ap.add_argument("--pattern_type", type=str, help='', default='binary')
# ap.add_argument("--n_grcs", type=int, help='', default=1847)
# ap.add_argument("--n_mfs", type=int, help='', default=497)
ap.add_argument("--n_grcs", type=int, help='', default=2541)
ap.add_argument("--n_mfs", type=int, help='', default=771)
ap.add_argument("--num_patterns", type=int, help='', default=8)
ap.add_argument("--test_mult", type=int, help='', default=10)
# ap.add_argument("--input_noise_pct", type=float, help='', default=10.0)
ap.add_argument("--pattern_size_pct", type=float, help='', default=30.0)
ap.add_argument("--unrelated_mfs_pct", type=float, help='', default=0.0)
ap.add_argument("--synapse_pct", type=float, help='', default=50.0)
ap.add_argument("--noise_during_training", type=int, help='', default=1)
# ap.add_argument("--dendrite_length", type=int, help='', default=21000)
ap.add_argument("--overgrow_factor", type=float, help='', default=1.2)
ap.add_argument("--clumpy_mfs", type=int, help='', default=0)
ap.add_argument("--increase_sharing", type=int, help='', default=0)
ap.add_argument("--skip_finished", type=int, help='', default=0)

config = ap.parse_args()

if config.skip_finished:
    model_desc = f'local_random_expanded2_1.2_{config.increase_sharing}_{config.clumpy_mfs}'
    save_args = (
                    f"{config.num_patterns}_"
                    f"size_{config.pattern_size_pct}_"
                )
    dirout = (f"{script_n}/{script_n}_{model_desc}_"
              f"{config.pattern_type}_{config.n_grcs}_{config.n_mfs}_"
              f"{save_args}"
              f"{config.activation_level}")
    fout_name = dirout+f"/{config.seed}.gz"
    if os.path.exists(fout_name):
        print(f'{fout_name} ran already.')
        exit()

run_tests.NO_DIM_SIM = True

def make_pct_seq(begin, end, step):
    begin = float(begin)
    end = float(end)
    mult = 10000
    return [k/mult*100 for k in range(int(begin*mult), int((end+step)*mult), int(step*mult))]

noise_vals = make_pct_seq(0, 1, 0.05)
synapse_pct_vals = make_pct_seq(0.025, 0.5, 0.025)

test_fn = functools.partial(run_tests.test_snr_across_noise,
                            num_patterns=config.num_patterns,
                            pattern_size_pct=config.pattern_size_pct,
                            unrelated_mfs_pct=config.unrelated_mfs_pct,
                            # input_noise_pct=config.input_noise_pct,
                            noise_during_training=config.noise_during_training,
                            synapse_pct=config.synapse_pct,
                            test_mult=config.test_mult,
                            loop_vals=noise_vals,
                            synapse_pcts=synapse_pct_vals,
                            )

save_args = (
                f"{config.num_patterns}_"
                f"size_{config.pattern_size_pct}_"
            )

model_desc, res = run_config(
    config, script_n, save_args=save_args, test_fn=test_fn,
    # batch_size=config.batch_size,
    seed=config.seed,
            # random_variation_kwargs=random_variation_kwargs,
            )

dirout = (f"{script_n}/{script_n}_{model_desc}_"
          f"{config.pattern_type}_{config.n_grcs}_{config.n_mfs}_"
          f"{save_args}"
          f"{config.activation_level}")
fout_name = dirout+f"/{config.seed}.gz"

os.makedirs(dirout, exist_ok=True)
assert model_desc is not None
compress_pickle.dump((
    res,
    ), fout_name)
