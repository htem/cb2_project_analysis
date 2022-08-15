import jax
jax.config.update('jax_platform_name', 'cpu')
import sys
import shutil
import argparse
import os
import functools
import compress_pickle
import random

sys.path.insert(0, '/n/groups/htem/Segmentation/tmn7/gunpowder-1.3-220316')
import gunpowder as gp
from gunpowder.jax import Train, Predict

import node_source_model

# from networks import LinearModel, SigmoidLinearModel
from networks import SigmoidPosLinearModel2 as SigmoidPosLinearModel
from source_models import RandomModel, MfGcModel, MfGcModelPartial
from source_models import MfGcModelPartialWithCalibration2 as MfGcModelPartialWithCalibration
from eval_fn import eval220317
from neurons import Simulation
sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')

outdir = os.path.basename(__file__.split('.')[0])

ap = argparse.ArgumentParser()
ap.add_argument("testcase", type=str, help='')
ap.add_argument("--lr", type=float, default=0.02)
ap.add_argument("--variability", type=float, default=0.4)
ap.add_argument("--num_patterns", type=int, default=150)
ap.add_argument("--test_noise", type=float, default=0.05)
ap.add_argument("--keep_pct", type=float, default=1.0)
ap.add_argument("--sigmoid_scale", type=float, default=16)
ap.add_argument("--momentum", type=float, default=0.9)
# ap.add_argument("--momentum", type=float, default=None)
ap.add_argument("--trial", type=int, default=0)
ap.add_argument("--num_epochs", type=int, default=300)
ap.add_argument("--selective", type=int, default=0)
config = ap.parse_args()
arg_config = vars(ap.parse_args())
for k, v in arg_config.items():
    globals()[k] = v

graph_path = '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/gen_db/mf_grc/input_graph_210611_grc_center_z_100_2_xlim_360000_600000_x_margin_20_z_margin_5.0.gz'
input_graph = compress_pickle.load(graph_path)
sim = Simulation(input_graph=input_graph)

batch_size = num_patterns
testcase = f'{testcase}_{keep_pct}_{variability}_{num_patterns}_{sigmoid_scale}_{test_noise}_{lr}_{momentum}_{num_epochs}_{selective}_{trial}'
log_dir = f'{outdir}/log_{testcase}'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(f'{outdir}/checkpoints', exist_ok=True)
# source_model = RandomModel(num_patterns, pattern_len, 0)
if selective:
    source_model = MfGcModelPartialWithCalibration(
        sim, num_patterns, variability=variability, seed=trial,
        test_noise=test_noise,
        keep_pct=keep_pct,
        )
else:
    source_model = MfGcModelPartial(
        sim, num_patterns, variability=variability, seed=trial,
        test_noise=test_noise,
        keep_pct=keep_pct,
        )
# pattern_len = sim.num_mfs
pattern_len = int(sim.num_grcs*keep_pct)

network = SigmoidPosLinearModel(
    is_training=False, lr=lr, momentum=momentum,
    act_rate=0.3,
    sigmoid_scale=sigmoid_scale)

pattern = gp.ArrayKey('pattern')
cls = gp.ArrayKey('cls')

source = node_source_model.SourceModel(
    source_model,
    {
        pattern: 'pattern',
        cls: 'cls',
    },
    batch_size=batch_size,
)

pipeline = source

validation_fn = functools.partial(eval220317,
                                  source=source_model,
                                  batch_size=num_patterns,
                                  seed=trial,
                                  log_dir=log_dir,
                                  )

# pipeline += PrintSeed()

pipeline += gp.PreCache(
        cache_size=16,
        num_workers=8)

# pipeline += gp.PreCache(
#         cache_size=32,
#         num_workers=16)

pipeline += Train(
    model=network,
    inputs={"pattern": pattern,
            "cls": cls},
    outputs={},
    checkpoint_basename=f'{outdir}/checkpoints/{testcase}',
    save_every=num_epochs,
    validation_fn=validation_fn,
    validation_every=100,
    # spawn_subprocess=True,
    # n_devices=1,
    log_dir=log_dir,
)

pipeline += gp.PrintProfilingStats(every=100)


request = gp.BatchRequest()
request[pattern] = gp.Roi((0,0), (batch_size, pattern_len))
request[cls] = gp.Roi((0,0), (batch_size, 1))

random.seed(42)

with gp.build(pipeline):
    for i in range(num_epochs):
        batch = pipeline.request_batch(request)
        # if i % 100 == 0:
        #     # print('.', end='')
        #     print(batch[pattern].data)
        #     print(batch[cls].data)

print(f'results: {log_dir}/results')
