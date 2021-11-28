import random
import copy
import logging
import sys

sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')

from neurons import GranuleCell, MossyFiber, Simulation
import analysis

'''Load data'''
import compress_pickle
# fname = '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114.gz'
input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114_restricted_z.gz')
# input_graph = compress_pickle.load(fname)

input_graph.randomize_graph_by_grc(
        mf_dist_margin=4000,
        single_connection_per_pair=True,
        # constant_grc_degree=4,
        # constant_dendrite_length=15000,
        # always_pick_closest_rosette=True,
        preserve_mf_degree=True,
    # preserve_mf_degree=True,
    # approximate_in_degree=True,
    # local_lengths=True,
    )

# logging.basicConfig(level=logging.DEBUG)
removed = input_graph.remove_empty_mfs()
print(f'Removed {len(removed)} mfs')

random.seed(0)
sim = Simulation(
    input_graph=input_graph,
    )

# num mfs = 339
n_pattern = 1024  # 248
n_pattern = 1024*4  # 309
# n_pattern = 1024*8  # 323
# n_pattern = 10240  # 326.18270708754324
patterns = sim.generate_patterns(count=n_pattern)
print(f'len(patterns): {len(patterns)}')
# print(patterns)
# print(patterns[-1])

# patterns_0 = copy.deepcopy(patterns)
# for i in range(0):
#     ps = []
#     for j in range(len(patterns_0)):
#         p = sim.add_input_noise(patterns_0[j][0], .3)
#         ps.append((p, patterns_0[j][1]))
#     patterns.extend(ps)
# print(f'len(patterns): {len(patterns)}')
# # sim.print_grc_weights()
# # print(patterns[-1])

sim.reset()
# sim.grc_act_threshold = .999
# sim.train(patterns)
# sim.evaluate(patterns, no_random=True)
sim.evaluate(patterns, no_random=True, calibrate_activation_level=.1)
sim.evaluate(patterns, no_random=True)
dim = analysis.get_dim_from_acts(sim.get_mfs_activities())
print(dim)
dim = analysis.get_dim_from_acts(sim.get_grc_activities())
print(dim)
