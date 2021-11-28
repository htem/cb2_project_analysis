import random
import copy
import logging
import sys
import os

script_n = os.path.basename(__file__).split('.')[0]

from run_tests_201204 import *

'''Load data'''
import compress_pickle
# fname = '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114.gz'
# input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114_restricted_z.gz')
input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114_xlim_320_560_zlim_10_40.gz')
# input_graph = compress_pickle.load(fname)

def shuffle(
        graph, model,
        num_claws=4,
        mf_dist_margin=4000,
        constant_dendrite_length=15000,
        ):
    random.seed(0)
    if model == 'constant_grc':
        graph.randomize_graph_by_grc(
            mf_dist_margin=mf_dist_margin,
            single_connection_per_pair=True,
            constant_grc_degree=num_claws,
            preserve_mf_degree=True,
        )
    if model == 'constant_length':
        graph.randomize_graph_by_grc(
            mf_dist_margin=mf_dist_margin,
            single_connection_per_pair=True,
            constant_dendrite_length=constant_dendrite_length,
            preserve_mf_degree=True,
        )
    if model == 'classic_random':
        graph.randomize_graph(
            random_model=True,
            constant_grc_degree=num_claws,
            )
    if model == 'no_same_mf':
        graph.randomize_graph_by_grc(
            mf_dist_margin=mf_dist_margin,
            single_connection_per_pair=True,
            )
    if model == 'naive_random':
        graph.randomize_graph_by_grc(
            mf_dist_margin=mf_dist_margin,
            single_connection_per_pair=True,
            constant_grc_degree=4,
            constant_dendrite_length=constant_dendrite_length,
            always_pick_closest_rosette=True,
            )
    return graph


db = {}
try:
    db = compress_pickle.load(f'{script_n}.gz')
    db = db[0]
except:
    db = {}

# for model in ['Data', 'classic_random', 'naive_random']:
for model in ['data', 'classic_random']:
    print(f'Running {model}')
    g = shuffle(input_graph, model=model)
    res = test_similarity_by_activation_level_201205(g, print_output=True)
    db[model] = {}
    db[model][0] = res

compress_pickle.dump((
    db,
    ), f"{script_n}.gz")


for model in ['naive_random']:
    print(f'Running {model}')
    for dendrite_len in [17, 15]:
        print(f'dendrite_len: {dendrite_len}')
        for mf_dist_margin in [4]:
            print(f'mf_dist_margin: {mf_dist_margin}')
            g = shuffle(input_graph, model=model,
                mf_dist_margin=mf_dist_margin*1000,
                constant_dendrite_length=dendrite_len*1000,
                )
            res = test_similarity_by_activation_level_201205(g, print_output=False)
            db[f'{model}_{dendrite_len}_{mf_dist_margin}'] = {}
            db[f'{model}_{dendrite_len}_{mf_dist_margin}'][0] = res


compress_pickle.dump((
    db,
    ), f"{script_n}.gz")



# for model in ['naive_random']:
#     print(f'Running {model}')
#     for dendrite_len in [16, 18, 19, 20]:
#         print(f'dendrite_len: {dendrite_len}')
#         for mf_dist_margin in [4]:
#             print(f'mf_dist_margin: {mf_dist_margin}')
#             g = shuffle(input_graph, model=model,
#                 mf_dist_margin=mf_dist_margin*1000,
#                 constant_dendrite_length=dendrite_len*1000,
#                 )
#             res = test_similarity_by_activation_level_201205(g, print_output=False)
#             db[f'{model}_{dendrite_len}_{mf_dist_margin}'] = {}
#             db[f'{model}_{dendrite_len}_{mf_dist_margin}'][0] = res


# compress_pickle.dump((
#     db,
#     ), f"{script_n}.gz")


# if '--naive_random' in sys.argv:
#     input_graph.randomize_graph_by_grc(
#         single_connection_per_pair=True,
#         constant_grc_degree=4,
#         # constant_dendrite_length=20000,
#         # constant_dendrite_length=,
#         always_pick_closest_rosette=True,
#         )
