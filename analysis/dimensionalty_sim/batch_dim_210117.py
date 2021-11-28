import random
import copy
import logging
import sys

from run_tests_201204 import *

'''Load data'''
import compress_pickle
# fname = '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114.gz'
# input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114_restricted_z.gz')
input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114_xlim_320_560_zlim_10_40.gz')
# input_graph = compress_pickle.load(fname)

from shuffle import shuffle

db = {}


# for model in ['Data', 'classic_random', 'naive_random']:
for model in ['data', 'classic_random']:
    print(f'Running {model}')
    g = shuffle(input_graph, model=model,
        )
    res = test_dim_201205(g,
        print_output=True
        )
    db[model] = {}
    db[model][0] = res

# compress_pickle.dump((
#     db,
#     ), f"batch_dim_210117_db.gz")


# for model in ['Data', 'classic_random', 'naive_random']:
for model in ['expanded_random_30', 'expanded_random_50']:
    print(f'Running {model}')
    g = shuffle(input_graph, model=model)
    res = test_dim_201205(g, print_output=False)
    db[model] = {}
    db[model][0] = res

# compress_pickle.dump((
#     db,
#     ), f"batch_dim_210117_db.gz")


for model in ['naive_random']:
    print(f'Running {model}')
    # for dendrite_len in [15, 16, 17, 18, 19, 20]:
    for dendrite_len in [17]:
        print(f'dendrite_len: {dendrite_len}')
        # for mf_dist_margin in [1, 2, 3, 4, 5]:
        for mf_dist_margin in [1, 4]:
            print(f'mf_dist_margin: {mf_dist_margin}')
            g = shuffle(input_graph, model=model,
                mf_dist_margin=mf_dist_margin*1000,
                constant_dendrite_length=dendrite_len*1000,
                )
            res = test_dim_201205(g, print_output=False)
            db[f'{model}_{dendrite_len}_{mf_dist_margin}'] = {}
            db[f'{model}_{dendrite_len}_{mf_dist_margin}'][0] = res


compress_pickle.dump((
    db,
    ), f"batch_dim_210117_db.gz")

