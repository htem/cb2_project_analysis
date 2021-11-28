# import collections
# from collections import defaultdict
# import sys
# import json
# import random
# from jsmin import jsmin
# from io import StringIO
# import numpy as np
# import copy
# import importlib

# sys.path.insert(0, '/n/groups/htem/Segmentation/shared-dev/cb2_segmentation/segway.graph.tmn7')
# from segway.graph.synapse_graph import SynapseGraph
# # from segway.graph.plot_adj_mat import plot_adj_mat

# sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')
# from tools import *
# import tools_mf_graph

# n_mfs = 7000
# n_grcs = 204000
# input_graph = tools_mf_graph.GCLGraph()
# input_graph.min_synapses_per_bouton = 0

# mfs_list = []
# mfs_dict = {}
# for i in range(n_mfs):
#     mf_id = f'mf_{i}'
#     mfs_list.append((mf_id, [(i, i, i)]))
#     mfs_dict[mf_id] = [(i, i, i)]

# input_graph.add_mfs(mfs_dict)

# for j in range(n_grcs):
#     grc_id = f'grc_{j}'
#     xyz = (j+i+1, j+i+1, j+i+1)
#     for k in range(4):
#         mf_i = int(random.random()*len(mfs_list))
#         # mf_loc = mfs_list[mf_i][1]
#         mf_loc = [(mfs_list[mf_i][0], mfs_list[mf_i][1][0])]
#         input_graph.add_grc(grc_id, xyz, mf_loc, compute_rosette_distances=False)
#         # input_graph.add_grc(grc_id, xyz, mf_loc, compute_rosette_distances=True)

# input_graph.finalize_gt()

# # # test
# # input_graph.randomize_graph(
# #     preserve_mf_out_degree=True,
# #     mf_dist_margin=100000,
# #     )
# # asdf

# fout_name = f"input_graph_fake_7k_204k.gz"

# '''Save graph'''
# import compress_pickle
# compress_pickle.dump((
#     input_graph
#     ), fout_name)

