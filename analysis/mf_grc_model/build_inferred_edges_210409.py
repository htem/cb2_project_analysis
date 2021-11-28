import collections
from collections import defaultdict
import sys
import json
import random
from jsmin import jsmin
from io import StringIO
import numpy as np
import copy
import importlib
import argparse
import compress_pickle

sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')
import tools_mf_graph


ap = argparse.ArgumentParser()
ap.add_argument("--shuffle_model", type=str, help='', default=None)
# ap.add_argument("--xlim", type=float, help='', default=(90, 140), nargs='+')
# ap.add_argument("--zlim", type=float, help='', default=(4, 40), nargs='+')
# ap.add_argument("--wrapz", type=int, help='', default=1)
config = ap.parse_args()

fname = ('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/'\
                                   'mf_grc_model/input_graph_210407_xlim_90_140_full.gz')
input_graph = compress_pickle.load(fname)

if config.shuffle_model:
    tools_mf_graph.shuffle(input_graph, config.shuffle_model)

input_graph.inferred_edges_joint_probability = None
bucket_size = 10000
bottom_edge = 7500+70*40
top_edge = 44000+70*40-7500
input_graph.build_inferred_edges_joint_probability(
    (bottom_edge, bottom_edge+bucket_size),
    (top_edge-bucket_size, top_edge),
    height_bucket_size=10000,
)
fname = 'inferred_edges_210409a'
if config.shuffle_model:
    fname += f'_{config.shuffle_model}'
compress_pickle.dump(
    input_graph.inferred_edges_joint_probability, fname+'.gz')
