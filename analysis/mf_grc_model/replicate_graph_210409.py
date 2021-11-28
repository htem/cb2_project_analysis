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

# sys.path.insert(0, '/n/groups/htem/Segmentation/shared-dev/cb2_segmentation/segway.graph.tmn7')
# from segway.graph.synapse_graph import SynapseGraph
sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')
# from tools import *
import tools_mf_graph


ap = argparse.ArgumentParser()
ap.add_argument("--shuffle_model", type=str, help='', default=None)
ap.add_argument("--n_replicates", type=int, help='', default=1)
ap.add_argument("--xlim", type=float, help='', default=(90, 140), nargs='+')
ap.add_argument("--zlim", type=float, help='', default=(4, 40), nargs='+')
ap.add_argument("--wrapz", type=int, help='', default=1)
config = ap.parse_args()

fname = ('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/'\
                                   f'mf_grc_model/input_graph_210407_xlim_{config.xlim[0]}_{config.xlim[1]}_zlim_{config.zlim[0]}_{config.zlim[1]}_exclude_oob_edges.gz')
input_graph = compress_pickle.load(fname)

if config.shuffle_model:
    tools_mf_graph.shuffle(input_graph, config.shuffle_model)
    inferred_edges = compress_pickle.load(f'inferred_edges_210409a_{config.shuffle_model}.gz')
else:
    inferred_edges = compress_pickle.load('inferred_edges_210409a.gz')
input_graph.inferred_edges_joint_probability = inferred_edges
input_graph.replicate(config.zlim[0]*1000+70*40, config.zlim[1]*1000+70*40, config.n_replicates, config.wrapz)

if not config.wrapz:
    save_fname = f'input_graph_210407_xlim_{config.xlim[0]}_{config.xlim[1]}_zlim_{config.zlim[0]}_{config.zlim[1]}_rep_{config.n_replicates}'
else:
    save_fname = f'input_graph_210407_xlim_{config.xlim[0]}_{config.xlim[1]}_zlim_{config.zlim[0]}_{config.zlim[1]}_rep_{config.n_replicates}_wrapz'

if config.shuffle_model:
    save_fname += f'_{config.shuffle_model}'
    
print(f'Saving to {save_fname}')
compress_pickle.dump(input_graph, save_fname+'.gz')
