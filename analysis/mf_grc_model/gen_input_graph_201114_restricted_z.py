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

sys.path.insert(0, '/n/groups/htem/Segmentation/shared-dev/cb2_segmentation/segway.graph.tmn7')
from segway.graph.synapse_graph import SynapseGraph
# from segway.graph.plot_adj_mat import plot_adj_mat

sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')
from tools import *
import tools_mf_graph

def to_ng_coord(coord):
    return (
        int(coord[0]/4),
        int(coord[1]/4),
        int(coord[2]/40),
        )

# def reload():
#     importlib.reload(tools_mf_graph)
#     # from tools_mf_graph import GrC
#     # from tools_mf_graph import MF
#     # from tools_mf_graph import GCLGraph

config_f = "config_grc_mf_201004.json"
# with open(config_f) as js_file:
#     minified = jsmin(js_file.read())
#     config = json.load(StringIO(minified))

overwrite = False
if "--overwrite" in sys.argv:
    overwrite = True

fout_postpend = ""

restricted = False
if "--restricted" in sys.argv:
    restricted = True
    fout_postpend += "_restricted_z"


graph = SynapseGraph(config_f, overwrite=overwrite)
g = graph.g
random.seed(0)
# syn_dict = get_syn_locs(graph)

'''Load data'''
import compress_pickle
fname = 'mf_grc_weights/grc-mf-locs-setup22-setup09-201114.gz'
grc_mfs_locs, grc_mfs_adj_counts = compress_pickle.load(fname)

# mfs_locs = defaultdict(list)
grc_mfs_locs_filtered = defaultdict(list)
# mfs_synapse_count = defaultdict(int)

for grc in grc_mfs_locs:
    if 'ml' in grc or 'tmn7_ml' in g.nodes[grc]['tags']:
        print(f"Skipped {grc}")
        continue
    xyz = get_node_pos(g, grc)
    if restricted:
        if xyz[0] < 90000*4 or xyz[0] > 130000*4:
            print(f"Skipped {grc} outside X (xyz={xyz})")
            continue
        if xyz[2] < 15000 or xyz[2] > 35000:
            print(f"Skipped {grc} outside X (xyz={xyz})")
            continue
    for mf in grc_mfs_locs[grc]:
        # mfs_locs[mf].extend(grc_mfs_locs[grc][mf])
        for loc in grc_mfs_locs[grc][mf]:
            grc_mfs_locs_filtered[grc].append((mf, loc))
        # mfs_synapse_count[mf] += grc_mfs_adj_counts[grc][mf]
    # grc_mfs_locs_filtered[grc] = grc_mfs_locs[grc]

'''load mfs_bouton_locs'''
import compress_pickle
mfs_bouton_locs = compress_pickle.load("mfs_bouton_locs_201114.gz")

input_graph = tools_mf_graph.GCLGraph()
input_graph.add_mfs(mfs_bouton_locs)

'''Make GT graph'''
for grc_id in grc_mfs_locs_filtered:
    xyz = get_node_pos(g, grc_id)
    input_graph.add_grc(grc_id, xyz, grc_mfs_locs_filtered[grc_id])

input_graph.finalize_gt()

# # test
# input_graph.randomize_graph(
#     preserve_mf_out_degree=True,
#     mf_dist_margin=100000,
#     )
# asdf

fout_name = f"input_graph_201114{fout_postpend}.gz"

'''Save graph'''
import compress_pickle
compress_pickle.dump((
    input_graph
    ), fout_name)

