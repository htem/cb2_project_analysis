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

config_f = "config_grc_mf_210407.json"
# with open(config_f) as js_file:
#     minified = jsmin(js_file.read())
#     config = json.load(StringIO(minified))

ap = argparse.ArgumentParser()
ap.add_argument("--xlim", type=float, help='', default=(90, 140), nargs='+')
ap.add_argument("--zlim", type=float, help='', default=(17, 27), nargs='+')
ap.add_argument("--include_oob_edges", type=int, help='', default=True)
config = ap.parse_args()
xlim = config.xlim
zlim = config.zlim

overwrite = False
if "--overwrite" in sys.argv:
    overwrite = True

graph = SynapseGraph(config_f, overwrite=overwrite)
g = graph.g
random.seed(0)
# syn_dict = get_syn_locs(graph)


'''Load data'''
import compress_pickle
fname = 'mf_grc_weights/grc-mf-locs-setup22-setup09-210407.gz'
grc_mfs_locs, grc_mfs_adj_counts = compress_pickle.load(fname)

# mfs_locs = defaultdict(list)
grc_mfs_locs_filtered = defaultdict(list)
# mfs_synapse_count = defaultdict(int)

def within_bound(loc, xlim, zlim):
    if loc[0] < xlim[0]*1000*4 or loc[0] > xlim[1]*1000*4:
        return False
    if loc[2] < (zlim[0]*1000+40*70) or loc[2] >= (zlim[1]*1000+40*70):
        return False
    return True

for grc in grc_mfs_locs:
    if 'ml' in grc or 'tmn7_ml' in g.nodes[grc]['tags']:
        print(f"Skipped {grc}")
        continue
    if g.nodes[grc]['cell_type'] != 'grc':
        print(f"Cell type is not grc, skipping {grc}")
        continue
    xyz = get_node_pos(g, grc)
    if not within_bound(xyz, xlim, zlim):
        continue
    for mf in grc_mfs_locs[grc]:
        for loc in grc_mfs_locs[grc][mf]:
            if config.include_oob_edges or within_bound(loc, xlim, zlim):
                grc_mfs_locs_filtered[grc].append((mf, loc))

'''load mfs_bouton_locs'''
import compress_pickle
mfs_bouton_locs = compress_pickle.load("mfs_bouton_locs_210407.gz")

input_graph = tools_mf_graph.GCLGraph()
input_graph.add_mfs(mfs_bouton_locs)

'''Make GT graph'''
for grc_id in grc_mfs_locs_filtered:
    xyz = get_node_pos(g, grc_id)
    input_graph.add_grc(grc_id, xyz, grc_mfs_locs_filtered[grc_id])

input_graph.finalize_gt()

print(f'{len(input_graph.grcs)} grcs')

# # test
# input_graph.randomize_graph(
#     preserve_mf_out_degree=True,
#     mf_dist_margin=100000,
#     )
# asdf

fout_name = f"input_graph_210407_xlim_{xlim[0]}_{xlim[1]}_zlim_{zlim[0]}_{zlim[1]}"
if not config.include_oob_edges:
    fout_name += '_exclude_oob_edges'

'''Save graph'''
import compress_pickle
compress_pickle.dump((
    input_graph
    ), fout_name+'.gz')

