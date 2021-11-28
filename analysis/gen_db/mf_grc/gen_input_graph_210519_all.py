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
import os

script_n = os.path.basename(__file__).split('.')[0]
script_n = script_n.split('_', 1)[1]

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

# config_f = "../../config_grc_mf_210407.json"
config_f = "/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/config_mf_grc_210429_setup01_v2.json"

# ap = argparse.ArgumentParser()
# ap.add_argument("--xlim", type=float, help='', default=(90, 140), nargs='+')
# ap.add_argument("--zlim", type=float, help='', default=(17, 27), nargs='+')
# config = ap.parse_args()
# xlim = config.xlim
# zlim = config.zlim

overwrite = False
if "--overwrite" in sys.argv:
    overwrite = True

graph = SynapseGraph(config_f, overwrite=overwrite)
g = graph.g


asdff = (79456, 112472, 846)

whitelist = set([
    (79456, 112472, 846),
    (91524, 113152, 858),
    (82188, 107752, 167),
    (79520, 105064, 153),
    (83068, 107308, 249),
    (87696, 108192, 223),
    (84452, 105472, 707),
    (78192, 117572, 594),
    (asdff),
    (asdff),
    (asdff),

])

'''Load data'''
import compress_pickle
fname = 'gen_210518_setup01_v2_syndb_threshold_20_coalesced.gz'
grc_mfs_syns = compress_pickle.load(fname)
grc_mfs_locs = defaultdict(lambda: defaultdict(list))

score_threshold = 100

for grc in grc_mfs_syns:
    for mf in grc_mfs_syns[grc]:
        for syn in grc_mfs_syns[grc][mf]:
            if syn['score'] > score_threshold:
                grc_mfs_locs[grc][mf].append(syn['syn_loc0'])

grc_mfs_locs_filtered = defaultdict(list)

for grc in grc_mfs_locs:
    if 'ml' in grc or 'tmn7_ml' in g.nodes[grc]['tags']:
        print(f"Skipped {grc}")
        continue
    if g.nodes[grc]['cell_type'] != 'grc':
        print(f"Cell type is not grc, skipping {grc}")
        continue
    xyz = get_node_pos(g, grc)
    # if xyz[0] < xlim[0]*1000*4 or xyz[0] > xlim[1]*1000*4:
        # print(f"Skipped {grc} outside X (xyz={xyz})")
        # continue
    # if xyz[2] < (zlim[0]*1000+40*70) or xyz[2] > (zlim[1]*1000+40*70):
    #     # print(f"Skipped {grc} outside Z (xyz={xyz})")
    #     continue
    for mf in grc_mfs_locs[grc]:
        for loc in grc_mfs_locs[grc][mf]:
            grc_mfs_locs_filtered[grc].append((mf, loc))

'''load mfs_bouton_locs'''
import compress_pickle
mfs_bouton_locs = compress_pickle.load("mf_locs_210518.gz")

input_graph = tools_mf_graph.GCLGraph()
input_graph.add_mfs(mfs_bouton_locs)

'''Make GT graph'''
for grc_id in grc_mfs_locs_filtered:
    xyz = get_node_pos(g, grc_id)
    # print(grc_mfs_locs_filtered[grc_id])
    input_graph.add_grc(grc_id, xyz, grc_mfs_locs_filtered[grc_id],
        min_synapses_per_bouton=2,
        verbose=1,
        )

# '''Make GT graph'''
# input_graph = tools_mf_graph.GCLGraph()
# input_graph.add_mfs(mfs_bouton_locs)

# for grc_id in grc_mfs_locs_filtered:
#     xyz = get_node_pos(g, grc_id)
#     # print(grc_mfs_locs_filtered[grc_id])
#     try:
#         input_graph.add_grc(grc_id, xyz, grc_mfs_locs_filtered[grc_id])
#     except:
#         pass

input_graph.finalize_gt()

fout_name = f"{script_n}.gz"

'''Save graph'''
import compress_pickle
compress_pickle.dump((
    input_graph
    ), fout_name)


asdf

def print_mf_locs(grc_id):
    for loc in grc_mfs_locs_filtered[grc_id]:
        print((loc[0], to_ng_coord(loc[1])))

def print_synapses(grc_id):
    for mf in grc_mfs_syns[grc_id]:
        print(mf)
        for syn in grc_mfs_syns[grc_id][mf]:
            print((syn['syn_loc'],
                    syn['score'],
                    ))

print_synapses('grc_3263')


# 100/3: seems to have quite a bit of FNs
# 100/2: more FPs than FNs






