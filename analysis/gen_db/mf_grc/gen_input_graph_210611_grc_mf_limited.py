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
config_f = "/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/config_mf_grc_210611_setup01_v2.json"

ap = argparse.ArgumentParser()
ap.add_argument("--xlim", type=float, help='', default=[90, 150], nargs='+')
ap.add_argument("--ylim", type=float, help='', default=None, nargs='+')
ap.add_argument("--zlim", type=float, help='', default=None, nargs='+')
ap.add_argument("--score_threshold", type=int, default=100)
ap.add_argument("--min_synapses", type=int, default=2)
ap.add_argument("--min_claws", type=int, default=2)
config = ap.parse_args()

xlim = config.xlim
ylim = config.ylim
zlim = config.zlim
# score_threshold = 100
if xlim is not None:
    xlim[0] *= 4000
    xlim[1] *= 4000
if ylim is not None:
    ylim[0] *= 4000
    ylim[1] *= 4000
if zlim is not None:
    assert False
    zlim[0] *= 4000
    zlim[1] *= 4000


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
    # 210611
    # these synapses are single but verified to be a claw
    (149892, 108212, 208),
    (95284, 113804, 741),
    (95480, 110016, 643),
    (140436, 96872, 988),
    (113960, 118892, 180),
    (97296, 110080, 714),
    (92632, 108736, 384),
    (96660, 115156, 958),
    (98632, 116900, 468),
    (104712, 119372, 682),
    (124928, 117224, 632),
    (125964, 117096, 638),
    (113040, 110072, 72),
    (105656, 107304, 660),
    (142944, 108928, 788),
    (130400, 98832, 745),
    (115196, 117820, 633),
    (122204, 106992, 208),
    (142068, 103100, 423),
    (135512, 105464, 156),
    (137056, 110092, 201),
    (125188, 112260, 564),
    (149328, 105848, 1141),
    (143244, 106784, 81),
    (144428, 102808, 579),
    (155048, 100648, 533),
    (108816, 115376, 83),
    (149932, 96328, 1109),
    (143052, 114120, 1137),
    (109792, 113424, 1154),
    (asdff),
])

# these are MFs that are almost out of volume bounds and so do not have many synapses
mf_whitelist = set([
    'mf_719', 'mf_572', 'mf_475', 'mf_478', 'mf_481', 'mf_724'
])

'''Load data'''
import compress_pickle
fname = 'gen_210518_setup01_v2_syndb_threshold_20_coalesced.gz'
grc_mfs_syns = compress_pickle.load(fname)
grc_mfs_locs = defaultdict(lambda: defaultdict(list))

for grc in grc_mfs_syns:
    for mf in grc_mfs_syns[grc]:
        for syn in grc_mfs_syns[grc][mf]:
            if syn['score'] > config.score_threshold:
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

    if xlim and (xyz[0] < xlim[0] or xyz[0] > xlim[1]):
        print(f"Skipped {grc} outside X (xyz={xyz})")
        continue
    if zlim and (xyz[2] < zlim[0] or xyz[2] > zlim[1]):
        print(f"Skipped {grc} outside Z (xyz={xyz})")
        continue

    for mf in grc_mfs_locs[grc]:
        for loc in grc_mfs_locs[grc][mf]:
            grc_mfs_locs_filtered[grc].append((mf, loc))

'''load mfs_bouton_locs'''
import compress_pickle
mfs_bouton_locs = compress_pickle.load("mf_locs_210518.gz")

importlib.reload(tools_mf_graph)
input_graph = tools_mf_graph.GCLGraph()
input_graph.add_mfs(mfs_bouton_locs, xlim=xlim, zlim=zlim)
'''Make GT graph'''
for grc_id in grc_mfs_locs_filtered:
    xyz = get_node_pos(g, grc_id)
    # print(grc_mfs_locs_filtered[grc_id])
    input_graph.add_grc(grc_id, xyz, grc_mfs_locs_filtered[grc_id],
        min_synapses_per_bouton=config.min_synapses,
        min_claws=config.min_claws,
        # verbose=1,
        verbose=config.min_synapses-1,
        mf_whitelist=mf_whitelist,
        synapse_whitelist=whitelist,
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
input_graph.remove_empty_mfs()

fout_name = f"{script_n}_{config.score_threshold}_{config.min_synapses}"
if xlim:
    fout_name += f'_xlim_{xlim[0]}_{xlim[1]}'
if ylim:
    fout_name += f'_ylim_{ylim[0]}_{ylim[1]}'
if zlim:
    fout_name += f'_zlim_{zlim[0]}_{zlim[1]}'

'''Save graph'''
import compress_pickle
compress_pickle.dump((
    input_graph
    ), fout_name+'.gz')

print(f'grcs: {len(input_graph.grcs)}')
print(f'mfs: {len(input_graph.mfs)}')
print(f'mf_locs: {len(input_graph.mf_locs)}')


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






