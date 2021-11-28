import collections
from collections import defaultdict
import sys
import json
import random
from jsmin import jsmin
from io import StringIO
import numpy as np
import copy
import compress_pickle
import time

import daisy
daisy.block.Block.BLOCK_ID_ADD_ONE_FIX = True


sys.path.insert(0, '/n/groups/htem/Segmentation/shared-dev/cb2_segmentation/segway.graph.tmn7')
sys.path.insert(0, '/n/groups/htem/Segmentation/tmn7/segway.dahlia')

import segway.dahlia.connected_segment_server
from segway.graph.synapse_graph import SynapseGraph

sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')
from tools import to_ng_coord
from mesh_tool import getBoxId, getClosestVertexPyramidFromPoint


# Load pf-PC synapse locs and get synlocs per PC
syn_locs = defaultdict(list)
gzdb = '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/gen_db/pfs/' \
       'gen_210429_setup01_syndb_threshold_10_coalesced.gz'
db = compress_pickle.load(gzdb)
pc_list = set()
for pf in db:
    for pc in db[pf]:
        pc_list.add(pc)
        for syn in db[pf][pc]:
            # print(syn)
            # raise Error()
            syn_locs[pc].append(syn['syn_loc0'])

pfs_list = db.keys()

ds_factors = [
    (64, 64, 64),  # 2560
    (32, 32, 32),  # 1280
    (16, 16, 16),  # 640
    (8, 8, 8),  # 320
    (4, 4, 4),  # 160
    (2, 2, 2),  # 80
    (1, 1, 1),  # 40
    ]
mesh_voxel_size = (32, 32, 40)
# print(ds_factors)

fname = f'mesh_db_210708/db.gz'
print(f"Loading {fname}...")
db = compress_pickle.load(fname)

import importlib
import mesh_tool
importlib.reload(mesh_tool)
res = defaultdict(dict)
# pfs_list = graph.neuron_db.find_neuron({'cell_type': 'pf'})
for pf in pfs_list:
    try:
        (vert_by_box, vert_to_neuron) = db[pf]
    except:
        continue
    print(f"Running touch algorithm for {pf}...")
    min_distance = defaultdict(lambda: sys.maxsize)
    vertices_pyramid_cache = {}
    vertices_pyramid_reverse_cache = defaultdict(set)
    vertices = []
    for boxid in vert_by_box:
        # print('.', end='', flush=True)
        vertices.extend(vert_by_box[boxid])
    vertices = set(vertices)
    for pc in pc_list:
        # for syn_loc in syn_locs[pc]:
        for syn_loc in syn_locs[pc]:
            # syn_loc = (88552*4, 64644*4, 248*40)
            closest_vert, dist = mesh_tool.getClosestVertexPyramidFromPoint(
                syn_loc, vertices,
                vertices_pyramid_cache, vertices_pyramid_reverse_cache,
                ds_factors, mesh_voxel_size)
            if closest_vert:
                if dist < min_distance[syn_loc]:
                    min_distance[syn_loc] = dist
                    res[to_ng_coord(syn_loc)][pf] = (dist, to_ng_coord(closest_vert))
    # print(res); asdf
    # break

# for pf in sorted(res.keys()):
#     print(pf)
#     print(res[pf])


'''write to files for debugging'''
with open('syn_pf_distance_db_210707', 'w') as fout:
    for syn_loc in sorted(res.keys()):
        fout.write(str(syn_loc) + '\n')
        for entry in sorted(res[syn_loc].keys()):
            fout.write(f'   {entry}: {res[syn_loc][entry]}\n')

'''pickle'''
fname = 'syn_pf_distance_db_210707.gz'
print(f"Writing to {fname}...")
compress_pickle.dump(res, fname)
