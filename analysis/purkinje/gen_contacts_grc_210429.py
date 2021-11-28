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
from mesh_tool import getBoxId, getClosestVertexPyramid

# get proofread'ed fragments and ignore them in this pass
grcs_with_complete_axons_f = '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/completed_neurons_with_axons_210421'
grcs_with_complete_axons = set()
with open(grcs_with_complete_axons_f) as fin:
    for line in fin:
        try:
            id = line.strip()
            grcs_with_complete_axons.add(id)
        except:
            pass
grcs_with_complete_axons.remove('')

config_f = '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/config_grc_210429_setup01.json'

graph = SynapseGraph(config_f,
    # overwrite=True,
    )
g = graph.g

print("Getting all grcs segments...")
neuron_box_segs = defaultdict(lambda: defaultdict(list))
for grc_id in grcs_with_complete_axons:
    grc = graph.neuron_db.get_neuron(grc_id, with_children=True)
    for sid in grc.segments:
        if sid == 0:
            continue
        neuron_box_segs[grc_id][getBoxId(sid)].append(sid)

neuron_box_segs_dict = dict()
for grc_id in neuron_box_segs:
    neuron_box_segs_dict[grc_id] = dict(neuron_box_segs[grc_id])

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

touches = defaultdict(dict)
pc_list = graph.neuron_db.find_neuron({'cell_type': 'pc'})
for pc_id in pc_list:
# for pc_id in ['pc_0']:
    fname = f'mesh_db_dendrites_210429/mesh_db_pc.{pc_id}.gz'
    print(f"Loading {fname}...")
    start = time.time()
    try:
        (local_pc_vert_by_box, local_pc_vert_to_neuron) = compress_pickle.load(fname)
    except:
        continue
    print(f"Took {time.time() - start}")
    print("Running touch algorithm...")
    min_distance = defaultdict(lambda: sys.maxsize)
    for boxid in local_pc_vert_by_box:
        print('.', end='', flush=True)
        pc_vertices_pyramid_cache = {}
        pc_vertices_pyramid_reverse_cache = defaultdict(set)
        pc_vertices = set(local_pc_vert_by_box[boxid])
        for grc_id in neuron_box_segs_dict:
            if boxid in neuron_box_segs_dict[grc_id]:
                segs = neuron_box_segs_dict[grc_id][boxid]
                closest_vert, dist = getClosestVertexPyramid(
                    segs, pc_vertices,
                    pc_vertices_pyramid_cache, pc_vertices_pyramid_reverse_cache,
                    ds_factors, mesh_voxel_size)
                if closest_vert:
                    if dist < min_distance[grc_id]:
                        min_distance[grc_id] = dist
                        touches[grc_id][pc_id] = (dist, to_ng_coord(closest_vert))
    print()


for grc_id in touches:
    print(f'{grc_id}: {touches[grc_id]}')


'''write to files for debugging'''
with open('grc_pc_touches_210429', 'w') as fout:
    for grc_id in sorted(touches.keys()):
        fout.write(grc_id + '\n')
        for entry in sorted(touches[grc_id].keys()):
            fout.write(f'   {entry}: {touches[grc_id][entry]}\n')

'''pickle'''
fname = 'grc_pc_touches_210429.gz'
print(f"Writing to {fname}...")
compress_pickle.dump(touches, fname)
