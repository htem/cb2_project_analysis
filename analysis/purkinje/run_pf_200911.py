import collections
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
# from segway.graph.plot_adj_mat import plot_adj_mat

sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')
from tools import *
from mesh_tool import *

# config_f = "config_pc_pfs_200911.json"
config_f = "config_pfs_200911.json"
with open(config_f) as js_file:
    minified = jsmin(js_file.read())
    config = json.load(StringIO(minified))

overwrite = False
if len(sys.argv) == 2 and sys.argv[1] == "--overwrite":
    overwrite = True

graph = SynapseGraph(config_f, overwrite=overwrite)
g = graph.g
random.seed(0)


hierarchy_lut_path = '/n/f810/htem/Segmentation/cb2_v4/output.zarr/luts/fragment_segment'
super_lut_pre = 'super_1x2x2_hist_quant_50'

super_segment_graph = segway.dahlia.connected_segment_server.ConnectedSegmentServer(
    hierarchy_lut_path=hierarchy_lut_path,
    super_lut_pre=super_lut_pre,
    voxel_size=(40, 8, 8),
    find_segment_block_size=(4000, 4096, 4096),
    super_block_size=(4000, 8192, 8192),
    fragments_block_size=(400, 2048, 2048),
    super_offset_hack=(2800, 0, 0),
    base_threshold=0.5,
    )

# (pc_vert_by_box, pc_vert_to_neuron) = compress_pickle.load("mesh_db_pc.gz")
# print(pc_vert_to_neuron)

def grow_segments(s):
    return super_segment_graph.find_connected_super_fragments(
                selected_super_fragments=s,
                no_grow_super_fragments=[],
                threshold=.5,
                z_only=False,
                )

total_orphans = []
# total_orphans_debug = []
# pf_24
for n in g.nodes:
    orphans = []
    for s in graph.orphaned_post_segments[n]:
        if not graph.neuron_db.find_neuron_with_segment_id(s):
            # only add small orphans
            # check if s can be grown
            cc = grow_segments([s])
            if len(cc) == 1:
                orphans.append(cc)
                # total_orphans_debug.append(s)
            elif len(cc) == 2:
                # try again
                cc = grow_segments(cc)
                if len(cc) == 2:
                    orphans.append(cc)
                    # total_orphans_debug.append(s)
            # orphans.append(s)
    if len(orphans):
        print(f'Neuron {n} has {len(orphans)} orphan segments: {orphans}')
        total_orphans.extend(orphans)

# print(total_orphans_debug)

# for each orphaned segment, extract vertices and find the closest pc
print("Loading mesh_db_pc.gz...")
start = time.time()
(pc_vert_by_box, pc_vert_to_neuron) = compress_pickle.load("mesh_db_pc.gz")
# print(f'pc_vert_by_box: {pc_vert_by_box.keys()}')
print(f"Took {time.time() - start}")

orphans_by_pc = collections.defaultdict(list)

for orphan_segments in total_orphans:
    print(orphan_segments)
    closest_vert, dist, dist_xyz = getClosestVertex(orphan_segments, pc_vert_by_box)
    if closest_vert:
        closest_pc = pc_vert_to_neuron[closest_vert]
        orphans_by_pc[closest_pc].append(
            (orphan_segments, dist, dist_xyz))

# print(orphans_by_pc)
# for pc in orphans_by_pc:
#     print(pc)
#     for s in orphans_by_pc[pc]:
#         print(f'{s[0]}; {s[1]}; {s[2]}')

with open('pc_orphan_segments', 'w') as fout:
    for pc in sorted(orphans_by_pc.keys()):
        fout.write(pc + '\n')
        for entry in orphans_by_pc[pc]:
            for s in entry[0]:
                fout.write(f'{s}; {entry[1]}; {entry[2]}\n')

# print(graph.orphaned_post_segments["pf_24"])
# print(graph.orphaned_pre_segments["pf_24"])

# break

asdf

# syn_dict = get_syn_locs(graph)

interneuron_types = [
    'cc', 'golgi', 'lugaro', 'ubc', 'globular']

cfs = []
grcs = []
mlis = []
mfs = []
other_ints = collections.defaultdict(list)
other_ints_all = []
unclassifieds = []

for n, ct in g.nodes.data('cell_type'):

    if ct in interneuron_types:
        other_ints[ct].append(n)
    elif ct == 'stellate':
        g.nodes[n]['cell_type'] = 'mli'
        mlis.append(n)
    elif ct == 'basket':
        mlis.append(n)
        g.nodes[n]['cell_type'] = 'mli'
    elif ct == 'mf':
        mfs.append(n)
    elif ct == 'cf':
        cfs.append(n)
    elif ct == 'grc':
        grcs.append(n)
    elif ct == 'interneuron':
        tags = g.nodes[n]['tags']
        found = False
        for tag in tags:
            if tag in interneuron_types:
                other_ints[tag].append(n)
                g.nodes[n]['cell_type'] = tag
                found = True
        if not found:
            g.nodes[n]['cell_type'] = 'mli'
            mlis.append(n)

    else:
        print(f'{n} is not classified')
        unclassifieds.append(n)

interneurons = copy.deepcopy(mlis)
for k in other_ints:
    other_ints_all.extend(other_ints[k])
interneurons.extend(other_ints_all)

print(f'Num cfs: {len(cfs)}')  # 160
print(f'Num mfs: {len(mfs)}')  # 160
print(f'Num grcs: {len(grcs)}')  # 97
print(f'Num interneurons: {len(interneurons)}')  # 97
print(f'Num mlis: {len(mlis)}')  # 97
print(f'Num other_ints_all: {len(other_ints_all)}')  # 97
for ct in interneuron_types:
    print(f'{ct}: {len(other_ints[ct])}')  # 97

# neurons = other_ints['ubc']
# neurons = mfs
neurons = copy.deepcopy(mlis)
neurons.extend(cfs)
# neurons.extend(mlis)
neurons.extend(interneurons)


'''Get presyn partners of the ubcs'''
presyns = set()
for neuron in neurons:
    presyns |= graph.get_partners(
        neuron, 'presyn',
        synapse_min_count=2,
        # partner_type='grc',
        # neuron_subtype='axon',
        filter_list=cfs,
        )

print(presyns)
print(f'Num presyns: {len(presyns)}')  # 160


'''Get synapse locations'''
groups_by_type = collections.defaultdict(list)
for neuron in neurons:
    res = graph.get_partners(
        neuron, 'presyn',
        synapse_min_count=2,
        # partner_type='grc',
        # neuron_subtype='axon',
        filter_list=cfs,
        return_synapse_locs=True,
        )
    if len(res):
        ct = g.nodes[neuron]['cell_type']
        # print(res)
        # ct = g.nodes[res[0][1]]['cell_type']
        groups_by_type[ct].append(res)

for ct in groups_by_type:
    print(f'For neuron type {ct}')
    for res in groups_by_type[ct]:
        for i in res:
            print(f'{i}: {res[i]}')
