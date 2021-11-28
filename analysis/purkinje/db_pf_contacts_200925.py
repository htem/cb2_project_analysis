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
import sys

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

def to_ng_coord(coord):
    return (
        int(coord[0]/4),
        int(coord[1]/4),
        int(coord[2]/40),
        )

def reload():
    importlib.reload(tools_mf_graph)
    # from tools_mf_graph import GrC
    # from tools_mf_graph import MF
    # from tools_mf_graph import GCLGraph

config_f = "config_pfs_200925.json"
with open(config_f) as js_file:
    minified = jsmin(js_file.read())
    config = json.load(StringIO(minified))

overwrite = False
if len(sys.argv) == 2 and sys.argv[1] == "--overwrite":
    overwrite = True

graph = SynapseGraph(config_f, overwrite=overwrite)
g = graph.g
random.seed(0)
# syn_dict = get_syn_locs(graph)

interneuron_types = [
    'cc', 'golgi', 'lugaro', 'ubc', 'globular', 'glia']

cfs = []
glias = []
mlis = []
mfs = []
grcs = []
pfs = []
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
    elif ct == 'glia':
        glias.append(n)
    elif ct == 'pf':
        pfs.append(n)
    elif ct == 'interneuron':
        tags = g.nodes[n]['tags']
        found = False
        for tag in tags:
            if tag in interneuron_types:
                if tag != "glia":
                    other_ints[tag].append(n)
                    g.nodes[n]['cell_type'] = tag
                    found = True
                else:
                    glias.append(n)
        if not found:
            g.nodes[n]['cell_type'] = 'mli'
            mlis.append(n)

    else:
        print(f'{n} is not classified')
        unclassifieds.append(n)

for k in other_ints:
    other_ints_all.extend(other_ints[k])

interneurons = copy.deepcopy(mlis)
interneurons.extend(other_ints_all)

print(f'Num cfs: {len(cfs)}')  # 160
print(f'Num mfs: {len(mfs)}')  # 160
print(f'Num grcs: {len(grcs)}')  # 97
print(f'Num interneurons: {len(interneurons)}')  # 97
print(f'Num mlis: {len(mlis)}')  # 97
print(f'Num pfs: {len(pfs)}')  # 97
print(f'Num other_ints_all: {len(other_ints_all)}')  # 97
for ct in interneuron_types:
    print(f'{ct}: {len(other_ints[ct])}')  # 97

# neurons = other_ints['ubc']
# neurons = mfs
neurons = []
# neurons.extend(cfs)
# neurons.extend(mfs)
# neurons.extend(grcs)
# neurons.extend(mlis)
neurons.extend(pfs)

# neurons.extend(interneurons)
# neurons.extend(other_ints_all)

pfs = neurons

# grcs = g.nodes.keys()
# neuron:box:segs
print("Getting all pfs segments...")
pf_box_segs = defaultdict(lambda: defaultdict(list))
for pf_id in pfs:
    pf = graph.neuron_db.get_neuron(pf_id, with_children=True)
    # print(pf.segments)
    # pf_mesh_ids = 
    for sid in pf.segments:
        if sid == 0:
            continue
        pf_box_segs[pf_id][getBoxId(sid)].append(sid)


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
# for i in range(len(ds_factors)):
#     ds_factors[i] = tuple([k*d for k, d in zip(ds_factors[i], mesh_voxel_size)])
print(ds_factors)


# min_distance = defaultdict(lambda: sys.maxsize)
# min_distance = defaultdict(lambda: defaultdict(lambda: sys.maxsize))
touches = defaultdict(dict)
pc_list = graph.neuron_db.find_neuron_filtered({'cell_type': 'pc'})
for pc_id in pc_list:
    fname = f'mesh_db_dendrites_200921/mesh_db_pc.{pc_id}.gz'
    print(f"Loading {fname}...")
    start = time.time()
    (local_pc_vert_by_box, local_pc_vert_to_neuron) = compress_pickle.load(fname)
    print(f"Took {time.time() - start}")
    print("Running touch algorithm...")
    min_distance = defaultdict(lambda: sys.maxsize)
    for boxid in local_pc_vert_by_box:
        print('.', end='', flush=True)
        pc_vertices_pyramid_cache = {}
        pc_vertices_pyramid_reverse_cache = defaultdict(set)
        pc_vertices = set(local_pc_vert_by_box[boxid])
        for pf_id in pf_box_segs:
            segs = pf_box_segs[pf_id][boxid]
            closest_vert, dist = getClosestVertexPyramid(
                segs, pc_vertices,
                pc_vertices_pyramid_cache, pc_vertices_pyramid_reverse_cache,
                ds_factors, mesh_voxel_size)
            if closest_vert:
                if dist < min_distance[pf_id]:
                    min_distance[pf_id] = dist
                    touches[pf_id][pc_id] = (dist, to_ng_coord(closest_vert))
    print()


# for pf_id in touches:
#     print(f'{pf_id}: {touches[pf_id]}')


'''write to files for debugging'''
with open('db_pf_contacts_200925', 'w') as fout:
    for pf_id in sorted(touches.keys()):
        fout.write(pf_id + '\n')
        for entry in sorted(touches[pf_id].keys()):
            fout.write(f'   {entry}: {touches[pf_id][entry]}\n')

'''pickle'''
fname = 'db_pf_contacts_200925.gz'
print(f"Writing to {fname}...")
compress_pickle.dump(touches, fname)



asdf

'''write to files'''
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
