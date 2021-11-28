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
from segway.graph.plot_adj_mat import plot_adj_mat

sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')
from tools import *
import tools_mf_graph

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

config_f = "config_grc_mf_201004.json"
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
print(f'Num other_ints_all: {len(other_ints_all)}')  # 97
for ct in interneuron_types:
    print(f'{ct}: {len(other_ints[ct])}')  # 97

# neurons = other_ints['ubc']
# neurons = mfs
neurons = []
# neurons.extend(cfs)
# neurons.extend(mfs)
neurons.extend(grcs)
# neurons.extend(mlis)

# neurons.extend(interneurons)
# neurons.extend(other_ints_all)

'''Get presyn partners of the ubcs'''
# presyns = set()
# for neuron in neurons:
#     presyns |= graph.get_partners(
#         neuron, 'presyn',
#         synapse_min_count=2,
#         # partner_type='grc',
#         # neuron_subtype='axon',
#         filter_list=mfs,
#         )

# print(presyns)
# print(f'Num presyns: {len(presyns)}')  # 160


'''Get locations of synapses of each MF'''
mfs_locs = defaultdict(list)
for neuron in neurons:
    if 'ml' in neuron or 'tmn7_ml' in g.nodes[neuron]['tags']:
        print(f"Skipped {neuron}")
        continue
    xyz = get_node_pos(g, neuron)
    if xyz[0] < 90000*4 or xyz[0] > 130000*4:
        print(f"Skipped {neuron} outside X (xyz={xyz})")
        continue
    if xyz[2] < 15000 or xyz[2] > 35000:
        print(f"Skipped {neuron} outside X (xyz={xyz})")
        continue
    res = graph.get_partners(
        neuron, 'presyn',
        synapse_min_count=2,
        # partner_type='grc',
        # neuron_subtype='axon',
        filter_list=mfs,
        return_synapse_locs=True,
        )
    if len(res):
        for pair in res:
            for loc in res[pair]:
                loc = (loc[0]*4, loc[1]*4, loc[2]*40)
                mfs_locs[neuron].append((pair[0], loc))


'''load mfs_bouton_locs'''
import compress_pickle
mfs_bouton_locs = compress_pickle.load("mfs_bouton_locs_200917.gz")

input_graph = tools_mf_graph.GCLGraph()
input_graph.add_mfs(mfs_bouton_locs)

'''Make GT graph'''
for grc_id in mfs_locs:
    xyz = get_node_pos(g, grc_id)
    input_graph.add_grc(grc_id, xyz, mfs_locs[grc_id])

'''Save graph'''
import compress_pickle
compress_pickle.dump((
    input_graph
    ), "input_graph_201011_restricted_z.gz")

