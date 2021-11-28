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
from functools import partial


import scipy
import matplotlib.pyplot as plt
import pandas as pd

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

config_f = "../config_pf_200925.json"
with open(config_f) as js_file:
    minified = jsmin(js_file.read())
    config = json.load(StringIO(minified))

overwrite = False
if len(sys.argv) == 2 and sys.argv[1] == "--overwrite":
    overwrite = True

syn_score_threshold = 100

graph = SynapseGraph(config_f, overwrite=overwrite,
    db_name='cb2_v4_synapse_pred_setup22_synapsedb_0p6_threshold_5',
    # syn_score_threshold=100)
    syn_score_threshold=syn_score_threshold)

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

# '''Load grc-pc touch dictionary'''
# import compress_pickle
# print("Loading pf-pc touch dictionary...")
# fname = '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/purkinje/db_pf_contacts_200925.gz'
# touches_all = compress_pickle.load(fname)

# max_touch_dist = 100
# max_touch_dist = 0
# max_touch_dist = 60
# touches = defaultdict(set)
# touches_pc_grc = defaultdict(set)

# for pf_id in touches_all:
#     for pc_id in touches_all[pf_id]:
#         if touches_all[pf_id][pc_id][0] <= max_touch_dist:
#             touches[pf_id].add(pc_id)
#             touches_pc_grc[pc_id].add(pf_id)


'''Computing actual synapse grc-pc postsyns'''
print("Loading pf-pc synapses...")
pf_postsyn_pcs = defaultdict(set)
pf_postsyn_pcs_locs = defaultdict(lambda: defaultdict(list))
pf_postsyn_pcs_count = defaultdict(lambda: defaultdict(int))
pf_synapse_count = defaultdict(int)

pc_presyns = defaultdict(set)
pc_presyns_count = defaultdict(lambda: defaultdict(int))
pc_synapse_count = defaultdict(int)

all_pcs = set()

for pf in pfs:
    syns = graph.get_synapses(pf)
    for syn in syns:
        seg_id = syn['sf_post']
        # loc = syn['syn_loc']
        loc = syn['post_loc']
        postsyn_pc = graph.neuron_db.find_neuron_with_segment_id(seg_id)
        pf_synapse_count[pf] += 1
        if postsyn_pc is not None and \
                ('pc' in postsyn_pc or 'purkinje' in postsyn_pc):
            if 'pcl' in postsyn_pc:
                continue
            postsyn_pc = postsyn_pc.split('.')[0]
            pf_postsyn_pcs[pf].add(postsyn_pc)
            pf_postsyn_pcs_count[pf][postsyn_pc] += 1
            pf_postsyn_pcs_locs[pf][postsyn_pc].append(loc)
            pc_presyns[postsyn_pc].add(pf)
            pc_presyns_count[postsyn_pc][pf] += 1
            pc_synapse_count[postsyn_pc] += 1
            all_pcs.add(postsyn_pc)


# for n in pfs:
#     for s in graph.orphaned_post_segments[n]:
#         postsyn = graph.neuron_db.find_neuron_with_segment_id(s)
#         pf_synapse_count[n] += 1
#         if postsyn is not None and \
#                 ('pc' in postsyn or 'purkinje' in postsyn):
#             if 'pcl' in postsyn:
#                 continue
#             postsyn = postsyn.split('.')[0]
#             pf_postsyn_pcs[n].add(postsyn)
#             pf_postsyn_pcs_count[n][postsyn] += 1
#             pc_presyns[postsyn].add(n)
#             pc_presyns_count[postsyn][n] += 1
#             pc_synapse_count[postsyn] += 1
#             all_pcs.add(postsyn)

# all_pcs = [k for k in all_pcs]

# print 3+ syn connections
# for pf in pf_postsyn_pcs:
#     pcs = sorted([k for k in pf_postsyn_pcs[pf]])
#     # print(f'{pf}: {pcs}')
#     for pc in pcs:
#         locs = pf_postsyn_pcs_locs[pf][pc]
#         if len(locs) >= 3:
#             print(f'{pf} to {pc}: {locs}')

def to_voxel(coord, voxel_size):
    return (int(coord[0] / voxel_size[0]),
            int(coord[1] / voxel_size[1]),
            int(coord[2] / voxel_size[2]))

def get_eucledean_dist(a, b):
    return np.linalg.norm(
        (a[0]-b[0], a[1]-b[1], a[2]-b[2]))

def get_near_locs(locs, threshold=500):
    nears = []
    locs = sorted(locs)
    for i, loc0 in enumerate(locs):
        for j, loc1 in enumerate(locs):
            # if loc0 is loc1:
                # continue
            if j <= i:
                continue
            d = get_eucledean_dist(loc0, loc1)
            if d <= threshold:
                nears.append((loc0, loc1))
    return nears

threshold = 400
dist_type = 'std'

# nearlist = defaultdict(list)
nearlist = {}

for pf in pf_postsyn_pcs:
    pcs = sorted([k for k in pf_postsyn_pcs[pf]])
    # print(f'{pf}: {pcs}')
    for pc in pcs:
        locs = pf_postsyn_pcs_locs[pf][pc]
        nears = get_near_locs(locs, threshold=threshold)
        # nears = [to_voxel(k, graph.voxel_size_xyz) for k in nears]
        nears = [tuple(k) for k in nears]
        if len(nears):
            nearlist[(pf, pc)] = nears
            print(f'{pf}: {pc}')
            # print(nears)
            for pair in nears:
                pair = [to_voxel(k, graph.voxel_size_xyz) for k in pair]
                print(pair)

import compress_pickle
compress_pickle.dump((
    nearlist
    ), f"nearlist_{threshold}_{dist_type}.gz")



# print likely split syns



asdf


def print_box_plot(
        dataset, mult=None,
        percentile0=10,
        percentile1=25,
        ):
    n_samples = len(dataset)
    dataset = sorted(dataset)
    data0 = int(n_samples/100*percentile0)
    data1 = int(n_samples/100*percentile1)
    data2 = int(n_samples/100*(100-percentile1))
    data3 = int(n_samples/100*(100-percentile0))
    data0 = dataset[data0]
    data1 = dataset[data1]
    data2 = dataset[data2]
    data3 = dataset[data3]
    sum = 0
    for n in dataset:
        sum += n
    average = sum/n_samples
    if mult:
        average *= mult
        data0 *= mult
        data1 *= mult
        data2 *= mult
        data3 *= mult
    print(f'{data0}, {data1}, {data2}, {data3}, {average}')




'''Multi-synapses studies
- plot spread of multisyns per pf
- plot spread per PC
'''

'''All stats'''
all_syn_count_histogram = defaultdict(int)
total_syn_count = 0
for pf in pfs:
    for pc in pf_postsyn_pcs_count[pf]:
        count = pf_postsyn_pcs_count[pf][pc]
        all_syn_count_histogram[count] += 1
        total_syn_count += 1

for val in range(max(all_syn_count_histogram)+1):
    print(f'{val}, {all_syn_count_histogram[val]}')

'''PC spread'''
pc_syns_histogram_list = defaultdict(list)
all_counts = []
min_pc_contacts = 100  # should also try 100/50/400/1000?
max_pc_contacts = None
for pc in all_pcs:
    total_cnx = len(pc_presyns[pc])
    if total_cnx < min_pc_contacts:
        continue
    if max_pc_contacts and total_cnx > max_pc_contacts:
        continue
    local_counts = defaultdict(int)
    num_connections = 0
    for pf in pc_presyns_count[pc]:
        local_counts[pc_presyns_count[pc][pf]] += 1
        num_connections += 1
    for i in range(5):
        pc_syns_histogram_list[i].append(float(local_counts[i])/total_cnx)

for i in range(5):
    print_box_plot(pc_syns_histogram_list[i])


'''pfs spread'''
pf_syn_count_histogram_list = defaultdict(list)
min_pc_contacts = 4  # should also try 100/50/400/1000?
max_pc_contacts = None
n_samples = 0
for pf in pfs:
    total_cnx = len(pf_postsyn_pcs[pf])
    if min_pc_contacts and total_cnx < min_pc_contacts:
        continue
    if max_pc_contacts and total_cnx > max_pc_contacts:
        continue
    local_counts = defaultdict(int)
    num_connections = 0
    n_samples += 1
    for pc in pf_postsyn_pcs_count[pf]:
        local_counts[pf_postsyn_pcs_count[pf][pc]] += 1
        num_connections += 1
    for i in range(5):
        pf_syn_count_histogram_list[i].append(float(local_counts[i])/total_cnx)

print(f'Num samples = {n_samples}')
for i in range(5):
    print_box_plot(pf_syn_count_histogram_list[i])


# write plot data to disk

import compress_pickle
compress_pickle.dump((
    total_syn_count,
    all_syn_count_histogram,
    pc_syns_histogram_list,
    pf_syn_count_histogram_list,
    ), "multi_syn_ratio_201107_data.gz")


asdf

