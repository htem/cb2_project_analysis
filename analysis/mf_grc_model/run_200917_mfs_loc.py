import collections
from collections import defaultdict
import sys
import json
import random
from jsmin import jsmin
from io import StringIO
import numpy as np
import copy

sys.path.insert(0, '/n/groups/htem/Segmentation/shared-dev/cb2_segmentation/segway.graph.tmn7')
from segway.graph.synapse_graph import SynapseGraph
from segway.graph.plot_adj_mat import plot_adj_mat

sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')
from tools import *

config_f = "config_grc_mf_200917.json"
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
neurons.extend(mfs)
# neurons.extend(grcs)
# neurons.extend(mlis)

# neurons.extend(interneurons)
# neurons.extend(other_ints_all)


'''Get postsyn partners of the ubcs'''
postsyns = set()
for neuron in neurons:
    postsyns |= graph.get_partners(
        neuron, 'postsyn',
        synapse_min_count=2,
        # partner_type='grc',
        # neuron_subtype='axon',
        filter_list=grcs,
        )

print(postsyns)
print(f'Num postsyns: {len(postsyns)}')  # 160


'''Get locations of synapses of each MF'''
mfs_locs = defaultdict(list)
for neuron in neurons:
    res = graph.get_partners(
        neuron, 'postsyn',
        synapse_min_count=2,
        # partner_type='grc',
        # neuron_subtype='axon',
        filter_list=grcs,
        return_synapse_locs=True,
        )
    if len(res):
        for k in res:
            for loc in res[k]:
                loc = (loc[0]*4, loc[1]*4, loc[2]*40)
                mfs_locs[neuron].append(loc)


'''Cluster and extract locations of MF boutons'''
from sklearn.cluster import DBSCAN

mfs_bouton_locs = {}

'''if a bouton location has less than this many synapses then it won't be considered in order to reduce false positives'''
bouton_synapse_threshold = 5

for mf in mfs_locs:
    dbscan = DBSCAN(eps=8000)  # max dist set to 8um
    dbscan.fit(mfs_locs[mf])
    # dbscan.labels_
    loc_by_label = defaultdict(list)
    for loc, label in zip(mfs_locs[mf], dbscan.labels_):
        loc_by_label[label].append(loc)
    mf_bouton_locs = []
    for label in loc_by_label:
        if len(loc_by_label[label]) < bouton_synapse_threshold:
            continue
        sum = [0, 0, 0]
        for loc in loc_by_label[label]:
            sum = [sum[0]+loc[0], sum[1]+loc[1], sum[2]+loc[2]]
        center = [
            int(sum[0]/len(loc_by_label[label])),
            int(sum[1]/len(loc_by_label[label])),
            int(sum[2]/len(loc_by_label[label])),
            ]
        mf_bouton_locs.append(center)
    mfs_bouton_locs[mf] = mf_bouton_locs
    # print(mf_bouton_locs)
    # for loc in mf_bouton_locs:
        # print([int(loc[0]/4), int(loc[1]/4), int(loc[2]/40)])

mfs_bouton_count = defaultdict(list)
for mf in mfs_bouton_locs:
    mfs_bouton_count[len(mfs_bouton_locs[mf])].append(mf)

for count in mfs_bouton_count:
    print(f'{count}: {mfs_bouton_count[count]}')

for loc in mfs_bouton_locs['mf_252']:
    print([int(loc[0]/4), int(loc[1]/4), int(loc[2]/40)])

'''save mfs_bouton_locs'''
import compress_pickle
compress_pickle.dump((
    mfs_bouton_locs
    ), "mfs_bouton_locs_200917.gz")


asdf

'''Get synapse locations'''
groups_by_type = collections.defaultdict(list)
for neuron in neurons:
    res = graph.get_partners(
        neuron, 'postsyn',
        synapse_min_count=2,
        # partner_type='grc',
        # neuron_subtype='axon',
        filter_list=grcs,
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
