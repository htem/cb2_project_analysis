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

asdf

'''Save graph'''
import compress_pickle
compress_pickle.dump((
    input_graph
    ), "input_graph_200923.gz")


input_graph.randomize_graph()


asdf

'''Count num claws per grc'''
count = defaultdict(int)
for grc_id in input_graph.grcs:
    grc = input_graph.grcs[grc_id]
    count[grc.num_claws] += 1

for k in sorted(count.keys()):
    print(f'{k},{count[k]}')


'''Histogram of claw lengths'''
count = defaultdict(int)
for grc_id in input_graph.grcs:
    grc = input_graph.grcs[grc_id]
    for e in grc.edges:
        d = get_eucledean_dist(grc.soma_loc, e[1])
        d = int(d/1000)
        count[d] += 1

for k in sorted(count.keys()):
    print(f'{k},{count[k]}')

'''claws per mf'''

count = defaultdict(int)
for mf_id in input_graph.mfs:
    mf = input_graph.mfs[mf_id]
    count[len(mf.claws)] += 1

# for k in sorted(count.keys()):
for k in range(max(count.keys())+1):
    print(f'{k},{count[k]}')

'''claws per bouton'''

bouton_claw_count = defaultdict(int)
for mf_id in input_graph.mfs:
    mf = input_graph.mfs[mf_id]
    for c in mf.claws:
        bouton_claw_count[c[1]] += 1

count = defaultdict(int)
for b in bouton_claw_count:
    count[bouton_claw_count[b]] += 1

bucket = 2
for k in range(1, max(count.keys()), bucket):
    s = 0
    for i in range(k, k+bucket):
        s += count[i]
    print(f'{k},{s}')

'''print locations of boutons'''

for b in sorted(bouton_claw_count, key=lambda b: bouton_claw_count[b]):
    e = bouton_claw_count[b]
    print(f'{to_ng_coord(b)}: {e}')

# 56 grcs
print(input_graph.mfs['mf_29'].claws)
# verified: all TPs


'''MFs output to the same grc through multiple boutons'''

count = defaultdict(int)
for mf_id in input_graph.mfs:
    mf = input_graph.mfs[mf_id]
    if len(mf.locs) <= 1:
        continue
    grc_count = defaultdict(int)
    for c in mf.claws:
        grc_count[c[0]] += 1
    for grc in grc_count:
        if grc_count[grc] >= 2:
            count[grc_count[grc]] += 1
            print(f'{mf_id}, {grc}')
# for k in sorted(count.keys()):
#     print(f'{k},{count[k]}')
'''grcs receiving multiple boutons of same mf'''
'''same as above'''
count = defaultdict(int)
for grc_id in input_graph.grcs:
    mf_count = defaultdict(int)
    for c in input_graph.grcs[grc_id].edges:
        mf_count[c[0]] += 1
    for mf in mf_count:
        if mf_count[mf] >= 2:
            count[mf_count[mf]] += 1
            print(f'{grc_id}, {mf}')
# for k in sorted(count.keys()):
#     print(f'{k}, {count[k]}')


''' count MF doubles'''

count = defaultdict(lambda: defaultdict(list))
mfs = [k for k in input_graph.mfs.keys()]
for i in range(len(mfs)):
    i_set = set(input_graph.mfs[mfs[i]].grcs)
    for j in range(len(mfs)):
        if i == j:
            continue
        j_set = set(input_graph.mfs[mfs[j]].grcs)
        common_grcs = i_set & j_set
        count[mfs[i]][len(common_grcs)].append(mfs[j])

'''doubles'''
histogram = defaultdict(int)
doubles_count = []
threshold = 2
for mf in count:
    sum = 0
    for val in count[mf]:
        if val >= threshold:
            sum += len(count[mf][val])
    doubles_count.append(sum)
    histogram[sum] += 1

for k in range(max(histogram.keys())+1):
    print(f'{k}, {histogram[k]}')



'''triples'''
histogram = defaultdict(int)
doubles_count = []
threshold = 3
for mf in count:
    sum = 0
    for val in count[mf]:
        if val >= threshold:
            sum += len(count[mf][val])
    doubles_count.append(sum)
    histogram[sum] += 1

for k in range(max(histogram.keys())+1):
    print(f'{k}, {histogram[k]}')

'''quads'''
histogram = defaultdict(int)
doubles_count = []
threshold = 4
for mf in count:
    sum = 0
    for val in count[mf]:
        if val >= threshold:
            sum += len(count[mf][val])
    doubles_count.append(sum)
    histogram[sum] += 1

# doubles_count = sorted(doubles_count)
# print(doubles_count)
for k in sorted(histogram.keys()):
    print(f'{k}, {histogram[k]}')


'''quads'''
doubles_count = []
threshold = 6
for mf in count:
    sum = 0
    for val in count[mf]:
        if val >= threshold:
            print(f'{mf}: {count[mf][val]}')
            sum += len(count[mf][val])
    doubles_count.append(sum)

doubles_count = sorted(doubles_count)
print(doubles_count)

# debug
input_graph.mfs['mf_76'].grcs & input_graph.mfs['mf_29'].grcs
input_graph.mfs['mf_99'].grcs & input_graph.mfs['mf_29'].grcs



'''grcs receiving multiple boutons of same mf'''
'''FIGURE 201011'''

count = defaultdict(lambda: defaultdict(list))
grcs = [k for k in input_graph.grcs.keys()]
for i in range(len(grcs)):
    i_set = set(input_graph.grcs[grcs[i]].mfs)
    for j in range(len(grcs)):
        if i == j:
            continue
        j_set = set(input_graph.grcs[grcs[j]].mfs)
        common_mfs = i_set & j_set
        if len(common_mfs) > 1:
            count[grcs[i]][len(common_mfs)].append(grcs[j])

'''doubles'''
histogram = defaultdict(int)
doubles_count = []
threshold = 2
for mf in count:
    sum = 0
    for val in count[mf]:
        if val >= threshold:
            sum += len(count[mf][val])
    doubles_count.append(sum)
    histogram[sum] += 1

for k in range(max(histogram.keys())+1):
    print(f'{k}, {histogram[k]}')


'''triples'''
histogram = defaultdict(int)
doubles_count = []
threshold = 3
for mf in count:
    sum = 0
    for val in count[mf]:
        if val >= threshold:
            sum += len(count[mf][val])
    doubles_count.append(sum)
    histogram[sum] += 1

for k in range(max(histogram.keys())+1):
    print(f'{k}, {histogram[k]}')


'''quads debug'''
histogram = defaultdict(int)
doubles_count = []
threshold = 3
for mf in count:
    sum = 0
    for val in count[mf]:
        if val >= threshold:
            sum += len(count[mf][val])
            print(f'{mf}: {count[mf][val]}')
    doubles_count.append(sum)
    histogram[sum] += 1

asdf


'''check for 2-share grcs'''


count = defaultdict(lambda: defaultdict(list))
grcs = ["grc_1009", "grc_1038", "grc_1064", "grc_1109", "grc_1110", "grc_1135", "grc_118", "grc_1242", "grc_341", "grc_349", "grc_372", "grc_468", "grc_581", "grc_646", "grc_713", "grc_716", "grc_776", "grc_1125", "grc_1122", "grc_1361", "grc_321", "grc_581", "grc_349", "grc_388", "grc_639", "grc_646", "grc_349", "grc_1153", "grc_1135", "grc_1119", "grc_1338",]
grcs = list(set(grcs))
listed = set()

for i in range(len(grcs)):
    i_set = set(input_graph.grcs[grcs[i]].mfs)
    for j in range(len(grcs)):
        if i == j:
            continue
        j_set = set(input_graph.grcs[grcs[j]].mfs)
        common_mfs = i_set & j_set
        if len(common_mfs) == 1:
            # count[grcs[i]][len(common_mfs)].append(grcs[j])
            if (i, j) not in listed:
                print(f'{grcs[i]}, {grcs[j]}')
                listed.add((i, j))
                listed.add((j, i))



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
        # partner_type='mf',
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
