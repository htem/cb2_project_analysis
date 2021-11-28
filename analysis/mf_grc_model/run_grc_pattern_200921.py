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


'''
Get postsyn of grc axons; but only filtering for PCs
'''

# grcs = ["grc_1009", "grc_1038", "grc_1064", "grc_1109", "grc_1110", "grc_1135", "grc_118", "grc_1242", "grc_341", "grc_349", "grc_372", "grc_468", "grc_581", "grc_646", "grc_713", "grc_716", "grc_776", "grc_1125", "grc_1122", "grc_1361", "grc_321", "grc_581", "grc_349", "grc_388", "grc_639", "grc_646", "grc_349", "grc_1153", "grc_1135", "grc_1119", "grc_1338",]

# with open('input_graph_200923.gz') as f:
#     for line in f:

grcs = set()
with open('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/purkinje/proofread_grc_axons_200923') as f:
    for line in f:
        grcs.add(str(line.strip()))

grcs.remove('')
# grcs = [k for k in grcs]
print(grcs)

'''Load grc-pc touch dictionary'''
import compress_pickle
print("Loading grc-pc touch dictionary...")
fname = '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/purkinje/grc_pc_touches_200923.gz'
touches_all = compress_pickle.load(fname)

max_touch_dist = 0
touches = defaultdict(set)

for grc_id in touches_all:
    for pc_id in touches_all[grc_id]:
        if touches_all[grc_id][pc_id][0] <= max_touch_dist:
            touches[grc_id].add(pc_id)


'''Computing actual synapse grc-pc postsyns'''
grc_postsyn_pcs = defaultdict(set)
for n in grcs:
    for s in graph.orphaned_post_segments[n]:
        postsyn = graph.neuron_db.find_neuron_with_segment_id(s)
        if postsyn is not None and \
                ('pc' in postsyn or 'purkinje' in postsyn):
            if 'pcl' in postsyn:
                continue
            postsyn = postsyn.split('.')[0]
            grc_postsyn_pcs[n].add(postsyn)

for grc in grc_postsyn_pcs:
    pcs = sorted([k for k in grc_postsyn_pcs[grc]])
    print(f'{grc}: {pcs}')


'''Load mf-grc graph'''

print("Loading mf-grc graph...")
fname = '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_200919.gz'
input_graph = compress_pickle.load(fname)

to_remove = []
for grc_id in grcs:
    if grc_id not in input_graph.grcs:
        print(f"Skipping {grc_id} not in input_graph")
        to_remove.append(grc_id)

for g in to_remove:
    grcs.remove(g)

grcs = [k for k in grcs]


'''Analyze 3-share grcs'''

def analyze_pc_pattern(a, b,
        no_touches=False,
        no_common_zeros=False,
        ):
    all_a = set(touches[a])
    ones_a = set(grc_postsyn_pcs[a])
    all_a |= ones_a
    all_b = set(touches[b])
    ones_b = set(grc_postsyn_pcs[b])
    all_b |= ones_b
    # if no_touches:
    #     all_a = set(ones_a)
    #     all_b = set(ones_b)
    common_both = all_a & all_b
    if no_touches:
        common_both = ones_a | ones_b
        # all_a = set(ones_a)
        # all_b = set(ones_b)
    if len(common_both) == 0:
        return None
    pattern_a = ''
    pattern_b = ''
    common_both = [k for k in common_both]
    similarity = 0
    for pc_id in common_both:
        if pc_id in ones_a:
            pattern_a += '1'
        else:
            pattern_a += '0'
        if pc_id in ones_b:
            pattern_b += '1'
        else:
            pattern_b += '0'
        if pc_id in ones_a and pc_id in ones_b:
            similarity += 1
        if pc_id not in ones_a and pc_id not in ones_b:
            if not no_common_zeros:
                similarity += 1
    similarity = float(similarity) / len(common_both)
    summary = (similarity, common_both, pattern_a, pattern_b)
    return summary


grc_common_mfs_count = defaultdict(lambda: defaultdict(list))
for i in range(len(grcs)):
    i_set = set(input_graph.grcs[grcs[i]].mfs)
    for j in range(len(grcs)):
        if i == j:
            continue
        j_set = set(input_graph.grcs[grcs[j]].mfs)
        common_mfs = i_set & j_set
        grc_common_mfs_count[grcs[i]][len(common_mfs)].append(grcs[j])

'''2-share'''

samples = 1000
share_count = 0
min_pattern_len = 10
counted = set()
scores = {}
histogram = defaultdict(int)
for a in grc_common_mfs_count:
    if share_count in grc_common_mfs_count[a]:
        for b in grc_common_mfs_count[a][share_count]:
            if (a, b) not in counted:
                counted.add((a, b))
                counted.add((b, a))
                summary = analyze_pc_pattern(a, b)
                if summary is not None:
                    if len(summary[1]) >= min_pattern_len:
                        scores[(a, b)] = summary
                        score = int((summary[0]*min_pattern_len)+0.5)
                        histogram[score] += 1

for val in range(min_pattern_len+1):
    print(f'{val}, {histogram[val]}')

'''2-share but with normalized sampling'''

for share_count in [0, 1, 2, 3, 4]:
    counted = set()
    pairs = []
    pairs_min4 = []
    for a in grc_common_mfs_count:
        if share_count in grc_common_mfs_count[a]:
            for b in grc_common_mfs_count[a][share_count]:
                if (a, b) not in counted:
                    counted.add((a, b))
                    counted.add((b, a))
                    pairs.append((a, b))
                    summary = analyze_pc_pattern(a, b)
                    if summary and len(summary[1]) >= 4:
                        pairs_min4.append((a, b))
    print(f"share_count={share_count}: {len(pairs)}")
    print(f"share_count_min4={share_count}: {len(pairs_min4)}")


histogram = defaultdict(int)
count = 0
while count < n_samples:
    pair = pairs[int(random.random()*len(pairs))]
    a, b = pair
    summary = analyze_pc_pattern(a, b)
    if summary is not None:
        if len(summary[1]) >= min_pattern_len:
            scores[(a, b)] = summary
            score = int((summary[0]*min_pattern_len)+0.5)
            histogram[score] += 1
            count += 1

for val in range(min_pattern_len+1):
    print(f'{val}, {histogram[val]}')


def run_similarity_simulation(
        share_counts,
        min_pattern_len,
        n_samples
        ):
    histograms = {}
    for share_count in share_counts:
        counted = set()
        scores = {}
        pairs = []
        for a in grc_common_mfs_count:
            if share_count in grc_common_mfs_count[a]:
                for b in grc_common_mfs_count[a][share_count]:
                    if (a, b) not in counted:
                        counted.add((a, b))
                        counted.add((b, a))
                        pairs.append((a, b))
        histogram = defaultdict(int)
        count = 0
        while count < n_samples:
            pair = pairs[int(random.random()*len(pairs))]
            a, b = pair
            summary = analyze_pc_pattern(a, b)
            if summary is not None:
                if len(summary[1]) >= min_pattern_len:
                    scores[(a, b)] = summary
                    score = int((summary[0]*min_pattern_len)+0.5)
                    histogram[score] += 1
                    count += 1
        histograms[share_count] = histogram
    for val in range(min_pattern_len+1):
        for share_count in share_counts:
            print(histograms[share_count][val], end=', ')
        print()



def run_similarity_simulation2(
        share_counts,
        min_pattern_len,
        n_samples,
        no_touches=False,
        no_common_zeros=False,
        ):
    histograms = {}
    for share_count in share_counts:
        counted = set()
        scores = {}
        # pairs = []
        avail = set()
        for a in grc_common_mfs_count:
            if share_count in grc_common_mfs_count[a]:
                for b in grc_common_mfs_count[a][share_count]:
                    if (a, b) not in counted:
                        counted.add((a, b))
                        counted.add((b, a))
                        # pairs.append((a, b))
                avail.add(a)
        histogram = defaultdict(int)
        count = 0
        avail = [k for k in avail]
        while count < n_samples:
            # pair = pairs[int(random.random()*len(pairs))]
            a = avail[int(random.random()*len(avail))]
            bs = grc_common_mfs_count[a][share_count]
            b = bs[int(random.random()*len(bs))]
            summary = analyze_pc_pattern(a, b,
                no_touches=no_touches,
                no_common_zeros=no_common_zeros,
                )
            if summary is not None:
                if len(summary[1]) >= min_pattern_len:
                    scores[(a, b)] = summary
                    score = int((summary[0]*min_pattern_len)+0.5)
                    histogram[score] += 1
                    count += 1
        histograms[share_count] = histogram
    for val in range(min_pattern_len+1):
        for share_count in share_counts:
            print(histograms[share_count][val], end=', ')
        print()



run_similarity_simulation(
    [0, 1, 2], 4, 10000)

run_similarity_simulation2(
    [0, 1, 2], 4, 50)

run_similarity_simulation2(
    [0, 1, 2], 5, 10000)

run_similarity_simulation2(
    [0, 1, 2], 6, 10000)

run_similarity_simulation2(
    [0, 1, 2], 7, 10000)

run_similarity_simulation2(
    [0, 1, 2], 2, 10000)


run_similarity_simulation2(
    [0, 1, 2], 4, 10000, no_touches=True)

run_similarity_simulation2(
    [0, 1, 2], 4, 10000, no_common_zeros=True)




def run_similarity_simulation2_box(
        share_counts,
        min_pattern_len,
        n_samples,
        no_touches=False,
        no_common_zeros=False,
        fout_name=None,
        ):
    datasets = {}
    sums = {}
    for share_count in share_counts:
        counted = set()
        # scores = {}
        # pairs = []
        avail = set()
        sum = 0
        for a in grc_common_mfs_count:
            if share_count in grc_common_mfs_count[a]:
                for b in grc_common_mfs_count[a][share_count]:
                    if (a, b) not in counted:
                        counted.add((a, b))
                        counted.add((b, a))
                        # pairs.append((a, b))
                avail.add(a)
        dataset = []
        count = 0
        avail = [k for k in avail]
        while count < n_samples:
            # pair = pairs[int(random.random()*len(pairs))]
            a = avail[int(random.random()*len(avail))]
            all_bs = grc_common_mfs_count[a][share_count]
            b = all_bs[int(random.random()*len(all_bs))]
            summary = analyze_pc_pattern(a, b,
                no_touches=no_touches,
                no_common_zeros=no_common_zeros,
                )
            if summary is not None:
                if len(summary[1]) >= min_pattern_len:
                    # scores[(a, b)] = summary
                    # score = int((summary[0]*min_pattern_len)+0.5)
                    dataset.append(int(summary[0]*100))
                    sum += int(summary[0]*100)
                    count += 1
        datasets[share_count] = dataset
        sums[share_count] = sum
    for share_count in datasets:
        # for val in datasets[share_count]:
        #     print(val, end=', ')
        # print()
        dataset = datasets[share_count]
        dataset = sorted(dataset)
        data0 = int(n_samples/100*10)
        data1 = int(n_samples/100*35)
        data2 = int(n_samples/100*65)
        data3 = int(n_samples/100*90)
        average = sums[share_count]/n_samples
        print(f'{dataset[data0]}, {dataset[data1]}, {dataset[data2]}, {dataset[data3]}, {average}')
        # print(dataset[share_count])
    # # with open(fout_name, 'w') as fout:
    # for val in range(min_pattern_len+1):
    #     for share_count in share_counts:
    #         print(dataset[share_count][val], end=', ')
    #     print()


run_similarity_simulation2_box(
    [0, 1, 2], 4, 10000)



# counted = set()
# for a in grc_common_mfs_count:
#     for b in grc_common_mfs_count[grc_id][3]:
#     # if 3 in grc_common_mfs_count[grc_id]:
#         # print(f'{grc_id}: {grc_common_mfs_count[grc_id][3]}')



'''Analysis:
- # of contacts wrt # of dendrites
- correlation of MFs to # of synapses
- extract patterns of MFs to PC synapses
'''



