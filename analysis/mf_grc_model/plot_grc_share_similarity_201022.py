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

sys.path.insert(0, '/n/groups/htem/Segmentation/shared-dev/cb2_segmentation/segway.graph.tmn7')
from segway.graph.synapse_graph import SynapseGraph
from segway.graph.plot_adj_mat import plot_adj_mat

sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')
from tools import *
import tools_mf_graph

import tools_pattern

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

for n in grc_postsyn_pcs:
    touches[n] |= grc_postsyn_pcs[n]


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

share_count = 2
min_pattern_len = 4
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

run_similarity_simulation2(
    [0, 1, 2], 4, 10000)




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



def compute_similarity(
        share_count,
        min_pattern_len,
        get_all,
        get_ones,
        no_touches=False,
        dot_product=False,
        neuron_list=None,
        return_data=False,
        no_print=False,
        ):
    counted = set()
    scores = {}
    avail = set()
    histogram = defaultdict(int)
    data_points = []
    if neuron_list is None:
        neuron_list = list(grc_common_mfs_count.keys())
    for a in neuron_list:
        for b in neuron_list:
        # if share_count in grc_common_mfs_count[a]:
            # for b in grc_common_mfs_count[a][share_count]:
            if b in grc_common_mfs_count[a][share_count]:
                if (a, b) not in counted:
                    counted.add((a, b))
                    counted.add((b, a))
                    # print(counted)
                    summary = tools_pattern.compute_hamming_distance(
                        a, b,
                        get_all=get_all,
                        get_ones=get_ones,
                        )
                    if summary is not None:
                        if len(summary[1]) >= min_pattern_len:
                            scores[(a, b)] = summary
                            data_points.append(summary[0])
                            score = int((summary[0]*min_pattern_len)+0.5)
                            histogram[score] += 1
    if not no_print:
        for val in range(min_pattern_len+1):
            print(histogram[val], end=', ')
            print()
    if return_data:
        return data_points

def grc_get_pattern_ones(grc_id):
    return grc_postsyn_pcs[grc_id]

def grc_get_pattern_all(grc_id):
    return touches[grc_id]

importlib.reload(tools_pattern)

compare_grcs = partial(
    compute_similarity, 
    get_all=grc_get_pattern_all,
    get_ones=grc_get_pattern_ones,
    # percentile0=5,
    # percentile1=25,
    )

compare_grcs(2, 4)

f = '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/notes_grcs_with_full_axons_201022.json'
with open(f) as fin:
    grcs_full_axon = json.load(fin).get('list')

compare_grcs_full = partial(
    compute_similarity, 
    get_all=grc_get_pattern_all,
    get_ones=grc_get_pattern_ones,
    # percentile0=5,
    # percentile1=25,
    neuron_list=grcs_full_axon,
    )

compare_grcs_full(2, 4)

## Get connection rate of local GrCs
histogram = defaultdict(int)
bins = 4
min_contact_count = 4
num_touches = 0
num_connections = 0
for grc_id in grcs_full_axon:
    touch_count = len(touches[grc_id])
    if touch_count < min_contact_count:
        continue
    connection_count = len(grc_postsyn_pcs[grc_id])
    num_touches += touch_count
    num_connections += connection_count
    score = float(connection_count)/touch_count
    score = int((score*bins)+0.5)
    if score == 0:
        print(grc_id)
    # score *= 100/bins
    histogram[score] += 1

print(f'Total contacts: {num_touches}')
print(f'Total connections: {num_connections}')
print(f'Ratio: {num_connections/num_touches}')

# >>> print(f'Total contacts: {num_touches}')
# Total contacts: 2680
# >>> print(f'Total connections: {num_connections}')
# Total connections: 1765
# >>> print(f'Ratio: {num_connections/num_touches}')
# Ratio: 0.6585820895522388

for val in range(bins+1):
    print(histogram[val], end=', ')
    print()


random_ones = defaultdict(set)
random_ones_0p65858 = defaultdict(set)

def randomize_grc_pc_connections():
    global random_ones
    global random_ones_0p65858
    random_ones = defaultdict(set)
    random_ones_0p65858 = defaultdict(set)
    for grc_id in grcs_full_axon:
        for pc in touches[grc_id]:
            if random.random() < 0.5:
                random_ones[grc_id].add(pc)
            if random.random() < 0.6585820895522388:
                random_ones_0p65858[grc_id].add(pc)
    # return random_ones, random_ones_0p65858

# random_ones, random_ones_0p65858 = randomize_grc_pc_connections()

def grc_get_random_ones(grc_id):
    return random_ones[grc_id]

def grc_get_random_ones_0p65858(grc_id):
    return random_ones_0p65858[grc_id]

compare_grcs_random = partial(
    compute_similarity, 
    get_all=grc_get_pattern_all,
    get_ones=grc_get_random_ones,
    # percentile0=5,
    # percentile1=25,
    neuron_list=grcs_full_axon,
    )

randomize_grc_pc_connections(); compare_grcs_random(2, 4)

compare_grcs_random_0p65 = partial(
    compute_similarity, 
    get_all=grc_get_pattern_all,
    get_ones=grc_get_random_ones_0p65858,
    # percentile0=5,
    # percentile1=25,
    neuron_list=grcs_full_axon,
    )

randomize_grc_pc_connections(); compare_grcs_random_0p65(2, 4)

## Calculate significance
import scipy
import scipy.stats

share_2_data = compare_grcs_full(2, 4, return_data=True, no_print=True)
share_2_data_random = []
share_2_data_random_0p65 = []
for i in range(100):
    randomize_grc_pc_connections()
    share_2_data_random.extend(compare_grcs_random(2, 4, return_data=True, no_print=True))
    share_2_data_random_0p65.extend(compare_grcs_random_0p65(2, 4, return_data=True, no_print=True))

scipy.stats.ttest_ind(share_2_data, share_2_data_random)
scipy.stats.ranksums(share_2_data, share_2_data_random)
scipy.stats.ttest_ind(share_2_data, share_2_data_random_0p65)
scipy.stats.ranksums(share_2_data, share_2_data_random_0p65)

share_1_data = compare_grcs_full(1, 4, return_data=True, no_print=True)
share_1_data_random = []
share_1_data_random_0p65 = []
for i in range(20):
    randomize_grc_pc_connections()
    share_1_data_random.extend(compare_grcs_random(1, 4, return_data=True, no_print=True))
    share_1_data_random_0p65.extend(compare_grcs_random_0p65(1, 4, return_data=True, no_print=True))

scipy.stats.ttest_ind(share_1_data, share_1_data_random)
scipy.stats.ranksums(share_1_data, share_1_data_random)
scipy.stats.ttest_ind(share_1_data, share_1_data_random_0p65)
scipy.stats.ranksums(share_1_data, share_1_data_random_0p65)

share_0_data = compare_grcs_full(0, 4, return_data=True, no_print=True)
share_0_data_random = []
share_0_data_random_0p65 = []
for i in range(10):
    randomize_grc_pc_connections()
    share_0_data_random.extend(compare_grcs_random(0, 4, return_data=True, no_print=True))
    share_0_data_random_0p65.extend(compare_grcs_random_0p65(0, 4, return_data=True, no_print=True))

scipy.stats.ttest_ind(share_0_data, share_0_data_random)
scipy.stats.ranksums(share_0_data, share_0_data_random)
scipy.stats.ttest_ind(share_0_data, share_0_data_random_0p65)
scipy.stats.ranksums(share_0_data, share_0_data_random_0p65)


# scipy.stats.ranksums(share_2_data, share_0_data)
# scipy.stats.ranksums(share_2_data, share_1_data)
# scipy.stats.ranksums(share_0_data, share_1_data)
scipy.stats.ttest_ind(share_2_data, share_0_data)
scipy.stats.ttest_ind(share_2_data, share_1_data)
scipy.stats.ttest_ind(share_0_data, share_1_data)
scipy.stats.mannwhitneyu(share_2_data, share_0_data, alternative='greater')
scipy.stats.mannwhitneyu(share_2_data, share_1_data, alternative='greater')
scipy.stats.mannwhitneyu(share_1_data, share_0_data, alternative='greater')



