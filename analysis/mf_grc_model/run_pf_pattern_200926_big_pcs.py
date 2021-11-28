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

config_f = "config_pf_200925.json"
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

'''Load grc-pc touch dictionary'''
import compress_pickle
print("Loading pf-pc touch dictionary...")
fname = '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/purkinje/db_pf_contacts_200925.gz'
touches_all = compress_pickle.load(fname)

max_touch_dist = 100
max_touch_dist = 0
max_touch_dist = 60
touches = defaultdict(set)
touches_pc_grc = defaultdict(set)

for pf_id in touches_all:
    for pc_id in touches_all[pf_id]:
        if touches_all[pf_id][pc_id][0] <= max_touch_dist:
            touches[pf_id].add(pc_id)
            touches_pc_grc[pc_id].add(pf_id)


'''Computing actual synapse grc-pc postsyns'''
print("Loading pf-pc synapses...")
pf_postsyn_pcs = defaultdict(set)
pf_postsyn_pcs_count = defaultdict(lambda: defaultdict(int))
pf_synapse_count = defaultdict(int)

pc_presyns = defaultdict(set)
pc_presyns_count = defaultdict(lambda: defaultdict(int))
pc_synapse_count = defaultdict(int)

all_pcs = set()

for n in pfs:
    for s in graph.orphaned_post_segments[n]:
        postsyn = graph.neuron_db.find_neuron_with_segment_id(s)
        pf_synapse_count[n] += 1
        if postsyn is not None and \
                ('pc' in postsyn or 'purkinje' in postsyn):
            if 'pcl' in postsyn:
                continue
            postsyn = postsyn.split('.')[0]
            pf_postsyn_pcs[n].add(postsyn)
            pf_postsyn_pcs_count[n][postsyn] += 1
            pc_presyns[postsyn].add(n)
            pc_presyns_count[postsyn][n] += 1
            pc_synapse_count[postsyn] += 1
            all_pcs.add(postsyn)

all_pcs = [k for k in all_pcs]

for pf in pf_postsyn_pcs:
    pcs = sorted([k for k in pf_postsyn_pcs[pf]])
    print(f'{pf}: {pcs}')


# asdf


def analyze_pattern(a, b,
        touch_db,
        postsyn_db,
        no_touches=False,
        no_common_zeros=False,
        fake=False,
        subsample=False,
        ):
    if not fake:
        all_a = set(touch_db[a])
        ones_a = set(postsyn_db[a])
        all_a |= ones_a
        all_b = set(touch_db[b])
        ones_b = set(postsyn_db[b])
        all_b |= ones_b
        common_both = all_a & all_b
        if no_touches:
            common_both = ones_a | ones_b
        if len(common_both) == 0:
            return None
        if subsample:
            if len(common_both) < subsample:
                return None
            common_both_sub = set()
            common_both = [k for k in common_both]
            while len(common_both_sub) < subsample:
                e = common_both[int(random.random()*len(common_both))]
                common_both_sub.add(e)
            common_both = common_both_sub
    else:
        # fake data
        common_both = set()
        ones_a = set()
        ones_b = set()
        for i in range(fake):
            common_both.add(i)
            if random.random() > .5:
                ones_a.add(i)
            if random.random() > .5:
                ones_b.add(i)
    pattern_a = ''
    pattern_b = ''
    common_both = [k for k in common_both]
    similarity_count = 0
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
            similarity_count += 1
        if pc_id not in ones_a and pc_id not in ones_b:
            if not no_common_zeros:
                similarity_count += 1
    similarity_count = float(similarity_count) / len(common_both)
    summary = (similarity_count, common_both, pattern_a, pattern_b)
    return summary

def analyze_pattern2(a, b,
        get_all,
        get_ones,
        no_touches=False,
        no_common_zeros=False,
        fake=False,
        fake_ones_prob=0.5,
        subsample=False,
        ):
    if fake and subsample:
        # fake data with fixed length
        common_both = set()
        ones_a = set()
        ones_b = set()
        for i in range(subsample):
            common_both.add(i)
            if random.random() > .5:
                ones_a.add(i)
            if random.random() > .5:
                ones_b.add(i)
    else:
        all_a = get_all(a)
        ones_a = get_ones(a)
        all_b = get_all(b)
        ones_b = get_ones(b)
        all_a |= ones_a
        all_b |= ones_b
        common_both = all_a & all_b
        if no_touches:
            common_both = ones_a | ones_b
        if len(common_both) == 0:
            return None
        if fake:
            length = len(common_both)
            common_both = set()
            ones_a = set()
            ones_b = set()
            for i in range(length):
                common_both.add(i)
                if random.random() < fake_ones_prob:
                    ones_a.add(i)
                if random.random() < fake_ones_prob:
                    ones_b.add(i)
        if subsample:
            if len(common_both) < subsample:
                return None
            common_both_sub = set()
            common_both = [k for k in common_both]
            while len(common_both_sub) < subsample:
                e = common_both[int(random.random()*len(common_both))]
                common_both_sub.add(e)
            common_both = common_both_sub
    pattern_a = ''
    pattern_b = ''
    common_both = [k for k in common_both]
    similarity_count = 0
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
            similarity_count += 1
        if pc_id not in ones_a and pc_id not in ones_b:
            if not no_common_zeros:
                similarity_count += 1
    similarity_count = float(similarity_count) / len(common_both)
    summary = (similarity_count, common_both, pattern_a, pattern_b)
    return summary

def compare_pf_patterns(a, b,
        no_touches=False,
        no_common_zeros=False,
        ):
    return analyze_pattern(
        a, b, touches, pf_postsyn_pcs,
        no_touches=no_touches,
        no_common_zeros=no_common_zeros,
        )

def compare_pc_patterns(a, b,
        no_touches=False,
        no_common_zeros=False,
        fake=False,
        subsample=False,
        ):
    return analyze_pattern(
        a, b, touches_pc_grc, pc_presyns,
        no_touches=no_touches,
        no_common_zeros=no_common_zeros,
        fake=fake,
        subsample=subsample,
        )


grc_common_mfs_count = defaultdict(lambda: defaultdict(list))
for i in range(len(grcs)):
    i_set = set(input_graph.grcs[grcs[i]].mfs)
    for j in range(len(grcs)):
        if i == j:
            continue
        j_set = set(input_graph.grcs[grcs[j]].mfs)
        common_mfs = i_set & j_set
        grc_common_mfs_count[grcs[i]][len(common_mfs)].append(grcs[j])


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



def run_pf_similarity_simulation(
        min_pattern_len,
        n_samples,
        min_pc_contacts=50,
        no_touches=False,
        no_common_zeros=False,
        fake=False,
        subsample=False,
        percentile0=10,
        percentile1=25,
        ):
    
    histogram = defaultdict(int)
    count = 0
    counts = []
    if subsample:
        subsample = min_pattern_len
    while count < n_samples:
        a = None
        b = None
        if not fake:
            while a == b:
                a = all_pcs[int(random.random()*len(all_pcs))]
                while pc_contact_counts[a] < min_pc_contacts:
                    a = all_pcs[int(random.random()*len(all_pcs))]
                b = all_pcs[int(random.random()*len(all_pcs))]
                while pc_contact_counts[b] < min_pc_contacts:
                    b = all_pcs[int(random.random()*len(all_pcs))]
        else:
            fake = min_pattern_len
        # print(f'Comparing {a} and {b}')
        summary = compare_pc_patterns(a, b,
            no_touches=no_touches,
            no_common_zeros=no_common_zeros,
            fake=fake,
            subsample=subsample,
            )
        if summary is not None:
            if len(summary[1]) >= min_pattern_len:
                counts.append(summary[0])
                score = int((summary[0]*min_pattern_len)+0.5)
                histogram[score] += 1
                count += 1
    for val in range(min_pattern_len+1):
        print(histogram[val])
    print_box_plot(counts, mult=100,
        percentile0=percentile0,
        percentile1=percentile1,
        )


def run_pc_similarity_simulation(
        min_pattern_len,
        n_samples,
        no_touches=False,
        no_common_zeros=False,
        ):
    
    histogram = defaultdict(int)
    count = 0
    counts = []
    while count < n_samples:
        # pair = pairs[int(random.random()*len(pairs))]
        a = pfs[int(random.random()*len(pfs))]
        b = pfs[int(random.random()*len(pfs))]
        summary = compare_pf_patterns(a, b,
            no_touches=no_touches,
            no_common_zeros=no_common_zeros,
            )
        if summary is not None:
            if len(summary[1]) >= min_pattern_len:
                # print(summary[1])
                # scores[(a, b)] = summary
                counts.append(summary[0])
                score = int((summary[0]*min_pattern_len)+0.5)
                histogram[score] += 1
                count += 1
    for val in range(min_pattern_len+1):
        # print(histogram[val], end=', ')
        print(histogram[val])
    print_box_plot(counts, mult=100)

def compare_all_pc_pattern_pairs2(
        min_pattern_len,
        get_all,
        get_ones,
        percentile0=5,
        percentile1=25,
        subsample=False,
        no_common_zeros=False,
        fake=False,
        fake_ones_prob=.5,
        return_histogram=False,
        return_res=False,
        no_print=False,
        ):
    processed = set()
    histogram = defaultdict(int)
    histogram_map = defaultdict(list)
    counts = []
    if subsample:
        subsample = min_pattern_len
    for pc_a in all_pcs:
        for pc_b in all_pcs:
            if pc_a == pc_b:
                continue
            if (pc_a, pc_b) in processed:
                continue
            processed.add((pc_a, pc_b))
            processed.add((pc_b, pc_a))
            summary =  analyze_pattern2(
                pc_a, pc_b,
                get_all,
                get_ones,
                fake=fake,
                subsample=subsample,
                fake_ones_prob=fake_ones_prob,
                no_common_zeros=no_common_zeros,
                )
            if summary is not None:
                if len(summary[1]) >= min_pattern_len:
                    counts.append(summary[0])
                    score = int((summary[0]*min_pattern_len)+0.5)
                    histogram[score] += 1
                    histogram_map[score].append((pc_a, pc_b))
    if not no_print:
        for val in range(min_pattern_len+1):
            print(histogram[val])
        print_box_plot(counts, mult=100,
            percentile0=percentile0,
            percentile1=percentile1,
            )
    if return_histogram:
        return histogram_map
    elif return_res:
        return counts


def pc_get_ones(pc_id):
    return pc_presyns[pc_id]

def pc_get_all(pc_id):
    return touches_pc_grc[pc_id]

def pc_get_two_syns_plus(pc_id):
    pfs = set()
    for pf in pc_presyns_count[pc_id]:
        if pc_presyns_count[pc_id][pf] >= 2:
            pfs.add(pf)
    return pfs



# pc_all_fake = defaultdict(set)
def gen_fake_data(
        prob_single_synapse=0.5,
        prob_multi_synapse=0.1845723608):
    pc_ones_fake = defaultdict(set)
    pc_multi_fake = defaultdict(set)
    for pc_id in all_pcs:
        for pf_id in touches_pc_grc[pc_id]:
            if random.random() <= prob_single_synapse:
                pc_ones_fake[pc_id].add(pf_id)
                if random.random() <= prob_multi_synapse:
                    pc_multi_fake[pc_id].add(pf_id)
    return pc_ones_fake, pc_multi_fake

pc_ones_fake, pc_multi_fake = gen_fake_data()


def pc_get_ones_fake(pc_id):
    return pc_ones_fake[pc_id]

def pc_get_two_syns_plus_fake(pc_id):
    return pc_multi_fake[pc_id]

compare_pcs = partial(compare_all_pc_pattern_pairs2, 
    get_all=pc_get_all,
    get_ones=pc_get_ones,
    percentile0=5,
    percentile1=25,)

compare_pcs_two_syns = partial(compare_all_pc_pattern_pairs2, 
    get_all=pc_get_ones,
    get_ones=pc_get_two_syns_plus,
    percentile0=5,
    percentile1=25,
    no_print=False,)

compare_pcs_two_syns_fake = partial(compare_all_pc_pattern_pairs2, 
    get_all=pc_get_ones_fake,
    get_ones=pc_get_two_syns_plus_fake,
    percentile0=10,
    percentile1=25,)

compare_pcs_fake = partial(compare_all_pc_pattern_pairs2, 
    get_all=pc_get_all,
    get_ones=pc_get_ones_fake,
    percentile0=5,
    percentile1=25,)


asdf

'''Number of syns per pf'''
histogram = defaultdict(int)
for pf in pfs:
    count = pf_synapse_count[pf]
    histogram[count] += 1

for val in range(max(histogram)+1):
    print(f'{val}, {histogram[val]}')

'''Number of PC contacts per pf'''
histogram = defaultdict(int)
for pf in pfs:
    all_count = len(touches[pf] | pf_postsyn_pcs[pf])
    histogram[all_count] += 1

for val in range(max(histogram)+1):
    print(f'{val}, {histogram[val]}')

'''synapsed'''
histogram = defaultdict(int)
for pf in pfs:
    count = len(pf_postsyn_pcs[pf])
    histogram[count] += 1

for val in range(max(histogram)+1):
    print(f'{val}, {histogram[val]}')

'''ratio between 0s and 1s'''
histogram = defaultdict(int)
counts = []
for pf in pfs:
    ones_count = len(pf_postsyn_pcs[pf])
    all_count = len(touches[pf] | pf_postsyn_pcs[pf])
    count = float(ones_count)/all_count
    counts.append(count)
    count = int((count*6)+0.5)
    histogram[count] += 1

for val in range(max(histogram)+1):
    print(f'{val}, {histogram[val]}')

print_box_plot(counts, mult=100)


'''plot the number of synapses per connection'''

histogram = defaultdict(int)
for pf in pfs:
    for pc in pf_postsyn_pcs_count[pf]:
        count = pf_postsyn_pcs_count[pf][pc]
        histogram[count] += 1

for val in range(max(histogram)+1):
    print(f'{val}, {histogram[val]}')



'''PC ANALYSIS'''

'''Number of touches'''
histogram = defaultdict(int)
pc_contact_counts = {}
for pc in all_pcs:
    all_count = len(touches_pc_grc[pc] | pc_presyns[pc])
    histogram[all_count] += 1
    pc_contact_counts[pc] = all_count

for pc in sorted(pc_contact_counts.keys(), key=lambda e: pc_contact_counts[e]):
    print(f'{pc}: {pc_contact_counts[pc]}')

'''Num connections'''
histogram = defaultdict(int)
pc_connection_counts = {}
for pc in all_pcs:
    all_count = len(pc_presyns[pc])
    histogram[all_count] += 1
    pc_connection_counts[pc] = all_count

for pc in sorted(pc_contact_counts.keys(), key=lambda e: pc_contact_counts[e]):
    print(f'{pc}: {pc_connection_counts[pc]}')


'''Box plot of PC connection vs non connections'''

import my_plot
importlib.reload(my_plot)
from my_plot import MyPlotData, my_box_plot

mpd = MyPlotData()
for pc in sorted(pc_contact_counts.keys(), key=lambda e: pc_contact_counts[e]):
    if pc_contact_counts[pc] < 200:
        continue
    mpd.add_data_point(
        kind='Data', ratio=(pc_connection_counts[pc]/pc_contact_counts[pc]))

importlib.reload(my_plot); my_plot.my_box_plot(mpd, y='ratio', y_lims=[.25, .75])




'''Comparing patterns'''

run_pf_similarity_simulation(10, 10000)
run_pf_similarity_simulation(10, 1000, min_pc_contacts=100)
run_pf_similarity_simulation(10, 100, min_pc_contacts=10)

'''plot fake data spread'''
run_pf_similarity_simulation(10, 10000, fake=True)
run_pf_similarity_simulation(100, 10000, fake=True)
run_pf_similarity_simulation(20, 10000, fake=True)
run_pf_similarity_simulation(200, 10000, fake=True)
'''plot fake data spread'''
run_pf_similarity_simulation(10, 10000, percentile0=5, percentile1=25, fake=True)
run_pf_similarity_simulation(20, 10000, percentile0=5, percentile1=25, fake=True)
run_pf_similarity_simulation(30, 10000, percentile0=5, percentile1=25, fake=True)
run_pf_similarity_simulation(50, 10000, percentile0=5, percentile1=25, fake=True)
run_pf_similarity_simulation(100, 10000, percentile0=5, percentile1=25, fake=True)
run_pf_similarity_simulation(200, 10000, percentile0=5, percentile1=25, fake=True)

all_pcs_with_somas = ['pc_0', 'pc_1', 'purkinje_0', 'purkinje_1', 'pc_2', 'pc_3', 'pc_4', 'pc_5', 'pc_6', 'pc_7', 'pc_8', 'pc_9', 'pc_10', 'pc_11', 'pc_12', 'pc_13', 'pc_15', 'pc_16', 'pc_17', 'pc_18', 'pc_19', 'pc_20', 'pc_21', 'pc_22', 'pc_23', 'pc_24', 'pc_25', 'pc_26', 'pc_28', 'pc_29', 'pc_30', 'pc_31', 'pc_32', 'pc_33', 'pc_34', 'pc_35', 'pc_36', 'pc_37', 'pc_38', 'pc_39', 'pc_53', 'pc_54', 'pc_56', 'pc_57', 'pc_58', 'pc_163']
all_pcs_with_somas = set(all_pcs_with_somas) & set(pc_contact_counts.keys())
all_pcs_with_somas = [k for k in all_pcs_with_somas]
print(all_pcs_with_somas)

def compare_all_pc_pattern_pairs(
        min_pattern_len,
        min_pc_contacts=0,
        percentile0=10,
        percentile1=25,
        subsample=False,
        ):
    processed = set()
    histogram = defaultdict(int)
    histogram_map = defaultdict(list)
    counts = []
    if subsample:
        subsample = min_pattern_len
    for pc_a in all_pcs:
        if pc_contact_counts[pc_a] < min_pc_contacts:
            continue
        for pc_b in all_pcs:
            if pc_contact_counts[pc_b] < min_pc_contacts:
                continue
            if pc_a == pc_b:
                continue
            if (pc_a, pc_b) in processed:
                continue
            processed.add((pc_a, pc_b))
            processed.add((pc_b, pc_a))
            summary = compare_pc_patterns(
                pc_a, pc_b,
                subsample=subsample,
                )
            if summary is not None:
                if len(summary[1]) >= min_pattern_len:
                    counts.append(summary[0])
                    score = int((summary[0]*min_pattern_len)+0.5)
                    histogram[score] += 1
                    histogram_map[score].append((pc_a, pc_b))
    for val in range(min_pattern_len+1):
        print(histogram[val])
    print_box_plot(counts, mult=100,
        percentile0=percentile0,
        percentile1=percentile1,
        )
    return histogram_map



histogram_map = compare_all_pc_pattern_pairs(10, 0, 5, 25)
histogram_map = compare_all_pc_pattern_pairs(20, 0, 5, 25)
histogram_map = compare_all_pc_pattern_pairs(30, 0, 5, 25)
histogram_map = compare_all_pc_pattern_pairs(40, 0, 5, 25)

histogram_map = compare_all_pc_pattern_pairs(20, 0, 5, 25, subsample=True)
histogram_map = compare_all_pc_pattern_pairs(30, 0, 5, 25, subsample=True)
histogram_map = compare_all_pc_pattern_pairs(50, 0, 5, 25, subsample=True)
histogram_map = compare_all_pc_pattern_pairs(100, 0, 5, 25, subsample=True)

histogram_map = compare_all_pc_pattern_pairs(20, 0, subsample=True)
histogram_map = compare_all_pc_pattern_pairs(30, 0, subsample=True)
histogram_map = compare_all_pc_pattern_pairs(50, 0, subsample=True)
histogram_map = compare_all_pc_pattern_pairs(100, 0, subsample=True)


compare_pcs(min_pattern_len=20)
compare_pcs(min_pattern_len=20, fake=True, fake_ones_prob=.2)
compare_pcs(min_pattern_len=20, fake=True, fake_ones_prob=.15)
compare_pcs(min_pattern_len=40)
compare_pcs(min_pattern_len=40, fake=True, fake_ones_prob=.2)

pc_ones_fake, pc_multi_fake = gen_fake_data(); compare_pcs_fake(min_pattern_len=20)

'''Multi-synapse similarity'''

compare_pcs_two_syns(min_pattern_len=20)
compare_pcs_two_syns(min_pattern_len=20, fake=True, fake_ones_prob=.2)
compare_pcs_two_syns(min_pattern_len=40)
compare_pcs_two_syns(min_pattern_len=40, fake=True, fake_ones_prob=.2)
compare_pcs_two_syns(min_pattern_len=40, fake=True, fake_ones_prob=.19)
compare_pcs_two_syns(min_pattern_len=40, fake=True, fake_ones_prob=.18)
compare_pcs_two_syns(min_pattern_len=40, fake=True, fake_ones_prob=.17)

compare_pcs_two_syns(min_pattern_len=30, no_common_zeros=True)
compare_pcs_two_syns(min_pattern_len=30, no_common_zeros=True, fake=True, fake_ones_prob=.2)

compare_pcs_two_syns(min_pattern_len=40, no_common_zeros=True)
compare_pcs_two_syns(min_pattern_len=40, no_common_zeros=True, fake=True, fake_ones_prob=.2)


'''Generate fake data'''

pc_ones_fake, pc_multi_fake = gen_fake_data()
fake_data = compare_pcs_two_syns_fake(min_pattern_len=10, return_res=True)
fake_data = compare_pcs_two_syns_fake(min_pattern_len=20, return_res=True)
fake_data = compare_pcs_two_syns_fake(min_pattern_len=40, return_res=True)


compare_pcs_two_syns_fake(min_pattern_len=40)


true_data = compare_pcs_two_syns(min_pattern_len=10, return_res=True)
true_data = compare_pcs_two_syns(min_pattern_len=20, return_res=True)
true_data = compare_pcs_two_syns(min_pattern_len=40, return_res=True)

stats.ttest_ind(true_data, fake_data)

pc_ones_fake, pc_multi_fake = gen_fake_data()
fake_data = compare_pcs_two_syns_fake(min_pattern_len=10, return_res=True)
# stats.ttest_ind(true_data, fake_data)
scipy.stats.ranksums(true_data, fake_data)







#define F-test function
def f_test(x, y):
    x = np.array(x)
    y = np.array(y)
    f = np.var(x, ddof=1)/np.var(y, ddof=1) #calculate F test statistic 
    dfn = x.size-1 #define degrees of freedom numerator 
    dfd = y.size-1 #define degrees of freedom denominator 
    p = 1-scipy.stats.f.cdf(f, dfn, dfd) #find p-value of F test statistic 
    return f, p

pc_ones_fake, pc_multi_fake = gen_fake_data()
fake_data = compare_pcs_two_syns_fake(min_pattern_len=10, return_res=True)
# f_test(true_data, fake_data)
stats.ttest_ind(true_data, fake_data)


fake_data = []
for i in range(1000):
# for i in range(500):
# for i in range(100):
    pc_ones_fake, pc_multi_fake = gen_fake_data()
    fake_data.extend(compare_pcs_two_syns_fake(min_pattern_len=10, return_res=True, no_print=True))

scipy.stats.ranksums(true_data, fake_data)
stats.ttest_ind(true_data, fake_data)


'''Figure: plot box distribution of real vs fake multisynapses'''

sns.set_style('whitegrid')
plot_data = {
    'Data': true_data,
    'Random': fake_data,
    }
# plot_data = pd.DataFrame(plot_data)
plot_data = [true_data, fake_data]
ax = sns.boxplot(data=plot_data,
    linewidth=2.5,
    # height=4,
    # aspect=1.7,
    whis=(10, 90),
    )
plt.show()



for i in [6, 7]:
    print(f"Similarity {i}")
    for e in histogram_map[i]:
        print(f'{e[0]}.soma_0, {e[1]}.soma_0')


for i in [2, 3, 4]:
    print(f"Similarity {i}")
    for e in histogram_map[i]:
        print(f'{e[0]}.soma_0, {e[1]}.soma_0')






'''Plot pdf of common pattern sim between PCs'''
processed = set()
histogram = defaultdict(int)
histogram_map = defaultdict(list)
counts = []
for pc_a in all_pcs_with_somas:
    if pc_contact_counts[pc_a] < min_pc_contacts:
        continue
    for pc_b in all_pcs_with_somas:
        if pc_contact_counts[pc_b] < min_pc_contacts:
            continue
        if pc_a == pc_b:
            continue
        if (pc_a, pc_b) in processed:
            continue
        processed.add((pc_a, pc_b))
        processed.add((pc_b, pc_a))
        summary = compare_pc_patterns(pc_a, pc_b)
        if summary is not None:
            pattern_len = summary[0]
            counts.append(pattern_len)
            # histogram[pattern_len] += 1
            # histogram_map[pattern_len].append((pc_a, pc_b))

for val in sorted(counts):
    print(val, end=', ')



'''Plot pdf of common pattern len between PCs'''
processed = set()
histogram = defaultdict(int)
histogram_map = defaultdict(list)
counts = []
for pc_a in all_pcs_with_somas:
    if pc_contact_counts[pc_a] < min_pc_contacts:
        continue
    for pc_b in all_pcs_with_somas:
        if pc_contact_counts[pc_b] < min_pc_contacts:
            continue
        if pc_a == pc_b:
            continue
        if (pc_a, pc_b) in processed:
            continue
        processed.add((pc_a, pc_b))
        processed.add((pc_b, pc_a))
        summary = compare_pc_patterns(pc_a, pc_b)
        if summary is not None:
            pattern_len = len(summary[1])
            counts.append(pattern_len)
            # histogram[pattern_len] += 1
            # histogram_map[pattern_len].append((pc_a, pc_b))

for val in sorted(counts):
    print(val, end=', ')


asdf

'''Figure: ratio between logical ones and logical zeros'''

import pandas as pd

histogram = defaultdict(int)
counts = []
for pf in pfs:
    ones_count = len(pf_postsyn_pcs[pf])
    all_count = len(touches[pf])
    # if ones_count > all_count:
    #     print(pf)
    #     print(pf_postsyn_pcs[pf] - touches[pf])
    #     # print(pf_postsyn_pcs[pf])
    #     # print(touches[pf])
    all_count = len(touches[pf] | pf_postsyn_pcs[pf])
    count = float(ones_count)/all_count
    counts.append(count)
    count = count * 6
    count += 0.5
    count = int(count)
    histogram[count] += 1

for val in range(max(histogram)+1):
    print(f'{val}, {histogram[val]}')

print_box_plot(counts, mult=100)

sns.set_style('whitegrid')

plot_label = 'connections / total (per pf)'
plot_data = {
    plot_label: counts
    }

ax = sns.boxplot(y=plot_label, data=plot_data,
    linewidth=2.5,
    # height=4,
    # aspect=1.7,
    whis=(10, 90),
    )
plt.show()



'''Analysis:
- # of synapses/contacts per pf
    the ratio overall and distribution of ratio
    compute the ratio between 0 and 1
    *** wrt Y location (pia vs PC proximate)

- distribution of the # of contacts of 1s
    multi synapses
    *** wrt Y location (pia vs PC proximate)

- plot similarities between random PFs

- baseline PC projection similarity between pfs
    maybe randomly sample 100/1000/10000 out of all possible combinations
    can also see their spread

PC pattern

- get num connections per PC
- get synapse distribution per PC per connection
    probably need to sample to ensure even distribution
    basal vs distal

- compare touches/synapses (0 vs 1) between PCs
    between branches of PCs
    basal vs distal

- plot the histogram of hamming distances between all possible pairs
- map locations of pairs


'''



'''Multi-synapses studies
- plot spread of multisyns per pf
- plot spread per PC
'''

'''All stats'''
all_syn_count_histogram = defaultdict(int)
total = 0
for pf in pfs:
    for pc in pf_postsyn_pcs_count[pf]:
        count = pf_postsyn_pcs_count[pf][pc]
        all_syn_count_histogram[count] += 1
        total += 1

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

plot_data = {
    'type': [],
    'num_syns_per_connection': [],
    'pct': [],
}

type = 'Per PC'
for num_syn in pc_syns_histogram_list:
    if num_syn == 0:
        continue
    for pct in pc_syns_histogram_list[num_syn]:
        plot_data['type'].append(type)
        plot_data['num_syns_per_connection'].append(num_syn)
        plot_data['pct'].append(pct)

type = 'All'
for num_syn in all_syn_count_histogram:
    if num_syn == 0:
        continue
    count = all_syn_count_histogram[num_syn]
    pct = float(count) / total
    plot_data['type'].append(type)
    plot_data['num_syns_per_connection'].append(num_syn)
    plot_data['pct'].append(pct)


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

type = 'Per pf'
for num_syn in pf_syn_count_histogram_list:
    if num_syn == 0:
        continue
    for pct in pf_syn_count_histogram_list[num_syn]:
        plot_data['type'].append(type)
        plot_data['num_syns_per_connection'].append(num_syn)
        plot_data['pct'].append(pct)


plot_df = pd.DataFrame.from_dict(plot_data)
g = sns.catplot(
    data=plot_df,
    kind="bar",
    # kind="point",
    # x="type", y="pct", hue="num_syns_per_connection",
    x="num_syns_per_connection", y="pct", hue="type",
    # hue_order=['All', 'Per PC', 'Per pf'],
    hue_order=['All', 'Per PC'],
    ci="sd",
    # palette="dark", alpha=.6,
    # height=6, aspect=1.33
)
# g.despine(left=True)
g.set_axis_labels("# of synapses per connection", "Normalized Frequency")
# g.add_legend()
g.legend.set_title("")
plt.show()


'''Count the number of connections vs synapses'''



'''201014 plot of fake PC data'''
import my_plot

plot_mpd = my_plot.MyPlotData()
plot_mpd_cdf = my_plot.MyPlotData()
min_pattern_len=40
fake_data_total = []
# for i in range(1000):
# for i in range(500):
for i in range(1000):
    fake_data = []
    # pc_ones_fake, pc_multi_fake = gen_fake_data(prob_single_synapse=0.55)
    # pc_ones_fake, pc_multi_fake = gen_fake_data(prob_single_synapse=0.4892531411)
    pc_ones_fake, pc_multi_fake = gen_fake_data(prob_single_synapse=0.4868552608)
    fake_data = compare_pcs_fake(min_pattern_len=min_pattern_len, return_res=True, no_print=True)
    fake_data_total.extend(fake_data)
    histogram = defaultdict(int)
    mpd = my_plot.MyPlotData()
    for data_point in fake_data:
        score = int((data_point*min_pattern_len)+0.5)
        score*=(100/min_pattern_len)
        histogram[score] += 1
    for k in sorted(histogram.keys()):
        mpd.add_data_point(kind='Random', pct=k, count=histogram[k])
    mpd = mpd.to_pdf('count')
    plot_mpd.append(mpd)
    # mpd = mpd.to_pdf('count', cumulative=True)
    plot_mpd_cdf.append(mpd)

# importlib.reload(my_plot); my_plot.my_cat_bar_plot(plot_mpd, x='pct', y='count', hue='kind', x_axis_label='Pairwise Similarity (%)', y_axis_label='Normalized Frequency', save_filename='fig_pc_pattern_random.png', show=True)
# importlib.reload(my_plot); my_plot.my_cat_bar_plot(plot_mpd, x='pct', y='count', hue='kind', x_axis_label='Pairwise Similarity (%)', y_axis_label='Normalized Frequency')

# importlib.reload(my_plot); my_plot.my_cat_bar_plot(plot_mpd, plot_type='point', x='pct', y='count', hue='kind', x_axis_label='Pairwise Similarity (%)', y_axis_label='Normalized Frequency')
hue_order=['Random']

xticklabels=['', '15', '' '25', '', '30', '', '35', '', '40', '', '45', '', '50', '', '55', '', '60', '', '65', '', '70', '', '75', '', '80', '', '85', '', '90', '', '95', '', '100']
importlib.reload(my_plot); my_plot.my_cat_bar_plot(plot_mpd, x='pct', y='count', hue='kind', hue_order=hue_order, x_axis_label='Pairwise Similarity (%)', y_axis_label='Normalized Frequency', xticklabels=xticklabels)

if True:
    true_data = compare_pcs(min_pattern_len=min_pattern_len, return_res=True, no_print=True)
    histogram = defaultdict(int)
    mpd = my_plot.MyPlotData()
    for data_point in true_data:
        score = int((data_point*min_pattern_len)+0.5)
        score*=(100/min_pattern_len)
        histogram[score] += 1
    # for k in histogram:
    for k in sorted(histogram.keys()):
        mpd.add_data_point(kind='Data', pct=k, count=histogram[k])
    mpd = mpd.to_pdf('count')
    plot_mpd.append(mpd)
    # mpd = mpd.to_pdf('count', cumulative=True)
    plot_mpd_cdf.append(mpd)

# importlib.reload(my_plot); my_plot.my_cat_bar_plot(plot_mpd, x='pct', y='count', hue='kind', x_axis_label='Pairwise Similarity (%)', y_axis_label='Normalized Frequency', save_filename='fig_pc_pattern_random_vs_data.png', show=True)
# importlib.reload(my_plot); my_plot.my_cat_bar_plot(plot_mpd, x='pct', y='count', hue='kind', x_axis_label='Pairwise Similarity (%)', y_axis_label='Normalized Frequency')

xticklabels=['', '25', '', '30', '', '35', '', '40', '', '45', '', '50', '', '55', '', '60', '', '65', '', '70', '', '75', '', '80', '', '85', '', '90', '', '95', '', '100']
importlib.reload(my_plot); my_plot.my_cat_bar_plot(plot_mpd, x='pct', y='count', hue='kind', x_axis_label='Pairwise Similarity (%)', y_axis_label='Normalized Frequency', xticklabels=xticklabels)
# importlib.reload(my_plot); my_plot.my_cat_bar_plot(plot_mpd, plot_type='point', x='pct', y='count', hue='kind', x_axis_label='Pairwise Similarity (%)', y_axis_label='Normalized Frequency')


scipy.stats.ttest_ind(true_data, fake_data_total)
scipy.stats.ranksums(true_data, fake_data_total)
scipy.stats.ks_2samp(true_data, fake_data_total)





