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
        ):
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

def analyze_pc_pattern(a, b,
        no_touches=False,
        no_common_zeros=False,
        ):
    return analyze_pattern(
        a, b, touches, pf_postsyn_pcs,
        no_touches=no_touches,
        no_common_zeros=no_common_zeros,
        )

def analyze_pf_pattern(a, b,
        no_touches=False,
        no_common_zeros=False,
        ):
    return analyze_pattern(
        a, b, touches_pc_grc, pc_presyns,
        no_touches=no_touches,
        no_common_zeros=no_common_zeros,
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
        percentile1=35,
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
        ):
    
    histogram = defaultdict(int)
    count = 0
    counts = []
    while count < n_samples:
        a = None
        b = None
        while a == b:
            a = all_pcs[int(random.random()*len(all_pcs))]
            while pc_contact_counts[a] < min_pc_contacts:
                a = all_pcs[int(random.random()*len(all_pcs))]
            b = all_pcs[int(random.random()*len(all_pcs))]
            while pc_contact_counts[b] < min_pc_contacts:
                b = all_pcs[int(random.random()*len(all_pcs))]
        # print(f'Comparing {a} and {b}')
        summary = analyze_pf_pattern(a, b,
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
        summary = analyze_pc_pattern(a, b,
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

'''Comparing patterns'''

run_pf_similarity_simulation(10, 10000)
run_pf_similarity_simulation(10, 1000, min_pc_contacts=100)
run_pf_similarity_simulation(10, 100, min_pc_contacts=10)

def calc_all_pf_patterns(
        min_pattern_len,
        min_pc_contacts=0,
        percentile0=10,
        percentile1=35,
        ):
    processed = set()
    histogram = defaultdict(int)
    histogram_map = defaultdict(list)
    counts = []
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
            summary = analyze_pf_pattern(pc_a, pc_b)
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

calc_all_pf_patterns(10, 0, 5, 25)
calc_all_pf_patterns(20, 0, 5, 25)
calc_all_pf_patterns(30, 0, 5, 25)
calc_all_pf_patterns(40, 0, 5, 25)

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

'''Get pfs similarity'''
run_pf_similarity_simulation(4, 10000)
# test
run_pf_similarity_simulation(8, 1000)



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
#     for b in grc_common_mfs_count[pf_id][3]:
#     # if 3 in grc_common_mfs_count[pf_id]:
#         # print(f'{pf_id}: {grc_common_mfs_count[pf_id][3]}')


