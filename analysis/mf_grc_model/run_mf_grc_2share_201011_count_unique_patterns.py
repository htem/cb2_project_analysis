# import collections
from collections import defaultdict
import sys
# import json
import random
# from jsmin import jsmin
# from io import StringIO
import numpy as np
# import copy
# import importlib
import compress_pickle

sys.path.insert(0, '/n/groups/htem/Segmentation/shared-dev/cb2_segmentation/segway.graph.tmn7')
# from segway.graph.synapse_graph import SynapseGraph
# from segway.graph.plot_adj_mat import plot_adj_mat

sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')
# from tools import *
# import tools_mf_graph
from my_plot import MyPlotData

def to_ng_coord(coord):
    return (
        int(coord[0]/4),
        int(coord[1]/4),
        int(coord[2]/40),
        )

input_graph = compress_pickle.load('input_graph_201011_restricted_z.gz')
grcs = [k for k in input_graph.grcs.keys()]

mpd = MyPlotData()

def compute_mf_share():
    count = defaultdict(lambda: defaultdict(list))
    for i in range(len(grcs)):
        i_set = set(input_graph.grcs[grcs[i]].mfs)
        for j in range(len(grcs)):
            if i == j:
                continue
            j_set = set(input_graph.grcs[grcs[j]].mfs)
            common_mfs = i_set & j_set
            if len(common_mfs) > 1:
                count[grcs[i]][len(common_mfs)].append(grcs[j])
    return count

mpd = MyPlotData()

mf_share_true_data = compute_mf_share()

true_data = []
fake_data = []

'''doubles'''
histogram = defaultdict(int)
doubles_count = []
mf_share_threshold = 2
for grc in grcs:
    sum = 0
    for share_val in mf_share_true_data[grc]:
        if share_val >= mf_share_threshold:
            sum += len(mf_share_true_data[grc][share_val])
    # doubles_count.append(sum)
    histogram[sum] += 1
    true_data.append(sum)
    # mpd.add_data_point(kind='Data', num_partners=sum)

# test = mpd.to_histogram()
for k in range(max(histogram.keys())+1):
    # print(f'{k}, {histogram[k]}')
    mpd.add_data_point(kind='Data', num_partners=k, count=histogram[k])



def count_shuffle_2share(mf_share):
    histogram = defaultdict(int)
    mf_share_threshold = 2
    for grc in grcs:
        sum = 0
        for share_val in mf_share[grc]:
            if share_val >= mf_share_threshold:
                sum += len(mf_share[grc][share_val])
        histogram[sum] += 1
        fake_data.append(sum)
    for k in range(max(histogram.keys())+1):
        mpd.add_data_point(kind='Shuffle', num_partners=k, count=histogram[k])

# def randomize_graph():

# def add_shuffle_data():
#     input_graph.randomize_graph()
#     count_2share()

def count_unique_patterns(graph):
    patterns = set()
    for grc in graph.grcs:
        pattern = tuple(sorted([mf for mf, loc in graph.grcs[grc].edges]))
        patterns.add(pattern)
    print(len(patterns))
    print(len(input_graph.grcs))


asdf

random.seed(0)

print("Generating random graphs")
for i in range(10):
    print(i)
    input_graph.randomize_graph()
    mf_share_data = compute_mf_share()
    count_shuffle_2share(mf_share_data)
    count_unique_patterns(input_graph)


# compute KS p value
from scipy import stats
stats.ks_2samp(true_data, fake_data)
stats.brunnermunzel(true_data, fake_data)  # permutation test

# stats.ks_2samp(true_data, fake_data, alternative='greater')
# stats.ks_2samp(true_data, fake_data, alternative='less')


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

g = sns.catplot(
    data=mpd.to_dataframe(),
    kind="bar",
    # kind="point",
    x="num_partners", y="count", hue="kind",
    # hue_order=['All', 'Per PC', 'Per pf'],
    # hue_order=['All', 'Per PC'],
    ci="sd",
    # palette="dark", alpha=.6,
    height=6, aspect=1.33
)
# g.despine(left=True)
g.set_axis_labels("# of other similar patterns", "# of GrCs")
# g.add_legend()
g.legend.set_title("")
# plt.show()
plt.tight_layout()
plt.savefig("run_mf_grc_2share_201011.png")

