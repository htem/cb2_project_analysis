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

mpd_true = MyPlotData()

mf_share_true_data = compute_mf_share()

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
    # mpd.add_data_point(kind='Data', num_partners=sum)

# test = mpd.to_histogram()
for k in range(max(histogram.keys())+1):
    # print(f'{k}, {histogram[k]}')
    mpd_true.add_data_point(kind='Data', num_partners=k, count=histogram[k])

mpd_true = mpd_true.to_pdf(count_var='count', cumulative=True)


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context("talk")

g = sns.catplot(
    data=mpd_true.to_dataframe(),
    # kind="bar",
    kind="point",
    x="num_partners", y="count", hue="kind",
    # hue_order=['All', 'Per PC', 'Per pf'],
    # hue_order=['All', 'Per PC'],
    ci="sd",
    # palette="dark", alpha=.6,
    markers='.',
    height=6, aspect=1.33
)
# g.despine(left=True)
g.set_axis_labels("# of other similar patterns", "Cumulative Distribution")
# g.add_legend()
g.legend.set_title("")
g.ax.grid(which='both', axis='both')
g.ax.set_ylim([0, None])
g.ax.set_yticks(np.arange(0, 1.05, .1), minor=True)
g.ax.set_yticks(np.arange(0, 1.05, .1))
plt.tight_layout()
plt.savefig("run_mf_grc_2share_201011_cdf_true.png")

# asdf


def count_shuffle_2share(mf_share):
    mpd = MyPlotData()
    histogram = defaultdict(int)
    mf_share_threshold = 2
    for grc in grcs:
        sum = 0
        for share_val in mf_share[grc]:
            if share_val >= mf_share_threshold:
                sum += len(mf_share[grc][share_val])
        histogram[sum] += 1
    for k in range(max(histogram.keys())+1):
        mpd.add_data_point(kind='Shuffle', num_partners=k, count=histogram[k])
    mpd = mpd.to_pdf(count_var='count', cumulative=True)
    return mpd

# def randomize_graph():

# def add_shuffle_data():
#     input_graph.randomize_graph()
#     count_2share()

random.seed(0)

mpd_all = MyPlotData()

print("Generating random graphs")
# for i in range(1000):
for i in range(5):
    print(i)
    input_graph.randomize_graph()
    mf_share_data = compute_mf_share()
    mpd = count_shuffle_2share(mf_share_data)
    mpd_all.append(mpd)

mpd_all.append(mpd_true)

g = sns.catplot(
    data=mpd_all.to_dataframe(),
    kind="point",
    x="num_partners", y="count", hue="kind",
    hue_order=['Data', 'Shuffle'],
    ci="sd",
    # ci=95,
    # palette="dark", alpha=.6,
    markers='.',
    height=6, aspect=1.33
)
# g.despine(left=True)
g.set_axis_labels("# of other similar patterns", "Cumulative Distribution")
# g.add_legend()
g.legend.set_title("")
g.ax.grid(which='both', axis='both')
g.ax.set_ylim([0, None])
g.ax.set_yticks(np.arange(0, 1.05, .1), minor=True)
g.ax.set_yticks(np.arange(0, 1.05, .1))
plt.tight_layout()
# plt.show()
plt.savefig("run_mf_grc_2share_201011_cdf_all_1000_sd.png")
# sns.set_style({'xtick.bottom': True})

