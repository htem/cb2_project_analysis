from collections import defaultdict
import sys
import random
import numpy as np
import compress_pickle
import importlib
import copy
import os

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

input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114_restricted_z.gz')
grcs = [k for k in input_graph.grcs.keys()]

print(f"GT unique patterns: {input_graph.count_unique_patterns()}")
print(f"GT unsampled mfs: {input_graph.count_unsampled_mfs()}")
input_graph.print_mfs_connected_summary()

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

mpd_true_cdf = copy.deepcopy(mpd_true)
mpd_true_cdf = mpd_true_cdf.to_pdf(count_var='count', cumulative=True)
mpd_true = mpd_true.to_pdf(count_var='count', cumulative=False)


import my_plot
importlib.reload(my_plot)
from my_plot import MyPlotData, my_relplot

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
    mpd_cdf = copy.deepcopy(mpd)
    mpd = mpd.to_pdf(count_var='count', cumulative=False)
    mpd_cdf = mpd_cdf.to_pdf(count_var='count', cumulative=True)
    return mpd, mpd_cdf

# def randomize_graph():

# def add_shuffle_data():
#     input_graph.randomize_graph()
#     count_2share()


# mf_dist_margin = 500
# mf_dist_margin = 20000
# mf_dist_margin = 1000
# mf_dist_margin = 4000
# mf_dist_margin = 6000
# mf_dist_margin = 8000
# mf_dist_margin = 10000
mf_dist_margin = 5000

n_random = 1000
n_random = 5

for mf_dist_margin in [1000, 2000, 3000, 4000]:
# for mf_dist_margin in [2000, 5000, 10000, 20000]:

    random.seed(0)
    mpd_all = MyPlotData()
    mpd_all_cdf = MyPlotData()
    print(f"Generating random graphs for {mf_dist_margin}")
    # for i in range(1000):
    for i in range(n_random):
        print(i)
        input_graph.randomize_graph_by_mf(
            mf_dist_margin=mf_dist_margin,
            single_connection_per_pair=True,
            preserve_in_degree=True,
            local_lengths=True,
            )
        print(f"Random graph unique patterns: {input_graph.count_unique_patterns()}")
    print(f"Random graph unsampled mfs: {input_graph.count_unsampled_mfs()}/{len(input_graph.mfs)}")
    input_graph.print_mfs_connected_summary()
    print()
    mf_share_data = compute_mf_share()
    mpd, mpd_cdf = count_shuffle_2share(mf_share_data)
    mpd_all.append(mpd)
    mpd_all_cdf.append(mpd_cdf)
    # asdf

    mpd_all.append(mpd_true)
    mpd_all_cdf.append(mpd_true_cdf)

    script_n = os.path.basename(__file__).split('.')[0]

    importlib.reload(my_plot); my_plot.my_relplot(
        mpd_all_cdf, y='count', x='num_partners', hue='kind',
        kind='line',
        hue_order=['Data', 'Shuffle'],
        context='paper',
        xlim=(0, 13),
        ylim=[0, 1.01],
        height=4,
        aspect=1.33,
        y_axis_label='Cumulative Distribution',
        x_axis_label='# of other similar patterns',
        save_filename=f'{script_n}_{mf_dist_margin}_{n_random}_cdf.svg',
        )

    importlib.reload(my_plot); my_plot.my_catplot(
        mpd_all, y='count', x='num_partners', hue='kind',
        kind='bar',
        context='paper',
        xlim=(None, 13),
        height=4,
        aspect=1.33,
        y_axis_label='Cumulative Distribution',
        x_axis_label='# of other similar patterns',
        save_filename=f'{script_n}_{mf_dist_margin}_{n_random}_bar.svg',
        )


