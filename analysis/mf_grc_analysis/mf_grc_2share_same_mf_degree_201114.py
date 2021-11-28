from collections import defaultdict
import sys
import random
import numpy as np
import compress_pickle
import importlib
import copy

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

random.seed(0)

mpd_all = MyPlotData()
mpd_all_cdf = MyPlotData()

# mf_dist_margin = 500
mf_dist_margin = 1000

n_random = 1000
n_random = 5

print("Generating random graphs")
# for i in range(1000):
for i in range(n_random):
    print(i)
    input_graph.randomize_graph(
        preserve_mf_out_degree=True,
        mf_dist_margin=mf_dist_margin,
        grc_local=True,
        )
    mf_share_data = compute_mf_share()
    mpd, mpd_cdf = count_shuffle_2share(mf_share_data)
    mpd_all.append(mpd)
    mpd_all_cdf.append(mpd_cdf)

mpd_all.append(mpd_true)
mpd_all_cdf.append(mpd_true_cdf)

importlib.reload(my_plot); my_plot.my_relplot(
    mpd_all_cdf, y='count', x='num_partners', hue='kind',
    kind='line',
    hue_order=['Data', 'Shuffle'],
    # y='ratio', y_lims=[.25, .75],
    context='paper',
    # kind='violin',
    # font_scale=1.5,
    # width=4,
    y_lims=[0, 1.01],
    height=4,
    aspect=1.33,
    y_axis_label='Cumulative Distribution',
    x_axis_label='# of other similar patterns',
    save_filename=f'mf_grc_2share_mf_degree_exp_grc_local_dist_{mf_dist_margin}_{n_random}_cdf_201114.svg',
    )

importlib.reload(my_plot); my_plot.my_catplot(
    mpd_all, y='count', x='num_partners', hue='kind',
    kind='bar',
    # kind='violin',
    # y='ratio', y_lims=[.25, .75],
    context='paper',
    # font_scale=1.5,
    # width=4,
    height=4,
    aspect=1.33,
    y_axis_label='Cumulative Distribution',
    x_axis_label='# of other similar patterns',
    save_filename=f'mf_grc_2share_mf_degree_exp_grc_local_dist_{mf_dist_margin}_{n_random}_bar_201114.svg',
    )
