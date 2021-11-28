
import os
import sys
import importlib
from collections import defaultdict
sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')

from tools_pattern import get_eucledean_dist
import compress_pickle

import my_plot
importlib.reload(my_plot)
from my_plot import MyPlotData, my_box_plot

script_n = os.path.basename(__file__).split('.')[0]

def calc_mpd(input_data, kind):
    mpd = MyPlotData()
    for n, distances in enumerate(input_data[0]):
        hist = defaultdict(int)
        mpd_trial = MyPlotData()
        for d in distances:
            # mpd.add_data_point(
            #     dist=d,
            #     kind='Naive Random',
            #     trial=n,
            #     )
            hist[int(d)] += 1
        # for d in sorted(hist.keys()):
        for d in range(max(hist.keys())):
            mpd_trial.add_data_point(
                dist=d,
                count=hist[d],
                kind=kind,
                trial=n,
            )
        mpd_trial = mpd_trial.to_pdf('count', cumulative=True, fixed_scale=1)
        mpd.append(mpd_trial)
    return mpd

mpd_data = {}
labels = []

import compress_pickle
label = 'Data'
labels.append(label)
data = compress_pickle.load('2share_by_dist_observed.gz')
mpd_data[label] = calc_mpd(data, kind=label)

label = 'Local Random'
labels.append(label)
data = compress_pickle.load('2share_by_dist_naive_10.gz')
mpd_data[label] = calc_mpd(data, kind=label)

label = 'Anatomically Correct Shuffle'
labels.append(label)
data = compress_pickle.load('2share_by_dist_random_correct_10.gz')
mpd_data[label] = calc_mpd(data, kind=label)

label = 'Shuffle without Varying Dendrite Lengths'
labels.append(label)
data = compress_pickle.load('2share_by_dist_random_fixed_length_10.gz')
mpd_data[label] = calc_mpd(data, kind=label)

label = 'Shuffle without varying GrC Degree'
labels.append(label)
data = compress_pickle.load('2share_by_dist_random_constant_grc_degree_10.gz')
mpd_data[label] = calc_mpd(data, kind=label)

label = 'Shuffle without MF Overrepresentation'
labels.append(label)
data = compress_pickle.load('2share_by_dist_random_no_gt_mf_degree_10.gz')
mpd_data[label] = calc_mpd(data, kind=label)


mpd_total = MyPlotData()
for label in labels:
    mpd_total.append(mpd_data[label])
# mpd_total.append(mpd_naive)
# mpd_total.append(mpd_random_correct)

importlib.reload(my_plot); my_plot.my_relplot(
    mpd_total,
    x='dist',
    y='count',
    hue='kind',
    kind='line',
    # stat="density",
    # kde=True,
    # hue_order=['Data', 'Shuffle'],
    context='paper',
    # xlim=(None, 100),
    # ylim=[0, 1.01],
    height=4,
    aspect=1.66,
    y_axis_label='Cumulative Count',
    x_axis_label='Soma Distance (um)',
    save_filename=f'{script_n}_count_cumulative.svg',
    )




asdf


script_n = os.path.basename(__file__).split('.')[0]

mpd_random = MyPlotData()
mf_dist_margin = 5000
n_random = 5
n_random = 20
for i in range(n_random):
    print(i)
    input_graph.randomize_graph_by_grc(
        mf_dist_margin=mf_dist_margin,
        single_connection_per_pair=True,
        # preserve_mf_degree=True,
        # approximate_in_degree=True,
        # local_lengths=True,
        )
    mpd = get_prob(input_graph, kind='Random', trial=i)
    mpd_random.append(mpd)

# script_n = os.path.basename(__file__).split('.')[0]
# import compress_pickle
# compress_pickle.dump((
#     mpd_data,
#     mpd_random,
#     ), f"{script_n}_data.gz")

mpd_total = MyPlotData()
mpd_total.append(mpd_data)
mpd_total.append(mpd_random)

asdf

# importlib.reload(my_plot); my_plot.my_relplot(
#     mpd_random, x='soma_dist',
#     # hue='kind',
#     kind='line',
#     # stat="density",
#     # kde=True,
#     # hue_order=['Data', 'Shuffle'],
#     context='paper',
#     # xlim=(0, 13),
#     # ylim=[0, 1.01],
#     height=4,
#     aspect=3,
#     y_axis_label='Count',
#     x_axis_label='Soma Distance (um)',
#     # save_filename=f'{script_n}_data_hist.svg',
#     )


mf_dist_margin = 5000
# prob of random graphs
input_graph.randomize_graph_by_grc(
    mf_dist_margin=mf_dist_margin,
    single_connection_per_pair=True,
    # preserve_mf_degree=True,
    # approximate_in_degree=True,
    # local_lengths=True,
    )

get_prob(input_graph)

mpd = MyPlotData()

for i in range(100):
    input_graph.randomize_graph_by_grc(
        mf_dist_margin=mf_dist_margin,
        single_connection_per_pair=True,
        preserve_mf_degree=True,
        # approximate_in_degree=True,
        # local_lengths=True,
        )
    n = get_prob(input_graph)
    mpd.add_data_point(
        n=n)

script_n = os.path.basename(__file__).split('.')[0]
import compress_pickle
compress_pickle.dump((
    mpd,
    # random_counts,
    ), f"{script_n}_data.gz")

importlib.reload(my_plot); my_plot.my_catplot(
    mpd, y='n',
    kind='box',
    # y='ratio', y_lims=[.25, .75],
    context='paper',
    # kind='violin',
    # font_scale=1.5,
    width=4,
    aspect=1,
    y_axis_label='Count',
    x_axis_label='GrCs per Mossy Fiber',
    # save_filename=f'{script_n}_bar.svg',
    )


asdf

# random_claw_lengths = []
# for i in range(10):
#     input_graph.randomize_graph()
#     random_claw_lengths.append

script_n = os.path.basename(__file__).split('.')[0]

importlib.reload(my_plot); my_plot.my_relplot(
    mpd, x='claw_length',
    y='prob',
    context='paper',
    height=4,
    aspect=1.33,
    y_axis_label='Probability',
    x_axis_label='Distance (um)',
    save_filename=f'{script_n}.svg',
    )
