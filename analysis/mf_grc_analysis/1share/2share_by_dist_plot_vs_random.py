
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


norm = 555

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
        mpd_trial = mpd_trial.to_pdf('count', cumulative=True, fixed_scale=norm)
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

label = 'Anatomically\nCorrect Shuffle'
labels.append(label)
data = compress_pickle.load('2share_by_dist_random_correct_10.gz')
mpd_data[label] = calc_mpd(data, kind=label)

# label = 'Shuffle without Varying Dendrite Lengths'
# labels.append(label)
# data = compress_pickle.load('2share_by_dist_random_fixed_length_10.gz')
# mpd_data[label] = calc_mpd(data, kind=label)

# label = 'Shuffle without varying GrC Degree'
# labels.append(label)
# data = compress_pickle.load('2share_by_dist_random_constant_grc_degree_10.gz')
# mpd_data[label] = calc_mpd(data, kind=label)

# label = 'Shuffle without MF Overrepresentation'
# labels.append(label)
# data = compress_pickle.load('2share_by_dist_random_no_gt_mf_degree_10.gz')
# mpd_data[label] = calc_mpd(data, kind=label)


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
    aspect=.8,
    y_axis_label='Normalized Cumulative Pairs',
    x_axis_label='Soma Distance (um)',
    save_filename=f'{script_n}_count_cumulative.svg',
    )


