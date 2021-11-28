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
import math
import os

script_n = os.path.basename(__file__).split('.')[0]


sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')
import my_plot
importlib.reload(my_plot)
from my_plot import MyPlotData

from weight_database import WeightDatabase
weightdb = WeightDatabase()

weightdb.load_syn_db('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/gen_db/pfs/gen_201224_setup01_syndb_threshold_10_coalesced.gz')

mpd = MyPlotData()
mpd_raw = MyPlotData()
hist = defaultdict(int)
weights_db = weightdb.get_weights()

for neuron, pc_weights in weights_db.items():
    # print(n)
    for pc, weights in pc_weights.items():
        mpd_raw.add_data_point(
            num_syns=len(weights))
        hist[len(weights)] += 1

# print(hist)
for k in sorted([k for k in hist.keys()]):
    print(f'{k}: {hist[k]}')
    mpd.add_data_point(
        count=hist[k],
        num_syns=k)

# mpd = mpd.to_pdf('count', cumulative=False)
mpd_cdf = mpd.to_pdf('count', cumulative=False)


# importlib.reload(my_plot); my_plot.my_relplot(
#     mpd,
#     x="num_syns",
#     y="count",
#     # xlim=[None, .6],
#     kind='line',
#     context='paper',
#     height=4,
#     y_axis_label='Count',
#     x_axis_label='Diameter (um)',
#     show=True,
#     save_filename=f'{script_n}.svg',
#     )

# importlib.reload(my_plot); my_plot.my_relplot(
#     mpd,
#     x="num_syns",
#     y="count",
#     # xlim=[None, .6],
#     s=75,
#     kind='scatter',
#     context='paper',
#     height=4,
#     y_axis_label='Count',
#     x_axis_label='Diameter (um)',
#     show=True,
#     save_filename=f'{script_n}_scatter.svg',
#     )

importlib.reload(my_plot); my_plot.my_displot(
    mpd_raw,
    x="num_syns",
    # y="count",
    # xlim=[None, 1.0],
    # s=100,
    kind='hist',
    # log_scale_x=True,
    # binwidth=.04,
    # kde=True,
    # kde_kws={'bw_adjust': 3.5},
    stat='probability',
    discrete=True,
    context='paper',
    height=4,
    y_axis_label='Frequency',
    x_axis_label='Synapse count per Connection',
    show=True,
    save_filename=f'{script_n}_hist.svg',
    )

importlib.reload(my_plot); my_plot.my_displot(
    mpd_raw,
    x="num_syns",
    # y="count",
    # xlim=[None, 1.0],
    # s=100,
    kind='hist',
    # log_scale_x=True,
    # binwidth=.04,
    # kde=True,
    # kde_kws={'bw_adjust': 3.5},
    stat='count',
    discrete=True,
    context='paper',
    height=4,
    y_axis_label='Count',
    x_axis_label='Synapse count per Connection',
    show=True,
    save_filename=f'{script_n}_count.svg',
    )


# df = mpd_raw.to_dataframe()
# width = .04
# bins = np.linspace(df["num_syns"].min() - width/2, df["num_syns"].max() + width/2, int(df["num_syns"].max()/width)+1)
# print(bins)
# # df["A"].hist(bins=bins)

# importlib.reload(my_plot); my_plot.my_displot(
#     mpd_raw,
#     x="num_syns",
#     # y="count",
#     # xlim=[None, .6],
#     # s=100,
#     kind='hist',
#     # binwidth=.0399,
#     bins=bins,
#     kde=True,
#     kde_kws={'bw_adjust': 3.5},
#     context='paper',
#     height=4,
#     y_axis_label='Distribution',
#     x_axis_label='Diameter (um)',
#     show=True,
#     save_filename=f'{script_n}_hist.svg',
#     )



# importlib.reload(my_plot); my_plot.my_catplot(
#     mpd,
#     x="num_syns",
#     y="count",
#     # xlim=[None, .6],
#     kind='bar',
#     context='paper',
#     height=4,
#     y_axis_label='Count',
#     x_axis_label='Diameter (um)',
#     show=True,
#     save_filename=f'{script_n}_scatter.svg',
#     )



