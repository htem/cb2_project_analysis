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

def weight_fn(syn):
    z_len = syn['z_length'] - 40
    major_axis_length = syn['major_axis_length'] * .9
    diameter = max(z_len, major_axis_length)
    diameter = int(diameter/40+.5)
    diameter *= 40
    r = diameter/2
    area = math.pi*r*r
    return area

weightdb.load_syn_db('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/gen_db/mf_grc/gen_201229_setup01_syndb_threshold_20_coalesced.gz',
    weight_fn=weight_fn)

mpd = MyPlotData()
mpd_raw = MyPlotData()
hist = defaultdict(int)
weights_db = weightdb.get_weights()

for neuron, pc_weights in weights_db.items():
    # print(n)
    for pc, weights in pc_weights.items():
        if len(weights) <= 1:
            continue
        connection_weight = 0
        for w in weights:
            connection_weight += w
        connection_weight /= 1000000
        hist[connection_weight] += 1
        mpd_raw.add_data_point(
            cleft_area=connection_weight)

# print(hist)
for k in sorted([k for k in hist.keys()]):
    print(f'{k}: {hist[k]}')
    mpd.add_data_point(
        count=hist[k],
        cleft_area=k)

# mpd = mpd.to_pdf('count', cumulative=False)
mpd_cdf = mpd.to_pdf('count', cumulative=False)


# importlib.reload(my_plot); my_plot.my_relplot(
#     mpd,
#     x="cleft_area",
#     # xlim=[None, .6],
#     y="count",
#     kind='line',
#     context='paper',
#     height=4,
#     y_axis_label='Count',
#     x_axis_label='Area (um^2)',
#     show=True,
#     save_filename=f'{script_n}.svg',
#     )

# importlib.reload(my_plot); my_plot.my_relplot(
#     mpd,
#     x="cleft_area",
#     xlim=[None, .6],
#     y="count",
#     s=75,
#     kind='scatter',
#     context='paper',
#     height=4,
#     y_axis_label='Count',
#     x_axis_label='Area (um^2)',
#     show=True,
#     save_filename=f'{script_n}_scatter.svg',
#     )

# importlib.reload(my_plot); my_plot.my_displot(
#     mpd_raw,
#     x="cleft_area",
#     # y="count",
#     xlim=[None, .6],
#     # s=100,
#     kind='ecdf',
#     context='paper',
#     height=4,
#     y_axis_label='Distribution',
#     x_axis_label='Area (um^2)',
#     show=True,
#     save_filename=f'{script_n}_ecdf.svg',
#     )

importlib.reload(my_plot); my_plot.my_displot(
    mpd_raw,
    x="cleft_area",
    # y="count",
    # xlim=[None, .6],
    # s=100,
    kind='hist',
    # binwidth=.0399,
    # kde=True,
    # kde_kws={'bw_adjust': 3.5},
    context='paper',
    height=4,
    y_axis_label='Distribution',
    x_axis_label='Area (um^2)',
    show=True,
    save_filename=f'{script_n}_hist.svg',
    )

# df = mpd_raw.to_dataframe()
# width = .04
# bins = np.linspace(df["cleft_area"].min() - width/2, df["cleft_area"].max() + width/2, int(df["cleft_area"].max()/width)+1)
# print(bins)
# # df["A"].hist(bins=bins)

# importlib.reload(my_plot); my_plot.my_displot(
#     mpd_raw,
#     x="cleft_area",
#     # y="count",
#     xlim=[None, .6],
#     # s=100,
#     kind='hist',
#     # binwidth=.0399,
#     bins=bins,
#     kde=True,
#     kde_kws={'bw_adjust': 3.5},
#     context='paper',
#     height=4,
#     y_axis_label='Distribution',
#     x_axis_label='Area (um^2)',
#     show=True,
#     save_filename=f'{script_n}_hist.svg',
#     )

# importlib.reload(my_plot); my_plot.my_displot(
#     mpd_raw,
#     x="cleft_area",
#     # y="count",
#     xlim=[None, .6],
#     # s=100,
#     kind='kde',
#     # discrete=True,
#     bw_adjust=4,
#     # kde=True,
#     context='paper',
#     height=4,
#     y_axis_label='Distribution',
#     x_axis_label='Area (um^2)',
#     show=True,
#     save_filename=f'{script_n}_kde.svg',
#     )



