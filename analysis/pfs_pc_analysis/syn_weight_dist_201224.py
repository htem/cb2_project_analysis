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

sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')
import my_plot
importlib.reload(my_plot)
from my_plot import MyPlotData

from weight_database import WeightDatabase
weightdb = WeightDatabase()

# def weight_fn(syn):
#     diameter = syn['major_axis_length'] * .9
#     r = diameter/2
#     area = math.pi*r*r
#     return area

weightdb.load_syn_db('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/gen_db/pfs/gen_201224_setup01_syndb_threshold_10_coalesced.gz',
    weight_fn=None)

mpd = MyPlotData()
hist = defaultdict(int)
weights_db = weightdb.get_weights()

for neuron, pc_weights in weights_db.items():
    # print(n)
    for pc, weights in pc_weights.items():
        for w in weights:
            w /= 1000000
            mpd.add_data_point(
                cleft_area=w)
            # w = int((w/10000))
            # w *= 10000
            w = int(w*100)
            w /= 100
            hist[w] += 1

# print(hist)
for k in sorted([k for k in hist.keys()]):
    print(f'{k}: {hist[k]}')


importlib.reload(my_plot); my_plot.my_displot(
    mpd,
    x="cleft_area",
    # y="pct",
    # hue="type",
    # hue_order=['All', 'Per PC', 'Per pf'],
    # hue_order=['All', 'Per PC'],
    # y_lims=[.25, .75],
    xlim=[None, .6],
    kind='kde',
    context='paper',
    # kind='violin',
    # font_scale=1.5,
    height=4,
    y_axis_label='Normalized Frequency',
    x_axis_label='Cleft Area (um^2)',
    show=True,
    save_filename='syn_weight_dist_201224.svg',
    )



