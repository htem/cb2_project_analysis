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
    return diameter
    # r = diameter/2
    # area = math.pi*r*r
    # return area

weightdb.load_syn_db('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/gen_db/pfs/gen_201224_setup01_syndb_threshold_10_coalesced.gz',
    weight_fn=weight_fn)

mpd = MyPlotData()
mpd_raw = MyPlotData()
hist = defaultdict(int)
weights_db = weightdb.get_weights()

for neuron, pc_weights in weights_db.items():
    # print(n)
    for pc, weights in pc_weights.items():
        if len(weights) != 2:
            continue
        w1, w2 = weights
        w1 /= 1000
        w2 /= 1000
        mpd_raw.add_data_point(
            w1=w1,
            w2=w2,
            )

importlib.reload(my_plot); my_plot.my_jointplot(
    mpd_raw,
    x="w1",
    y="w2",
    # y_axis_label='Distance from PCL (um)',
    # x_axis_label='X offset from soma (um)',
    # xlim=[-200, 200],
    # kind=f'{kind}',
    kind='scatter',
    save_filename=f'{script_n}.svg',
    show=True,
    )


importlib.reload(my_plot); my_plot.my_jointplot(
    mpd_raw,
    x="w1",
    y="w2",
    ylim=[None, .9],
    xlim=[None, .9],
    # y_axis_label='Distance from PCL (um)',
    # x_axis_label='X offset from soma (um)',
    # xlim=[-200, 200],
    # kind=f'{kind}',
    kind='kde',
    save_filename=f'{script_n}_kde.svg',
    show=True,
    )



