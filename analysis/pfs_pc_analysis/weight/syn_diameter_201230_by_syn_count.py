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
        # if len(weights) != 3:
            # continue
        for w in weights:
            w /= 1000
            # w = math.log10(w)
            mpd_raw.add_data_point(
                cleft_area=w,
                syn_count=(len(weights)),
                )
            w = int(w*100)
            w /= 100
            hist[w] += 1

# print(hist)
for k in sorted([k for k in hist.keys()]):
    print(f'{k}: {hist[k]}')
    mpd.add_data_point(
        count=hist[k],
        cleft_area=k)

# mpd = mpd.to_pdf('count', cumulative=False)
mpd_cdf = mpd.to_pdf('count', cumulative=False)


importlib.reload(my_plot); my_plot.my_catplot(
    mpd_raw,
    y="cleft_area",
    x="syn_count",
    # ylim=[.30, .70],
    xlim=[None, 3.5],
    context='paper',
    kind='box',
    # add_swarm=True,
    height=4,
    width=4,
    y_axis_label='Cleft Width (um)',
    x_axis_label='# of synapses per connection',
    save_filename=f'{script_n}_box.svg',
    show=True,
    )


importlib.reload(my_plot); my_plot.my_catplot(
    mpd_raw,
    y="cleft_area",
    x="syn_count",
    # ylim=[.30, .70],
    xlim=[None, 3.5],
    context='paper',
    kind='violin',
    bw=.25,
    cut=0,
    # add_swarm=True,
    height=4,
    width=4,
    y_axis_label='Cleft Width (um)',
    x_axis_label='# of synapses per connection',
    save_filename=f'{script_n}_violin.svg',
    show=True,
    )
