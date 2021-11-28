import random
import copy
import logging
import sys

from run_tests_201204 import *

import os
import sys
import importlib
from collections import defaultdict
sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')
from tools_pattern import get_eucledean_dist
import compress_pickle
import my_plot
from my_plot import MyPlotData, my_box_plot

script_n = os.path.basename(__file__).split('.')[0]

show=False
if '--show' in sys.argv:
    show = True

db = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/dimensionality_sim/batch_dim_201205_db.gz')
db = db[0]

mpd = MyPlotData()
# for model_name, res in db.items():
# for model_name in ['data', 'classic_random',
#         # 'naive_random_17_1',
#         # 'naive_random_17_2',
#         # 'naive_random_17_3',
#         # 'naive_random_17_4',
#         # 'naive_random_17_5',
#         # 'naive_random_15_1',
#         # 'naive_random_16_1',
#         'naive_random_17_1',
#         # 'naive_random_18_1',
#         # 'naive_random_19_1',
#         # 'naive_random_20_1',
#         ]:

def get_plot_name(model_name):
    if 'data' in model_name:
        return 'Data'
    if 'classic_random' in model_name:
        return 'Classic Random'
    if 'naive_random' in model_name:
        return 'Local Random'
        # return model_name
    if 'data' in model_name:
        return 'Data'
    if 'data' in model_name:
        return 'Data'


for model_name in ['data',
        'naive_random_17_1',
        'classic_random',
        # 'naive_random_17_1',
        # 'naive_random_17_1',
        # 'naive_random_17_2',
        # 'naive_random_17_3',
        # 'naive_random_17_4',
        # 'naive_random_17_5',
        # 'naive_random_15_1',
        # 'naive_random_16_1',
        # 'naive_random_17_1',
        # 'naive_random_18_1',
        # 'naive_random_19_1',
        # 'naive_random_20_1',
        ]:
    res = db[model_name]
    mpd.add_data_point(
        model=get_plot_name(model_name),
        mf_dim=res[0]['mf_dim'],
        grc_dim=res[0]['grc_dim'],
        pct_grc=res[0]['pct_grc']/100,
        pct_mfs=res[0]['pct_mfs']/100,
        pct_mf_dim=res[0]['pct_mf_dim']/100,
        num_grcs=res[0]['num_grcs'],
        num_mfs=res[0]['num_mfs'],
        )

importlib.reload(my_plot); my_plot.my_catplot(
    mpd,
    x='model',
    y='pct_grc',
    context='paper',
    height=4,
    aspect=1.33,
    y_axis_label='Dim / # of GrCs',
    x_axis_label='Normalized Dimensionality',
    save_filename=f'{script_n}_norm_dim_grc.svg',
    show=show,
    )

importlib.reload(my_plot); my_plot.my_catplot(
    mpd,
    x='model',
    y='pct_mfs',
    context='paper',
    height=4,
    aspect=1.33,
    y_axis_label='Dim / MF dim',
    x_axis_label='Normalized Dimensionality',
    save_filename=f'{script_n}_pct_mfs.svg',
    show=show,
    )

importlib.reload(my_plot); my_plot.my_catplot(
    mpd,
    x='model',
    y='pct_mf_dim',
    context='paper',
    height=4,
    aspect=1.33,
    y_axis_label='Dim / MF dim',
    x_axis_label='Normalized Dimensionality',
    save_filename=f'{script_n}_pct_mf_dim.svg',
    show=show,
    )

