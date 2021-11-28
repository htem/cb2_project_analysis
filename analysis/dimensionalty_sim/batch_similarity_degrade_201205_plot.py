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

db = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/dimensionality_sim/batch_similarity_degrade_201205.gz')
db = db[0]

mpd = MyPlotData()

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
        # 'naive_random_15_4',
        # 'naive_random_16_4',
        'naive_random_17_4',
        # 'naive_random_18_4',
        # 'naive_random_19_4',
        # 'naive_random_20_4',
        'classic_random',
        ]:
    ress = db[model_name]
    ress = ress[0]
    for degrade_level, res in ress.items():
        if int(degrade_level*100) % 3:
            continue
        hamming_distance_norm = res['hamming_distance']/res['num_grcs']
        hamming_distance_norm_max = res['hamming_distance']/res['max_hamming_distance']
        mpd.add_data_point(
            model=get_plot_name(model_name),
            # mf_dim=res[0]['mf_dim'],
            degrade_level=degrade_level,
            grc_dim=res['grc_dim'],
            pct_grc=res['pct_grc']/100,
            pct_mfs=res['pct_mfs']/100,
            # pct_mf_dim=res['pct_mf_dim']/100,
            num_grcs=res['num_grcs'],
            num_mfs=res['num_mfs'],
            voi=res['voi'],
            binary_similarity=res['binary_similarity'],
            hamming_distance=res['hamming_distance'],
            hamming_distance_norm=hamming_distance_norm,
            hamming_distance_norm_max=hamming_distance_norm_max,
            )

importlib.reload(my_plot); my_plot.my_relplot(
    mpd,
    x='degrade_level',
    y='hamming_distance_norm',
    hue='model',
    context='paper',
    height=4,
    aspect=1,
    y_axis_label='Norm. Hamming Distance',
    x_axis_label='Degrade Level (%)',
    save_filename=f'{script_n}.svg',
    show=show,
    )


importlib.reload(my_plot); my_plot.my_relplot(
    mpd,
    x='degrade_level',
    y='hamming_distance_norm_max',
    hue='model',
    context='paper',
    height=4,
    aspect=1,
    y_axis_label='Hamming Distance\n / Max',
    x_axis_label='Degrade Level (%)',
    save_filename=f'{script_n}_norm_max.svg',
    show=show,
    )
