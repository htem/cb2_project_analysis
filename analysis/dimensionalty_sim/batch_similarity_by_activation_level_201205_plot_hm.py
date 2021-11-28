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

db = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/dimensionality_sim/batch_similarity_by_activation_level_201210_4096.gz')
db = db[0]


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


for activation_level in db['data'][0]:
    mpd = MyPlotData()
    for model_name in ['data',
            # 'naive_random_15_4',
            # 'naive_random_16_4',
            'naive_random_17_4',
            # 'naive_random_18_4',
            # 'naive_random_19_4',
            # 'naive_random_20_4',
            'classic_random',
            ]:
        resss = db[model_name]
        resss = resss[0]
        # for activation_level, ress in resss.items():
        # activation_level = .3
        # activation_level = .4
        # activation_level = .05
        # activation_level = .15
        # activation_level = .2
        # activation_level = .7
        ress = resss[activation_level]
        for noise_level, res in ress.items():
            # if int(activation_level*100) % 3:
            #     continue
            # hamming_distance_norm = res['hamming_distance']/res['num_grcs']
            # hamming_distance_norm_max = res['hamming_distance']/res['max_hamming_distance']
            hamming_distance_norm = res['hamming_distance']/res['num_grcs']
            mpd.add_data_point(
                model=get_plot_name(model_name),
                # mf_dim=res[0]['mf_dim'],
                activation_level=activation_level,
                noise_level=noise_level*100,
                grc_dim=res['grc_dim'],
                pct_grc=res['pct_grc']/100,
                pct_mfs=res['pct_mfs']/100,
                # pct_mf_dim=res['pct_mf_dim']/100,
                num_grcs=res['num_grcs'],
                num_mfs=res['num_mfs'],
                voi=res['voi'],
                grc_pop_corr=res['grc_pop_corr'],
                binary_similarity=res['binary_similarity'],
                hamming_distance=res['hamming_distance'],
                normalized_mse=res['normalized_mse'],
                hamming_distance_norm=hamming_distance_norm,
                # hamming_distance_norm_max=hamming_distance_norm_max,
                )

        # importlib.reload(my_plot); my_plot.my_relplot(
        #     mpd,
        #     x='noise_level',
        #     y='hamming_distance_norm',
        #     hue='model',
        #     context='paper',
        #     height=4,
        #     aspect=1,
        #     y_axis_label='Hamming Distance',
        #     x_axis_label='Noise Level (%)',
        #     save_filename=f'batch_similarity_by_activation_level_201205_plot_dim/{script_n}_{activation_level*100}_hm.svg',
        #     show=show,
        #     )

    importlib.reload(my_plot); my_plot.my_relplot(
        mpd,
        x='noise_level',
        y='normalized_mse',
        hue='model',
        context='paper',
        height=4,
        aspect=1,
        y_axis_label='Norm. MSD',
        x_axis_label='MF Input Noise (%)',
        save_filename=f'batch_similarity_by_activation_level_201205_plot_norm_msd/{script_n}_{activation_level*100}_norm_msd.svg',
        show=show,
        )
