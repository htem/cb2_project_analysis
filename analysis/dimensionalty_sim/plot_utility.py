# import random
# import copy
# import logging
# import sys
# import os
# import sys
# import importlib
import numpy as np
from collections import defaultdict
# sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')
# from tools_pattern import get_eucledean_dist
# import compress_pickle
# import my_plot
# from my_plot import MyPlotData, my_box_plot
# import seaborn as sns

import my_plot
from my_plot import MyPlotData, my_box_plot

def hist2list(hist):
    ret = []
    for k, v in hist.items():
        ret.extend([k]*v)
    return ret

def get_average(sums):
    if sums is None:
        return 0
    return sum(sums)/len(sums)

def get_average_delta(hist_sum, ref_sum, minus_hist_sum=None, minus_hist_sum2=None):
    return get_average(hist_sum) - ref_sum - get_average(minus_hist_sum) - get_average(minus_hist_sum2)

def get_signal_variance(hist_sum):
    return np.std(hist_sum, ddof=1)

def get_low_signal_val(hist_sum, pct=.025):
    return sorted(hist_sum, reverse=False)[int(len(hist_sum)*pct)]

def get_signal_variance_width(hist_sum):
    hist_sum = sorted(hist_sum)
    return hist_sum[int(.95*len(hist_sum))] - hist_sum[int(.05*len(hist_sum))]

def get_signal_loss(hist_sum, ref_sum):
    hist_sum = sorted(hist_sum)
    return ref_sum- hist_sum[int(.5*len(hist_sum))]

def average_value_feat(res, key, feature_size):
    averages = []
    for ress in res:
        avg = get_average(ress[feature_size][key])
        averages.append(avg)
    return sum(averages)/len(averages)


def build_mpd(feature_size, models, db):
    mpd = MyPlotData()
    # ress_ref = db['global_random'][.1][0][0]
    # resss_ref2 = db['global_random'][.1][0]
    for model_name in models:
        resss = db[model_name]
        for noise in resss:
            ress_tries = resss[noise][0]  # get the first element in tuple
            avg_random_sum = average_value_feat(ress_tries, 'random_sum_hist', feature_size)
            for n_try, ress in enumerate(ress_tries):
                # if n_try >= len(resss_ref2):
                #     print(n_try)
                #     continue
                # ress_ref2 = resss_ref2[n_try]
    #             for noise in ress:
                res = ress[feature_size]
                # res_ref2 = ress_ref2[feature_size]
                sums = hist2list(res['sum_hist'])
                random_sums = hist2list(res['random_sum_hist'])
                masked_random_sums = hist2list(res['random_masked_sum_hist'])
                noisy_ref_sums = hist2list(res['noisy_ref_sum_hist'])
                random_delta = get_average_delta(random_sums, res['ref_sum0'])
                avg_random_delta = avg_random_sum - res['ref_sum0']
                signal_delta = get_average_delta(sums, res['ref_sum0'])
                if len(sums) == 0:
                    print(res)
                assert len(sums)
                mpd.add_data_point(
                    model=model_name,
                    avg_delta=get_average_delta(sums, res['ref_sum0']),
                    avg_random_delta=get_average_delta(random_sums, res['ref_sum0']),
                    avg_masked_random_delta=get_average_delta(masked_random_sums, res['ref_sum0']),
                    avg_delta_minus_random=get_average_delta(sums, res['ref_sum0'], random_sums),
                    avg_delta_minus_masked_random=get_average_delta(sums, res['ref_sum0'], masked_random_sums),
                    # avg_delta_minus_masked_random2=get_average_delta(sums, 0, masked_random_sums, noisy_ref_sums),
                    avg_delta_minus_noisy_ref=get_average_delta(sums, 0, noisy_ref_sums),
                    avg_delta_div_random=signal_delta/avg_random_delta,
                    
                    ref_delta=res['ref_delta'],
                    variance=get_signal_variance(sums),
                    low_signal=get_low_signal_val(sums),
                    variance_width=get_signal_variance_width(sums),
                    signal_loss=get_signal_loss(sums, res['ref_sum1']),
                    feature_size=feature_size,
                    noise=noise,
                    )
    return mpd


def get_average_value_by_feature_size(db, model_name, key, noise):
    ret = {}
    vals = defaultdict(list)
    resss = db[model_name]
    ress_tries = resss[noise][0]  # get the first element in tuple
    for n_try, ress in enumerate(ress_tries):
        for feature_size in ress:
            res = ress[feature_size]
            sums = hist2list(res['sum_hist'])
            avg_delta = get_average_delta(sums, res['ref_sum0'])
            random_sums = hist2list(res['random_sum_hist'])
            avg_delta_minus_random = get_average_delta(sums, res['ref_sum0'], random_sums)
            if key == 'avg_delta_minus_random':
                val = avg_delta_minus_random
            else:
                assert False
            vals[feature_size].append(val)
    for feature_size in vals:
        ret[feature_size] = sum(vals[feature_size])/len(vals[feature_size])
    return ret

def build_mpd_by_noise(noise, models, db, feature_sizes=None, compare_averages=None, name_map=None):
    mpd = MyPlotData()
    # ress_ref = db['global_random'][.1][0][0]
    # resss_ref2 = db['global_random'][.1][0]
    if compare_averages is None:
        compare_averages = defaultdict(lambda: 1)
    for model_name in models:
        resss = db[model_name]
        # for noise in resss:
        ress_tries = resss[noise][0]  # get the first element in tuple
        # avg_random_sum = average_value_feat(ress_tries, 'random_sum_hist', feature_size)
        for n_try, ress in enumerate(ress_tries):
            # if n_try >= len(resss_ref2):
            #     print(n_try)
            #     continue
            # ress_ref2 = resss_ref2[n_try]
#             for noise in ress:
            if feature_sizes is None:
                feature_sizes = ress.keys()
            for feature_size in feature_sizes:
                res = ress[feature_size]
                # res_ref2 = ress_ref2[feature_size]
                sums = hist2list(res['sum_hist'])
                random_sums = hist2list(res['random_sum_hist'])
                masked_random_sums = hist2list(res['random_masked_sum_hist'])
                noisy_ref_sums = hist2list(res['noisy_ref_sum_hist'])
                random_delta = get_average_delta(random_sums, res['ref_sum0'])
                # avg_random_delta = avg_random_sum - res['ref_sum0']
                signal_delta = get_average_delta(sums, res['ref_sum0'])
                if len(sums) == 0:
                    print(res)
                assert len(sums)

                mpd.add_data_point(
                    model=name_map[model_name] if name_map else model_name,
                    avg_delta=get_average_delta(sums, res['ref_sum0']),
                    avg_random_delta=get_average_delta(random_sums, res['ref_sum0']),
                    avg_masked_random_delta=get_average_delta(masked_random_sums, res['ref_sum0']),
                    avg_delta_minus_random=get_average_delta(sums, res['ref_sum0'], random_sums),
                    avg_delta_minus_random_norm=get_average_delta(sums, res['ref_sum0'], random_sums)/compare_averages[feature_size],
                    avg_delta_minus_masked_random=get_average_delta(sums, res['ref_sum0'], masked_random_sums),
                    # avg_delta_minus_masked_random2=get_average_delta(sums, 0, masked_random_sums, noisy_ref_sums),
                    avg_delta_minus_noisy_ref=get_average_delta(sums, 0, noisy_ref_sums),
                    # avg_delta_div_random=signal_delta/avg_random_delta,
                    
                    ref_delta=res['ref_delta'],
                    variance=get_signal_variance(sums),
                    low_signal=get_low_signal_val(sums),
                    variance_width=get_signal_variance_width(sums),
                    signal_loss=get_signal_loss(sums, res['ref_sum1']),
                    feature_size=feature_size,
                    feature_size_pct=feature_size*100,
                    noise=noise,
                    )
    return mpd

