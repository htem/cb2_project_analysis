import itertools
import collections
from collections import defaultdict
import sys
import json
import random
# from jsmin import jsmin
# from io import StringIO
# import numpy as np
import copy
# import importlib
# from functools import partial
import math
# import os

sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')
import my_plot
from my_plot import MyPlotData

def round_weight(w, round_factor):
    if round_factor is None:
        return w
    w = int(w/round_factor+.5)*round_factor
    return w

def get_paired_weights(share_db, weights_db, grcs, n_shares,
        single_synapse=True,
        round_factor=40,
        avg_round_factor=20,
        ):

    syn_weights = [[], []]
    avg_data = []
    hist = defaultdict(int)
    mpd = MyPlotData()

    for n1, n2 in itertools.combinations(grcs, 2):
        if share_db[n1][n2] not in n_shares:
            continue
        common_pcs = set(weights_db[n1].keys()) & set(weights_db[n2].keys())
        for pc in common_pcs:
            w1 = copy.deepcopy(weights_db[n1][pc])
            w2 = copy.deepcopy(weights_db[n2][pc])
            # we will only process single synapse connections for simplicity
            if single_synapse and (len(w1) > 1 or len(w2) > 1):
                continue
            w1 = sum(w1)/len(w1)
            w1 = round_weight(w1, round_factor)
            w2 = sum(w2)/len(w2)
            w2 = round_weight(w2, round_factor)
            avg = (w1 + w2)/2
            avg = round_weight(avg, avg_round_factor)
            hist[avg] += 1
            syn_weights[0].append(w1)
            syn_weights[1].append(w2)
            mpd.add_data_point(
                avg_weight=avg,
            )
            avg_data.append(avg)

    return mpd, avg_data, syn_weights, hist

def get_paired_weights2(share_db, weights_db, grcs, n_shares,
        single_synapse=True,
        round_factor=40,
        avg_round_factor=20,
        combinations=False,
        max_combinations=None,
        ):
    '''This version does not average weights of connections'''
    syn_weights = [[], []]
    avg_data = []
    hist = defaultdict(int)
    mpd = MyPlotData()

    for n1, n2 in itertools.combinations(grcs, 2):
        if share_db[n1][n2] not in n_shares:
            continue
        common_pcs = set(weights_db[n1].keys()) & set(weights_db[n2].keys())
        for pc in common_pcs:
            weights1 = copy.deepcopy(weights_db[n1][pc])
            weights2 = copy.deepcopy(weights_db[n2][pc])
            if single_synapse and (len(weights1) > 1 or len(weights2) > 1):
                continue
            if combinations:
                combo = [c for c in itertools.product(weights1, weights2)]
                random.shuffle(combo)
                if max_combinations:
                    combo = combo[0:max_combinations-1]
                for w1, w2 in combo:
                    w1 = round_weight(w1, round_factor)
                    w2 = round_weight(w2, round_factor)
                    avg = (w1 + w2)/2
                    avg = round_weight(avg, avg_round_factor)
                    hist[avg] += 1
                    syn_weights[0].append(w1)
                    syn_weights[1].append(w2)
                    mpd.add_data_point(
                        avg_weight=avg,
                    )
                    avg_data.append(avg)
            else:
                random.shuffle(weights1)
                w1 = weights1[0]
                w1 = round_weight(w1, round_factor)
                random.shuffle(weights2)
                w2 = weights2[0]
                w2 = round_weight(w2, round_factor)
                avg = (w1 + w2)/2
                avg = round_weight(avg, avg_round_factor)
                hist[avg] += 1
                syn_weights[0].append(w1)
                syn_weights[1].append(w2)
                mpd.add_data_point(
                    avg_weight=avg,
                )
                avg_data.append(avg)

    return mpd, avg_data, syn_weights, hist

def compute_mf_share(mf_grc_db, grcs):
    share_db = defaultdict(lambda: defaultdict(int))
    for n1, n2 in itertools.permutations(grcs, 2):
        if n1 not in mf_grc_db or n2 not in mf_grc_db:
            continue
        i_set = set(mf_grc_db[n1].keys())
        j_set = set(mf_grc_db[n2].keys())
        common_mfs = i_set & j_set
        if len(common_mfs):
            share_db[n1][n2] = len(common_mfs)
    return share_db

def weight_fn(syn):
    z_len = syn['z_length'] - 40
    major_axis_length = syn['major_axis_length'] * .9
    diameter = max(z_len, major_axis_length)
    diameter = int(diameter/40+.5)
    diameter *= 40
    # if area:
    #     r = diameter/2
    #     return math.pi * r * r
    return diameter

def weight_fn_area(syn):
    z_len = syn['z_length'] - 40
    major_axis_length = syn['major_axis_length'] * .9
    diameter = max(z_len, major_axis_length)
    diameter = int(diameter/40+.5)
    diameter *= 40
    r = diameter/2
    return math.pi * r * r

def hist_to_mpd(hist):
    mpd = MyPlotData()
    for k in sorted(hist.keys()):
        mpd.add_data_point(
            avg_weight=k,
            count=hist[k],
        )
    return mpd
