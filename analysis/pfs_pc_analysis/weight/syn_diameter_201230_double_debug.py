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

import compress_pickle
syndb = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/gen_db/pfs/gen_201224_setup01_syndb_threshold_10_coalesced.gz')

pairs = []
for grc in syndb:
    for pc in syndb[grc]:
        syns = syndb[grc][pc]
        if len(syns) == 2:
            # print(f'{grc} to {pc}')
            # print([s['syn_loc0'] for s in syns])
            # print([s['major_axis_length'] for s in syns])
            # print()
            pairs.append(((grc, pc), tuple([s['syn_loc0'] for s in syns]), tuple([s['major_axis_length'] for s in syns])))

pairs.sort(key=lambda x: x[2])
pairs.sort(key=lambda x: x[2], reverse=True)

def to_ng(coord):
    return [
        int(coord[0]/4),
        int(coord[1]/4),
        int(coord[2]/40),
        ]

for p in pairs:
    print(p[0])
    print([to_ng(c) for c in p[1]])
    print(p[2])
    print()


