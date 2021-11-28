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
import daisy
import compress_pickle
import os
import argparse

script_n = os.path.basename(__file__).split('.')[0]

sys.path.insert(0, '/n/groups/htem/Segmentation/shared-dev/cb2_segmentation/segway.graph.tmn7')
import segway.graph.synapse_graph

sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')
from tools import to_ng_coord

seg_file = "/n/f810/htem/Segmentation/cb2_v4/output.zarr"
seg = daisy.open_ds(seg_file, 'volumes/super_1x2x2_segmentation_0.500_mipmap/s2')

# config_f = "../../config_pf_200925.json"
config_f = "/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/config_grc_210429_setup01_v2.json"
with open(config_f) as js_file:
    minified = jsmin(js_file.read())
    config = json.load(StringIO(minified))

ap = argparse.ArgumentParser()
ap.add_argument("--overwrite", type=bool, default=False)
ap.add_argument("--threshold", type=int, default=10)
config = ap.parse_args()

graph = segway.graph.synapse_graph.SynapseGraph(config_f, overwrite=config.overwrite,
    syn_score_threshold=None)

def get_segment_id(seg, loc):
    loc = daisy.Coordinate((loc[2], loc[1], loc[0]))
    return int(seg[loc])

locs_to_sid_fname = f'{script_n}_locs_to_sid.gz'

grcs = []
grcs_locs = defaultdict(lambda: defaultdict(list))
locs_to_sid = {}
try:
    locs_to_sid = compress_pickle.load(locs_to_sid_fname)
except:
    pass
# locs_to_sid, _ = locs_to_sid

f = '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/completed_neurons_with_axons_210421'
with open(f) as fin:
    for line in fin:
        line = line.strip()
        if 'grc' in line:
            grcs.append(line)

for grc in grcs:
    print(f'Processing {grc}...')
    syns = graph.get_synapses(grc, type='pre', score=config.threshold)
    for syn in syns:
        post_loc = syn['post_loc']
        # print(post_loc)
        if post_loc in locs_to_sid:
            sid = locs_to_sid[post_loc]
        else:
            sid = get_segment_id(seg, post_loc)
            locs_to_sid[post_loc] = sid
        assert sid is not None
        nid = graph.neuron_db.find_neuron_with_segment_id(sid)
        if nid:
             if (('pc' in nid and 'pcl' not in nid) or
                    'purkinje' in nid):
                nid = nid.split('.')[0]
                syn['sf_pre'] = int(syn['sf_pre'])
                syn['sf_post'] = int(syn['sf_post'])
                grcs_locs[grc][nid].append(syn)

compress_pickle.dump((
    dict(grcs_locs)
    ), f"{script_n}_syndb_threshold_{config.threshold}.gz")

compress_pickle.dump((
    locs_to_sid
    ), locs_to_sid_fname)

