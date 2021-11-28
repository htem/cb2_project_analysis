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

# seg_file = "/n/f810/htem/Segmentation/cb2_v4/output.zarr"
# seg = daisy.open_ds(seg_file, 'volumes/super_1x2x2_segmentation_0.500_mipmap/s2')

# config_f = "/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/config_mf_grc_201229_setup01.json"
config_f = "/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/config_mf_grc_210429_setup01_v2.json"
with open(config_f) as js_file:
    minified = jsmin(js_file.read())
    config = json.load(StringIO(minified))

ap = argparse.ArgumentParser()
ap.add_argument("--overwrite", type=bool, default=False)
ap.add_argument("--threshold", type=int, default=20)
config = ap.parse_args()

graph = segway.graph.synapse_graph.SynapseGraph(config_f, overwrite=config.overwrite,
    syn_score_threshold=None)

syn_db = defaultdict(lambda: defaultdict(list))

for nid in graph.g.nodes:
    if 'mf' not in nid:
        continue
    res = graph.get_partners(
        nid, 'postsyn',
        # synapse_min_count=2,
        partner_type='grc',
        # neuron_subtype='axon',
        # filter_list=mfs,
        # return_synapse_locs=True,
        return_synapses=True,
        score=config.threshold,
        )
    if len(res):
        for pair in res:
            mf_id, grc_id = pair
            for syn in res[pair]:
                syn['sf_pre'] = int(syn['sf_pre'])
                syn['sf_post'] = int(syn['sf_post'])
                syn_db[grc_id][mf_id].append(syn)

compress_pickle.dump((
    dict(syn_db)
    ), f"{script_n}_syndb_threshold_{config.threshold}.gz")
