import collections
import sys
import json
import random
from jsmin import jsmin
from io import StringIO
import numpy as np
import copy
import compress_pickle
import time

import daisy
daisy.block.Block.BLOCK_ID_ADD_ONE_FIX = True


sys.path.insert(0, '/n/groups/htem/Segmentation/shared-dev/cb2_segmentation/segway.graph.tmn7')
sys.path.insert(0, '/n/groups/htem/Segmentation/tmn7/segway.dahlia')

import segway.dahlia.connected_segment_server
from segway.graph.synapse_graph import SynapseGraph
# from segway.graph.plot_adj_mat import plot_adj_mat

sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')
from tools import *
# from mesh_tool import *

# config_f = "config_pc_pfs_200911.json"
config_f = "../config_grc_201207.json"
with open(config_f) as js_file:
    minified = jsmin(js_file.read())
    config = json.load(StringIO(minified))

overwrite = False
if len(sys.argv) == 2 and sys.argv[1] == "--overwrite":
    overwrite = True

graph = SynapseGraph(config_f, overwrite=overwrite)
g = graph.g


# get proofread'ed fragments and ignore them in this pass
grcs_with_complete_axons_f = '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/completed_neurons_with_axons_201208'
grcs_with_complete_axons = set()
with open(grcs_with_complete_axons_f) as fin:
    for line in fin:
        try:
            id = line.strip()
            grcs_with_complete_axons.add(id)
        except:
            pass


# for each grc, find the most distal synapse and get the xy location
# also delta x in relation to the grc soma
soma_locs = {}
synapse_locs = {}
for grc in grcs_with_complete_axons:
    if grc not in g.nodes:
        print(f'{grc} not in graph!')
        continue
    soma_xyz = get_node_pos(g, grc)
    syn_locs = graph.get_synapse_locs(grc, dir='pre', in_nm=True)
    most_distal = soma_xyz
    for loc in syn_locs:
        # lower y means higher in the upper half of the EM volume
        if loc[1] < most_distal[1]:
            most_distal = loc
    if most_distal != soma_xyz:
        soma_locs[grc] = soma_xyz
        synapse_locs[grc] = most_distal


import compress_pickle
compress_pickle.dump((
    soma_locs,
    synapse_locs
    ), "ascending_axon_locs.gz")



