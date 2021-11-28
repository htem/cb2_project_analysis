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
# from tools import *
# from mesh_tool import *


# get proofread'ed fragments and ignore them in this pass
grcs_with_complete_axons_f = '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/completed_neurons_with_axons_210421'
grcs_with_complete_axons = set()
with open(grcs_with_complete_axons_f) as fin:
    for line in fin:
        try:
            id = line.strip()
            grcs_with_complete_axons.add(id)
        except:
            pass
grcs_with_complete_axons.remove('')


# get proofread'ed fragments and ignore them in this pass
proofread_fnames = [
    '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/purkinje/grc_with_axons_orphan_segments_201207_combined',
    '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/purkinje/grc_with_axons_orphan_segments',
    '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/purkinje/grc_with_axons_orphan_segments_201223',
    '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/purkinje/grc_with_axons_orphan_segments_201223_setup01',
    '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/purkinje/grc_with_axons_orphan_segments_210111_setup01'
]
previously_proofread_fragments = set()
for proofread_segments_f in proofread_fnames:
    with open(proofread_segments_f) as fin:
        for line in fin:
            try:
                id = int(line)
                previously_proofread_fragments.add(id)
            except:
                pass




overwrite = False
if '--overwrite' in sys.argv:
    overwrite = True


hierarchy_lut_path = '/n/f810/htem/Segmentation/cb2_v4/output.zarr/luts/fragment_segment'
super_lut_pre = 'super_1x2x2_hist_quant_50'

super_segment_graph = segway.dahlia.connected_segment_server.ConnectedSegmentServer(
    hierarchy_lut_path=hierarchy_lut_path,
    super_lut_pre=super_lut_pre,
    voxel_size=(40, 8, 8),
    find_segment_block_size=(4000, 4096, 4096),
    super_block_size=(4000, 8192, 8192),
    fragments_block_size=(400, 2048, 2048),
    super_offset_hack=(2800, 0, 0),
    base_threshold=0.5,
    )

# (pc_vert_by_box, pc_vert_to_neuron) = compress_pickle.load("mesh_db_pc.gz")
# print(pc_vert_to_neuron)

def grow_segments(s):
    return super_segment_graph.find_connected_super_fragments(
                selected_super_fragments=s,
                no_grow_super_fragments=[],
                threshold=.5,
                z_only=False,
                )


# config_f = "config_grc_200916.json"
# # config_f = "config_grc_201207.json"

orphans_by_neuron = collections.defaultdict(list)
for config_f in [
        # 'config_grc_200916.json',
        # 'config_grc_201207.json',
        # '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/config_grc_201221_setup22_scale_0p7.json',
        # '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/config_grc_210111_setup01.json',
        '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/config_grc_210429_setup01.json',
        ]:
    graph = SynapseGraph(config_f,
        # overwrite=True,
        )
    g = graph.g
    # total_orphans = []
    # for n in ['grc_425', 'grc_573', 'grc_729', 'grc_312', 'grc_571', 'grc_1394']:
    # for n in g.nodes:
    for n in grcs_with_complete_axons:
        orphans = []
        for s in graph.orphaned_post_segments[n]:
            if int(s) in previously_proofread_fragments:
                continue
            if not graph.neuron_db.find_neuron_with_segment_id(s):
                # only add small orphans
                # check if s can be grown
                cc = grow_segments([s])
                if len(cc) == 1:
                    orphans.append(cc)
                    orphans_by_neuron[n].append(s)
                    # total_orphans_debug.append(s)
                elif len(cc) == 2:
                    # try again
                    cc = grow_segments(cc)
                    if len(cc) == 2:
                        orphans.append(cc)
                        orphans_by_neuron[n].extend(cc)
                        # total_orphans_debug.append(s)
                # orphans.append(s)
        if len(orphans):
            print(f'Neuron {n} has {len(orphans)} orphan segments: {orphans}')
            # total_orphans.extend(orphans)

# print(total_orphans_debug)

total_orphans = []
for n in orphans_by_neuron:
    for s in orphans_by_neuron[n]:
        total_orphans.append(s)

# for n in orphans_by_neuron:
#     print(n)
#     for s in orphans_by_neuron[n]:
#         print(s)
#     print()

# print(total_orphans_debug)
orphans_by_index = collections.defaultdict(set)
for fid in total_orphans:
    # for fid in fragment_ids:
        index = super_segment_graph.get_super_index(fid)
        # group multiple indices together
        # index = (0, int(index[1]/2), int(index[2]/2))
        index = [int(index[0]/16), int(index[1]/8), int(index[2]/8)]
        if index[0] >= 2:
            index[0] = 2
        index = tuple(index)
        orphans_by_index[index].add(fid)

for n in orphans_by_index:
    print(n)
    for s in orphans_by_index[n]:
        print(s)
    print()

asdf

with open('grc_with_axons_orphan_segments_210429_setup01', 'w') as fout:
    for n in sorted(orphans_by_index.keys()):
        fout.write(str(n) + '\n')
        for s in orphans_by_index[n]:
            fout.write(str(s) + '\n')
        fout.write('\n')
