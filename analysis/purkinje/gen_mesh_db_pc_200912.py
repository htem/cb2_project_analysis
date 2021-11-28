import collections
import sys
import json
import random
from jsmin import jsmin
from io import StringIO
import numpy as np
import copy
import compress_pickle

import daisy
daisy.block.Block.BLOCK_ID_ADD_ONE_FIX = True

import segway.dahlia.db_server

from mesh_tool import *

# neuron_db = segway.dahlia.db_server.NeuronDBServer(
#             db_name='production_200107_cb2_v4_synapse_setup09_synapsedb',
#             host="mongodb://10.117.28.250:27018/")



neuron_db = segway.dahlia.db_server.NeuronDBServer(
            db_name='neurondb_cb2_v4',
            host="mongodb://10.117.28.139:27017/")

# sys.path.insert(0, '/n/groups/htem/Segmentation/shared-dev/cb2_segmentation/segway.graph.tmn7')
# sys.path.insert(0, '/n/groups/htem/Segmentation/tmn7/segway.dahlia')
# from segway.graph.synapse_graph import SynapseGraph
# from segway.graph.plot_adj_mat import plot_adj_mat

# sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')
# from tools import *

# config_f = "config_pc_pfs_200911.json"
# config_f = "/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/db_pc_200911_test.json"
config_f = "/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/db_pc_200911.json"
with open(config_f) as js_file:
    minified = jsmin(js_file.read())
    config = json.load(StringIO(minified))

pc_list = config['input_neurons_list']

print(pc_list)

vert_by_box = collections.defaultdict(list)
vert_to_neuron = {}
vert_count = 0

for pc_name in pc_list:
    pc = neuron_db.get_neuron(pc_name, with_children=False)
    pc_objects = pc.children
    print(pc_objects)

    pc_objects = {}
    for obj in pc.children:
        pc_objects[obj] = pc.segments_by_children[obj]
    pc_objects[pc_name] = pc.segments

    for obj in pc_objects:
        print(f'Processing {obj}...')
        mesh_ids = pc_objects[obj]
        # for each mesh, get the vertices coords, which should already be in nm
        for mesh_id in mesh_ids:
            boxid = getBoxId(mesh_id)
            try:
                vertices = getMeshVertices(mesh_id)
                vert_count += len(vertices)
                for v in vertices:
                    vert_by_box[boxid].append(v)
                    vert_to_neuron[v] = obj
            except IOError as e:
                pass

for box in vert_by_box:
    print(f'{box}: {len(vert_by_box[box])}')

print(f'vert_count: {vert_count}')

print("Writing to mesh_db_pc.gz...")

compress_pickle.dump((
    vert_by_box, vert_to_neuron
    ), "mesh_db_pc.gz")



# graph = SynapseGraph(config_f, overwrite=overwrite)
# g = graph.g
# random.seed(0)


