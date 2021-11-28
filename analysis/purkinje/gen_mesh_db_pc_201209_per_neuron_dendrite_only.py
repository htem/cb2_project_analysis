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

overwrite = False
if '--overwrite' in sys.argv:
    overwrite = True

neuron_db = segway.dahlia.db_server.NeuronDBServer(
            db_name='neurondb_cb2_v4',
            host="mongodb://10.117.28.139:27017/")

config_f = "/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/db_pc_200911.json"
with open(config_f) as js_file:
    minified = jsmin(js_file.read())
    config = json.load(StringIO(minified))

# pc_list = config['input_neurons_list']
pc_list = neuron_db.find_neuron_filtered({'cell_type': 'pc'})

print(pc_list)

for pc_name in pc_list:

    fname = f'mesh_db_dendrites_201209/mesh_db_pc.{pc_name}.gz'
    if not overwrite and os.path.exists(fname):
        print(f'Skipping {fname}...')
        continue

    vert_by_box = collections.defaultdict(list)
    # vert_by_neuron = collections.defaultdict(set)
    vert_to_neuron = {}
    vert_count = 0

    pc = neuron_db.get_neuron(pc_name, with_children=False)
    pc_objects = pc.children
    print(pc_objects)

    pc_objects = {}
    for obj in pc.children:
        pc_objects[obj] = pc.segments_by_children[obj]
    pc_objects[pc_name] = pc.segments

    for obj in pc_objects:
        print(f'Processing {obj}...')
        if 'axon' in obj or 'soma' in obj:
            print(f"Skipping {obj}")
            continue
        mesh_ids = pc_objects[obj]
        # for each mesh, get the vertices coords, which should already be in nm
        for mesh_id in mesh_ids:
            boxid = getBoxId(mesh_id)
            try:
                vertices = getMeshVertices(mesh_id)
                # vert_set |= vertices
                vert_count += len(vertices)
                for v in vertices:
                    vert_by_box[boxid].append(v)
                    # vert_by_neuron[obj].add(v)
                    vert_to_neuron[v] = obj
            except IOError as e:
                pass

    # for box in vert_by_box:
    #     print(f'{box}: {len(vert_by_box[box])}')

    print(f'vert_count: {vert_count}')


    print(f"Writing to {fname}...")

    compress_pickle.dump((
        vert_by_box,
        # vert_by_neuron
        vert_to_neuron
        ), fname)



# graph = SynapseGraph(config_f, overwrite=overwrite)
# g = graph.g
# random.seed(0)


