import collections
import sys
import json
import random
from jsmin import jsmin
from io import StringIO
import numpy as np
import copy
import os
import compress_pickle

import daisy
daisy.block.Block.BLOCK_ID_ADD_ONE_FIX = True

import segway.dahlia.db_server

from mesh_tool import *

overwrite = False
overwrite_db = False
if '--overwrite' in sys.argv:
    overwrite = True
if '--overwrite_db' in sys.argv:
    overwrite_db = True

neuron_db = segway.dahlia.db_server.NeuronDBServer(
            db_name='neurondb_cb2_v4',
            host="mongodb://10.117.28.139:27017/",
            )

config_f = "/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/config_pfs_214029_setup01.json"
with open(config_f) as js_file:
    minified = jsmin(js_file.read())
    config = json.load(StringIO(minified))

# pfs_list = config['input_neurons_list']
# pfs_list = neuron_db.find_neuron_filtered({'cell_type': 'pc'})
pfs_list = neuron_db.find_neuron({'cell_type': 'pf'})
print(pfs_list)
fname = f'mesh_db_210708/db.gz'
os.makedirs('mesh_db_210708', exist_ok=True)

res = {}

for pf_name in pfs_list:

    vert_by_box = collections.defaultdict(list)
    # vert_by_neuron = collections.defaultdict(set)
    vert_to_neuron = {}
    vert_count = 0

    pf_obj = neuron_db.get_neuron(pf_name, with_children=False)
    print(f'Processing {pf_name}...')
    mesh_ids = pf_obj.segments
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
                vert_to_neuron[v] = pf_name
        except IOError as e:
            pass
    # for box in vert_by_box:
    #     print(f'{box}: {len(vert_by_box[box])}')
    print(f'vert_count: {vert_count}')
    res[pf_name] = (vert_by_box, vert_to_neuron)

print(f"Writing to {fname}...")
compress_pickle.dump(
    res,
    fname)



# graph = SynapseGraph(config_f, overwrite=overwrite)
# g = graph.g
# random.seed(0)


