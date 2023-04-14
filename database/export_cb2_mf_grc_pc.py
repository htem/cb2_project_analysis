import json
from jsmin import jsmin

import sys
sys.path.insert(0, '/n/groups/htem/cb2/repos/segway.utils')
from segway.utils.mongodb_to_sqlite import check_keys, export_neurondb, add_reverse_segment_map

data_keys = ['neuron_name', 'name_prefix', 'segments', 'blacklist_segments', 'cell_type', 'cell_subtype', 'tags', 'location_tags', 'location_tags_zyx', 'notes', 'uncertain_xyz', 'mergers_xyz', 'location_notes', 'location_notes_zyx', 'version', 'annotator', 'reviewed', 'finished', 'soma_loc', 'parent_segment']
missing_ok = ['uncertain_xyz', 'mergers_xyz']

indices = ['neuron_name', 'name_prefix', 'annotator', 'cell_type', 'parent_segment']
unique_indices = ['neuron_name']

input_list_key = 'neuron_name'
input_list = []

def get_list(json_file):
    with open(json_file) as f:
        return json.loads(jsmin(f.read()))['input_neurons_list']

input_list += get_list('../graphs/configs/grc_list.json')
input_list += get_list('../graphs/configs/mf_list.json')
input_list += get_list('../graphs/configs/pc_list.json')

export_neurondb(mongo_host="mongodb://10.117.28.139:27017/",  # dwalin
       mongo_db_name="neurondb_cb2_v4",
       mongo_collection_name='neurons',
       out_file='./data/neurons.db',
       data_keys=data_keys,
       data_keys_missing_ok=missing_ok,
       indices=indices,
       unique_indices=unique_indices,
       input_list=input_list,
       input_list_key=input_list_key,
       )

add_reverse_segment_map('./data/neurons.db', 'neurons', 'segments')
