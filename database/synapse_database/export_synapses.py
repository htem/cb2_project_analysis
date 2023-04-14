import json
from jsmin import jsmin

from segway.utils.mongodb_to_sqlite import check_keys, export_neurondb

mongo_host = "mongodb://10.117.28.139:27017"
mongo_db_name = "cb2_v4_synapse_pred_cleft_setup01_synapsedb_area_210429"
mongo_collection_name = 'synapses'

keys = check_keys(mongo_host=mongo_host,
                  mongo_db_name=mongo_db_name,
                  mongo_collection_name=mongo_collection_name)
print(keys)

data_keys = ['id', 'x', 'y', 'z', 'pre_z', 'pre_y', 'pre_x', 'post_z', 'post_y', 'post_x', 'score', 'id_superfrag_pre', 'id_superfrag_post', 'props']
missing_ok = []
indices = ['id']
unique_indices = ['id']

export_neurondb(mongo_host=mongo_host,
                mongo_db_name=mongo_db_name,
                mongo_collection_name=mongo_collection_name,
                out_file='./data/synapses.db',
                data_keys=data_keys,
                data_keys_missing_ok=missing_ok,
                indices=indices,
                unique_indices=unique_indices,
                # input_list=input_list,
                # input_list_key=input_list_key,
       )
