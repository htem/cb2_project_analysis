
import sys
sys.path.insert(0, '/n/groups/htem/Segmentation/tmn7/segway.dahlia')

import segway.dahlia.db_server

def make_neurondb_server():
    db = segway.dahlia.db_server.NeuronDBServer(
        db_name='neurondb_cb2_v4',
        # host='mongodb://10.117.28.250:27018/',  # balin
        host='mongodb://10.117.28.139:27017/',  # dwalin
        )
    db.connect()
    return db

neuron_db = make_neurondb_server()

pcs = neuron_db.find_neuron(cell_type='pc')

db = {}

for pcid in pcs:
    pc = neuron_db.get_neuron(pcid)
    db[pcid] = (
                    pc.soma_loc['x'],
                    pc.soma_loc['y'],
                    pc.soma_loc['z'],
                )

print(db)

import compress_pickle
compress_pickle.dump(
    db,
    'pc_soma_locs.gz')

