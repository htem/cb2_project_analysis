import sys
import argparse
import pickle
import gzip

import networkx as nx
import numpy as np
import compress_pickle

from funlib.geometry import Coordinate


# from segway.graph.synapse_graph import SynapseGraph
from segway.mdseg.database.neuron_db import NeuronDBServer

'''
`syndb` is an internal format used to represent connectivity between two layers. This script converts syndb to NetworkX.

Ex: 
```
python convert_syndb_to_networkx_grc_pc.py \
    ../analysis/gen_db/grc_axons/gen_210429_setup01_syndb_threshold_10_coalesced.gz \
    graph_grc_pc_synapse_210429_coalesced.gz \
    --neurondb_url mongodb://10.117.28.139:27017/neurondb_cb2_v4
```
```
python convert_syndb_to_networkx_grc_pc.py \
    ../analysis/gen_db/grc_axons/gen_210429_setup01_syndb_threshold_10.gz \
    graph_grc_pc_synapse_210429_all.gz \
    --neurondb_url mongodb://10.117.28.139:27017/neurondb_cb2_v4
```
```
python convert_syndb_to_networkx_grc_pc.py \
    ../analysis/gen_db/pfs/gen_210429_setup01_syndb_threshold_10_coalesced.gz \
    graph_pfs_pc_synapse_210429_coalesced.gz \
    --neurondb_url mongodb://10.117.28.139:27017/neurondb_cb2_v4
```
```
python convert_syndb_to_networkx_grc_pc.py \
    ../analysis/gen_db/pfs/gen_210429_setup01_syndb_threshold_10.gz \
    graph_pfs_pc_synapse_210429_all.gz \
    --neurondb_url mongodb://10.117.28.139:27017/neurondb_cb2_v4
```


'''

def get_attrs(nid, neurondb):
    attrs = {}
    if neurondb is not None:
        try:
            db_attrs = neurondb.get_neuron(nid).to_json()
            attrs = {
                'cell_type': db_attrs['cell_type'],
                'xyz': (db_attrs['soma_loc']['x'],
                        db_attrs['soma_loc']['y'],
                        db_attrs['soma_loc']['z'],),
                'tags': db_attrs['tags'],
            }
        except RuntimeError as e:
            print(e)
    return attrs

def load_syndb(in_fname):
    return compress_pickle.load(in_fname)

def convert(in_fname, neurondb_url=None, bad_neurons=[]):

    neurondb = None
    if neurondb_url is not None:
        # we need synapsegraph to fill in neuron attributes
        neurondb = NeuronDBServer(neurondb_url)

    syndb = load_syndb(in_fname)
    G = nx.MultiDiGraph()
    for grc_id, synapses in syndb.items():
        if grc_id in bad_neurons:
            print(f'Skipping {grc_id}')
            continue  # skip bad constructions
        if grc_id not in G:
            grc_attrs = get_attrs(grc_id, neurondb)
            G.add_node(grc_id, **grc_attrs)
        for pc_id, syns in synapses.items():
            if pc_id not in G:
                pc_attrs = get_attrs(pc_id, neurondb)
                G.add_node(pc_id, **pc_attrs)
            added_syn_locs = set()
            for syn in syns:
                for attr in syn:
                    if isinstance(syn[attr], Coordinate):
                        # convert `Coordinate`s to simple tuples
                        syn[attr] = tuple(syn[attr])
                if syn['syn_loc'] in added_syn_locs:
                    continue  # skip duplicated syns
                added_syn_locs.add(syn['syn_loc'])
                G.add_edge(grc_id, pc_id, **syn)

    return G

def save(graph, networkx_path):
    with gzip.open(networkx_path, 'wb') as f:
        pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)

def load(networkx_path):
    with gzip.open(networkx_path, 'rb') as f:
        G = pickle.load(f)
    return G

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("in_fname", type=str, help='Input path to convert')
    ap.add_argument("out_fname", type=str, help='')
    ap.add_argument("--neurondb_url", type=str, help='URL to the proofreading database')
    config = ap.parse_args()
    arg_config = vars(ap.parse_args())
    for k, v in arg_config.items():
        globals()[k] = v

    bad_neurons = set()
    # if in_fname.startswith('gen_210429'):
    if 'gen_210429' in in_fname:
        # spot checking for bad neurons on 230304
        bad_neurons.add('grc_667')  # axon is actually from grc_451, unfixed 230304
        bad_neurons.add('pf_190')  # axon is actually from pf_229, unfixed 230304
        bad_neurons.add('pf_1865')  # axon shared with pf_1826, fixed 230304
        bad_neurons.add('pf_2198')  # bad construction, unfixed 230304
        bad_neurons.add('pf_228')  # bad construction, overlaps pf_223, unfixed 230304
        bad_neurons.add('pf_2782')  # replicate of pf_2148, undeleted 230304
        bad_neurons.add('pf_3008')  # bad construction, overlaps pf_2992, unfixed 230304
        bad_neurons.add('pf_3435')  # bad construction, overlaps pf_3444, unfixed 230304
        bad_neurons.add('pf_777')  # bad construction, overlaps pf_745, unfixed 230304
        bad_neurons.add('pf_983')  # bad construction, overlaps pf_960, unfixed 230304
        bad_neurons.add('pf_301')  # bad construction, entirely overlaps grc_425, undeleted 230304

    G = convert(in_fname, neurondb_url=neurondb_url, bad_neurons=bad_neurons)
    save(G, out_fname)


