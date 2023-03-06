import sys
import argparse
import pickle
import gzip

import networkx as nx
import numpy as np
import compress_pickle

from funlib.geometry import Coordinate

from segway.graph.synapse_graph import SynapseGraph

'''
`syndb` is an internal format used to represent connectivity between two layers. This script converts syndb to NetworkX.

Ex: 
- `python convert_syndb_to_networkx_mf_grc.py ../analysis/gen_db/mf_grc/gen_210518_setup01_v2_syndb_threshold_20_coalesced.gz graph_mf_grc_synapse_210518_coalesced.gz --synapsegraph ./synapsegraph_mf_grc_230301.npz`
- `python convert_syndb_to_networkx_mf_grc.py ../analysis/gen_db/mf_grc/gen_210518_setup01_v2_syndb_threshold_20.gz graph_mf_grc_synapse_210518_all.gz --synapsegraph ./synapsegraph_mf_grc_230301.npz`
'''

def load_syndb(in_fname):
    return compress_pickle.load(in_fname)

def convert(in_fname, synapsegraph_file=None):

    neurondb = None
    if synapsegraph_file is not None:
        # we need synapsegraph to fill in neuron attributes
        neurondb = SynapseGraph.from_file(synapsegraph_file).neuron_db_data

    syndb = load_syndb(in_fname)
    G = nx.MultiDiGraph()
    for grc_id, synapses in syndb.items():
        if grc_id not in G:
            grc_attrs = {}
            if neurondb is not None:
                graph_attrs = neurondb[grc_id]
                grc_attrs = {
                    'cell_type': graph_attrs['cell_type'],
                    'xyz': (graph_attrs['soma_loc']['x'],
                            graph_attrs['soma_loc']['y'],
                            graph_attrs['soma_loc']['z'],),
                    'tags': graph_attrs['tags'],
                }
            G.add_node(grc_id, **grc_attrs)
        for mf_id, syns in synapses.items():
            if mf_id not in G:
                mf_attrs = {}
                if neurondb is not None:
                    graph_attrs = neurondb[mf_id]
                    mf_attrs = {
                        'cell_type': graph_attrs['cell_type'],
                        'tags': graph_attrs['tags'],
                    }
                G.add_node(mf_id, **mf_attrs)
            for syn in syns:
                for attr in syn:
                    if isinstance(syn[attr], Coordinate):
                        # convert `Coordinate`s to simple tuples
                        syn[attr] = tuple(syn[attr])
                G.add_edge(mf_id, grc_id, **syn)
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
    ap.add_argument("--synapsegraph", type=str, help='')
    config = ap.parse_args()
    arg_config = vars(ap.parse_args())
    for k, v in arg_config.items():
        globals()[k] = v

    G = convert(in_fname, synapsegraph)
    save(G, out_fname)


