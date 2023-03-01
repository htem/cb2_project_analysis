import sys
import argparse
import pickle
import gzip

import networkx as nx
import numpy as np
import compress_pickle

sys.path.insert(0, '../analysis')
import tools_mf_graph

'''
`input_graph` is an internal format used to represent binary connectivity between two layers.
This script converts this format to a more standard networkX format

Ex: `python convert_input_graph_to_networkx.py ../analysis/gen_db/mf_grc/input_graph_210519_all.gz graph_mf_grc_binary_210519.gz`
'''

def convert(fname):

    G = nx.DiGraph()
    input_graph = compress_pickle.load(fname)
    bouton_counter = 0
    bouton_map = {}

    def bouton_loc_to_id(loc, mf_id):
        nonlocal bouton_counter, bouton_map
        if loc in bouton_map:
            return bouton_map[loc]
        bouton_map[loc] = f'{mf_id}__{bouton_counter}'
        bouton_counter += 1
        return bouton_map[loc]

    for grc_id in input_graph.grcs:
        grc_attrs = input_graph.grcs[grc_id]
        G.add_node(grc_id, **{
            'xyz': grc_attrs.soma_loc,
            'cell_type': 'grc',
        })
        for mf_id, bouton_loc in grc_attrs.edges:
            bouton_id = bouton_loc_to_id(bouton_loc, mf_id)
            G.add_node(bouton_id, **{
                'mf_id': mf_id,
                'xyz': bouton_loc,
                'cell_type': 'mf',
            })
            G.add_edge(bouton_id, grc_id)

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
    config = ap.parse_args()
    arg_config = vars(ap.parse_args())
    for k, v in arg_config.items():
        globals()[k] = v

    G = convert(in_fname)
    save(G, out_fname)
