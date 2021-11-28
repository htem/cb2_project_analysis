import collections
import sys
import json
import random
from jsmin import jsmin
from io import StringIO
import numpy as np


sys.path.insert(0, '/n/groups/htem/Segmentation/shared-dev/cb2_segmentation/segway.graph.tmn7')
from segway.graph.synapse_graph import SynapseGraph
# from segway.graph.plot_adj_mat import plot_adj_mat

def get_eucledean_dist(a, b):
    return np.linalg.norm(
        (a[0]-b[0], a[1]-b[1], a[2]-b[2]))

def get_partner(g, neuron, partner_type, cell_type=None, condition_fn=None, synapse_min_count=None):

    if condition_fn is None:
        def condition_fn(a, b): return True

    if partner_type == 'presyn':
        out = [x for x in g.predecessors(neuron)]
        def partner_map(a, b):
            return b, a
    else:
        out = [x for x in g.successors(neuron)]
        def partner_map(a, b):
            return a, b

    if cell_type:
        out = [
            x for x in out if g.nodes[x]['cell_type'] == cell_type
            ]

    if condition_fn:
        out = [
            x for x in out if condition_fn(*partner_map(neuron, x))
            ]

    if synapse_min_count:
        def condition_fn(a, b):
            if len(g.synapse_locs[(a, b)]) >= synapse_min_count:
                return True
            return False
        out = [
            x for x in out
            if g.nodes[x]['cell_type'] == cell_type and condition_fn(*partner_map(neuron, x))
            ]

    return set(out)

def get_postsyn(g, neuron, cell_type=None, condition_fn=None, synapse_min_count=None):
    return get_partner(g, neuron, 'postsyn', cell_type, condition_fn, synapse_min_count)

def get_presyn(g, neuron, cell_type=None, condition_fn=None, synapse_min_count=None):
    return get_partner(g, neuron, 'presyn', cell_type, condition_fn, synapse_min_count)

def get_eucledean_dist(a, b):
    return np.linalg.norm(
        (a[0]-b[0], a[1]-b[1], a[2]-b[2]))

def init():

    config_f = sys.argv[1]
    with open(config_f) as js_file:
        minified = jsmin(js_file.read())
        config = json.load(StringIO(minified))

    overwrite = False
    if len(sys.argv) == 3 and sys.argv[2] == "--overwrite":
        overwrite = True

    synapse_graph = SynapseGraph(config_f, overwrite=overwrite)
    g = synapse_graph.g
    random.seed(0)

    return config, synapse_graph, g

def get_node_pos(g, neuron, in_nm=False):
    # note: division by 4 because of bug with cb2
    loc = (
        int(g.nodes[neuron]['x']/4),
        int(g.nodes[neuron]['y']/4),
        int(g.nodes[neuron]['z']),
        )
    if in_nm:
        loc = tuple([k*f for k, f in zip(loc, (4, 4, 40))])
    return loc


def to_ng_coord(coord):
    return (
        int(coord[0]/4),
        int(coord[1]/4),
        int(coord[2]/40),
        )

def print_presyn_synapse_loc(g, syn_dict, neuron, cell_type=None):
    neuron_list = get_presyn(g, neuron, cell_type)
    for presyn_neuron in neuron_list:
        if (presyn_neuron, neuron) in syn_dict and len(syn_dict[(presyn_neuron, neuron)]):
            print(f'{presyn_neuron}: {neuron}')
        syn_locs = syn_dict[(presyn_neuron, neuron)]
        for loc in syn_locs:
            print(to_ng_coord(loc))

