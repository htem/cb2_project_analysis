import logging
logging.basicConfig(level=logging.INFO)

from processing_cb2 import get_graph

'''
Example of getting all synapses from specific neurons.
'''

neurondb_file = '../../database/data/neurons.db'
synapsedb_file = '../../database/synapse_database/data/synapses.db'
superfragmentdb_file = '../../database/synapse_database/data/superfragments.db'

# by default, the library will only graph between these neurons
neuron_list = ['grc_1651', 'mf_181']

# we can make it so that it will also grab all neurons that receive syns from these neurons
neuron_list_all_from = ['mf_181']
# and all neurons that make synapses to these neurons
neuron_list_all_to = ['grc_1651']
# note that unknown superfragments will have raw IDs as nodes in the graph

# set to print debug info
# logging.getLogger('segway.graph.synapse_graph').setLevel(logging.DEBUG)

graph_db = get_graph(neuron_list=neuron_list,
                     neurondb_file=neurondb_file,
                     synapsedb_file=synapsedb_file,
                     superfragmentdb_file=superfragmentdb_file,
                     synapse_score_threshold=20,
                     neuron_list_all_from=neuron_list_all_from,
                     neuron_list_all_to=neuron_list_all_to,
                   )

graph_db.save('networkx_example.npz')
