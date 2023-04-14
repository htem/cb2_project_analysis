
import segway.graph.synapse_graph
from segway.mdseg.database import NeuronDBServerSQLite, KeyValueDatabaseSQLite

'''
TODO 230414 (Tri):
- Add option to filter out synapses if the outgoing synapse is not from an "axon" sub-object
'''

def get_graph(neuron_list,
               neurondb_file,
               synapsedb_file,
               superfragmentdb_file,
               synapse_score_threshold=20,
               neuron_list_all_from=None,
               neuron_list_all_to=None,
    ):

    neuron_db = NeuronDBServerSQLite(neurondb_file)
    synapse_db = KeyValueDatabaseSQLite(synapsedb_file, collection_name='synapses')
    superfragment_db = KeyValueDatabaseSQLite(superfragmentdb_file, collection_name='superfragments')

    def add_node_attributes(item):
        '''Defines how to add additional attributes to nodes'''
        return {
            'cell_type': item.data['cell_type'],
            'tags': item.data['tags'],
            'xyz': (item.data['soma_loc']['x'],
                    item.data['soma_loc']['y'],
                    item.data['soma_loc']['z'],),
        }

    def synapse_score_fn(syn):
        '''Defines how to get the `score` from raw measurements.
        Here we return the "score" of the synaptic cleft but only if the area of the cleft is non zero (likely not FP)'''
        if 'area_erode0' not in syn['props'] or 'mesh_area' not in syn['props']['area_erode0']:
            return 0
        if syn['props']['area_erode0']['mesh_area'] == 0:
            return 0
        return syn.get('score', 0)

    # def synapse_score_threshold(syn, presyn_nid, postsyn_nid, G):
    #     '''We can use this to filter out grc->mf connections'''

    def add_edge_attributes(syn):
        syn_area = syn['props']['area_erode0']['mesh_area']
        score = syn.get('score')
        return {
            'xyz': (syn['x']/4, syn['y']/4, syn['z']/40),
            'cleft_area': syn_area,
            'score': score,
        }

    graph_db = segway.graph.synapse_graph.SynapseGraph(neuron_db=neuron_db,
                                                       synapse_db=synapse_db,
                                                       superfragment_db=superfragment_db,
                                                       )

    graph_db.add_neurons(neuron_list,
                         add_node_attributes=add_node_attributes,
                         add_edge_attributes=add_edge_attributes,
                         synapse_score_fn=synapse_score_fn,
                         synapse_score_threshold=synapse_score_threshold,
                         neuron_list_all_from=neuron_list_all_from,
                         neuron_list_all_to=neuron_list_all_to,
                         )

    return graph_db

 


