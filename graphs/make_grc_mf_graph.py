import argparse
import segway.graph.synapse_graph

config = {
    "input_method": "user_list",
    "input_config_files": ["configs/grc_list.json", "configs/mf_list.json"],
    "synapse_db" : "mongodb://10.117.28.139:27017/cb2_v4_synapse_pred_cleft_setup01_synapsedb_area_210429",
    "neuron_db" : "mongodb://10.117.28.139:27017/neurondb_cb2_v4",
    "voxel_size_xyz" : [4, 4, 40],
    "syn_score_threshold": 20,
}

ap = argparse.ArgumentParser()
ap.add_argument("--overwrite", type=int, default=0)
# ap.add_argument("--threshold", type=int, default=20)
ap.add_argument("--output", type=str, default="synapsegraph_mf_grc.npz")
args = ap.parse_args()

graph_db = segway.graph.synapse_graph.SynapseGraph(args.output,
                                                   config=config,
                                                   overwrite=args.overwrite,
                                                   )

# G = graph_db.make_networkx_graph()
