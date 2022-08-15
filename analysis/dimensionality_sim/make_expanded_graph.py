import copy
from collections import defaultdict
import compress_pickle
import random
import sys

sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')
# from global_random_model import GlobalRandomModel
import tools_mf_graph

z_replication = 3

full_graph_path = '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/gen_db/mf_grc/input_graph_210611_grc_mf_limited_100_2_xlim_360000_600000.gz'
grc_dendrite_dist = None
mf_size_dist = None

# get distributions from the full graph
full_input_graph = compress_pickle.load(full_graph_path)
grc_dendrite_dist, mf_size_dist = tools_mf_graph.get_distributions(
    full_input_graph, z_margin_pct=40, x_margin_pct=25)

if __name__ == "__main__":
    print("\ninner grc_dendrites:")
    print(len(grc_dendrite_dist))
    tools_mf_graph.print_box_plot(grc_dendrite_dist)
    print("\ninner mf_size_dist:")
    print(len(mf_size_dist))
    tools_mf_graph.print_box_plot(mf_size_dist)
    print()

    # tools_mf_graph.count_redundancy_graph(g)

    g = tools_mf_graph.replicate_expand(
        full_input_graph,
        grc_dendrite_dist=grc_dendrite_dist,
        mf_size_dist=mf_size_dist,
        xyz_lim=((420000, 540000), (None, None), (2800+2000, 46800-2000)),  # 120 x Y x 40
        z_replication=z_replication,
        seed=0,
        )

    # tools_mf_graph.count_redundancy_graph(g)

    # grc_dendrite_dist, mf_size_dist = tools_mf_graph.get_distributions(
    #     g, z_margin_pct=0, x_margin_pct=0)
    # print("\ngrc_dendrites:")
    # print(len(grc_dendrite_dist))
    # tools_mf_graph.print_box_plot(grc_dendrite_dist)
    # print("\nmf_size_dist:")
    # print(len(mf_size_dist))
    # tools_mf_graph.print_box_plot(mf_size_dist)
    # print()

    compress_pickle.dump(g, "/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/gen_db/mf_grc/"
                            "input_graph_210611_expanded_graph_220722.gz")

