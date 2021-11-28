
import random

def shuffle(
        graph, model,
        num_claws=4,
        mf_dist_margin=4000,
        constant_dendrite_length=15000,
        seed=0,
        ):
    random.seed(seed)
    if model == 'constant_grc':
        graph.randomize_graph_by_grc(
            mf_dist_margin=mf_dist_margin,
            single_connection_per_pair=True,
            constant_grc_degree=num_claws,
            preserve_mf_degree=True,
        )
    if model == 'constant_length':
        graph.randomize_graph_by_grc(
            mf_dist_margin=mf_dist_margin,
            single_connection_per_pair=True,
            constant_dendrite_length=constant_dendrite_length,
            preserve_mf_degree=True,
        )
    if model == 'classic_random':
        graph.randomize_graph(
            random_model=True,
            constant_grc_degree=num_claws,
            )
    if model == 'no_same_mf':
        graph.randomize_graph_by_grc(
            mf_dist_margin=mf_dist_margin,
            single_connection_per_pair=True,
            )
    if model == 'naive_random':
        graph.randomize_graph_by_grc(
            single_connection_per_pair=True,
            constant_grc_degree=4,
            constant_dendrite_length=constant_dendrite_length,
            always_pick_closest_rosette=True,
            )
    if model == 'naive_random2':
        graph.randomize_graph_by_grc(
            single_connection_per_pair=True,
            constant_grc_degree=4,
            constant_dendrite_length=constant_dendrite_length,
            # always_pick_closest_rosette=True,
            mf_dist_margin=1000,
            )
    if model == 'naive_random3':
        graph.randomize_graph_by_grc(
            single_connection_per_pair=True,
            constant_grc_degree=4,
            constant_dendrite_length=constant_dendrite_length,
            # always_pick_closest_rosette=True,
            mf_dist_margin=4000,
            )
    if model == 'naive_random_17_2':
        graph.randomize_graph_by_grc(
            single_connection_per_pair=True,
            constant_grc_degree=4,
            constant_dendrite_length=17000,
            # always_pick_closest_rosette=True,
            mf_dist_margin=2000,
            )
    if model == 'naive_random_17_3':
        graph.randomize_graph_by_grc(
            single_connection_per_pair=True,
            constant_grc_degree=4,
            constant_dendrite_length=17000,
            # always_pick_closest_rosette=True,
            mf_dist_margin=3000,
            )
    if model == 'naive_random_17_4':
        graph.randomize_graph_by_grc(
            single_connection_per_pair=True,
            constant_grc_degree=4,
            constant_dendrite_length=17000,
            # always_pick_closest_rosette=True,
            mf_dist_margin=4000,
            )
    if model == 'naive_random_15_1':
        graph.randomize_graph_by_grc(
            single_connection_per_pair=True,
            constant_grc_degree=4,
            constant_dendrite_length=15000,
            # always_pick_closest_rosette=True,
            mf_dist_margin=1000,
            )
    if model == 'naive_random_15_2':
        graph.randomize_graph_by_grc(
            single_connection_per_pair=True,
            constant_grc_degree=4,
            constant_dendrite_length=15000,
            # always_pick_closest_rosette=True,
            mf_dist_margin=2000,
            )
    if model == 'naive_random_15_3':
        graph.randomize_graph_by_grc(
            single_connection_per_pair=True,
            constant_grc_degree=4,
            constant_dendrite_length=15000,
            # always_pick_closest_rosette=True,
            mf_dist_margin=3000,
            )
    if model == 'naive_random_15_4':
        graph.randomize_graph_by_grc(
            single_connection_per_pair=True,
            constant_grc_degree=4,
            constant_dendrite_length=15000,
            # always_pick_closest_rosette=True,
            mf_dist_margin=4000,
            )
    if model == 'expanded_random_15':
        graph.randomize_graph_by_grc(
            single_connection_per_pair=True,
            constant_grc_degree=4,
            # constant_dendrite_length=constant_dendrite_length,
            # always_pick_closest_rosette=True,
            dendrite_range=(0, 15000),
            )
    if model == 'expanded_random_30':
        graph.randomize_graph_by_grc(
            single_connection_per_pair=True,
            constant_grc_degree=4,
            # constant_dendrite_length=constant_dendrite_length,
            # always_pick_closest_rosette=True,
            dendrite_range=(0, 30000),
            )
    if model == 'expanded_random_50':
        graph.randomize_graph_by_grc(
            single_connection_per_pair=True,
            constant_grc_degree=4,
            # constant_dendrite_length=constant_dendrite_length,
            # always_pick_closest_rosette=True,
            dendrite_range=(0, 50000),
            )
    return graph
