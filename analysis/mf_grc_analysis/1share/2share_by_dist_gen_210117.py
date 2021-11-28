
import os
import sys
import importlib
from collections import defaultdict
sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')

from tools_pattern import get_eucledean_dist

'''Load data'''
import compress_pickle
fname = '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114.gz'
input_graph = compress_pickle.load(fname)

script_n = '2share_by_dist_210117'
n_samples = 200

import compress_pickle
input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114_restricted_z.gz')
# input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114.gz')
grcs = [k for k in input_graph.grcs.keys()]

# z_min = 15
# z_max = 35
z_min = 20000
z_max = 30000
x_min = 360000
x_max = 520000

# for mf_id, mf in input_graph.mfs.items():
#     rosette_capacities = mf.get_rosette_loc_capacity()
#     for rosette_loc, claw_count in rosette_capacities.items():
#         x, y, z = rosette_loc
#         if x < 360000 or x > 520000:
#             continue
#         if z < z_min*1000 or z > z_max*1000:
#             continue
#         mpd.add_data_point(
#             x=x/1000,
#             y=y/1000,
#             z=z/1000,
#             claw_count=claw_count,
#             )

def get_prob(in_graph):
    common_pair_dist = []
    n_pairs = 0
    n_common_pairs = 0
    processed = set()
    for i in in_graph.grcs:
        grc_i = in_graph.grcs[i]
        rosettes_i = set([mf[1] for mf in grc_i.edges])
        for j in in_graph.grcs:
            if i == j:
                continue
            if (i, j) in processed:
                continue
            processed.add((i, j))
            processed.add((j, i))
            grc_j = in_graph.grcs[j]
            common_rosettes = set([mf[1] for mf in grc_j.edges])
            common_rosettes = common_rosettes & rosettes_i
            n_pairs += 1
            if len(common_rosettes) >= 2:
                dist = get_eucledean_dist(grc_i.soma_loc, grc_j.soma_loc)
                dist = dist/1000
                n_common_pairs += 1
                common_pair_dist.append(dist)
    return common_pair_dist

print(f'Generating {script_n}_observed')

observed_data = [get_prob(input_graph)]
compress_pickle.dump((
    observed_data,
    ), f"{script_n}_observed.gz")


print(f'Generating naive_data3')
naive_data3 = []
for i in range(n_samples):
    input_graph.randomize_graph_by_grc(
        mf_dist_margin=4000,
        single_connection_per_pair=True,
        constant_grc_degree=4,
        constant_dendrite_length=15000,
        # always_pick_closest_rosette=True,
        # preserve_mf_degree=True,
        # approximate_in_degree=True,
        # local_lengths=True,
        )
    naive_data3.append(get_prob(input_graph))

compress_pickle.dump((
    naive_data3,
    ), f"{script_n}_naive3_{n_samples}.gz")



asdf


# naive
print(f'Generating naive_data2')
naive_data2 = []
for i in range(n_samples):
    input_graph.randomize_graph_by_grc(
        # mf_dist_margin=mf_dist_margin,
        single_connection_per_pair=True,
        constant_grc_degree=4,
        constant_dendrite_length=15000,
        # always_pick_closest_rosette=True,
        # preserve_mf_degree=True,
        # approximate_in_degree=True,
        # local_lengths=True,
        )
    naive_data2.append(get_prob(input_graph))


compress_pickle.dump((
    naive_data2,
    ), f"{script_n}_naive2_{n_samples}.gz")



# local_random ex30
print(f'Generating ex30')
ex30 = []
for i in range(n_samples):
    input_graph.randomize_graph_by_grc(
            single_connection_per_pair=True,
            constant_grc_degree=4,
            dendrite_range=(0, 30000),
        )
    ex30.append(get_prob(input_graph))

compress_pickle.dump((
    ex30,
    ), f"{script_n}_localex30_{n_samples}.gz")


# local_random ex50
print(f'Generating ex50')
ex50 = []
for i in range(n_samples):
    input_graph.randomize_graph_by_grc(
            single_connection_per_pair=True,
            constant_grc_degree=4,
            dendrite_range=(0, 50000),
        )
    ex50.append(get_prob(input_graph))

compress_pickle.dump((
    ex50,
    ), f"{script_n}_localex50_{n_samples}.gz")


# naive
print(f'Generating naive_data')
naive_data = []
for i in range(n_samples):
    input_graph.randomize_graph_by_grc(
        # mf_dist_margin=mf_dist_margin,
        single_connection_per_pair=True,
        constant_grc_degree=4,
        constant_dendrite_length=15000,
        always_pick_closest_rosette=True,
        # preserve_mf_degree=True,
        # approximate_in_degree=True,
        # local_lengths=True,
        )
    naive_data.append(get_prob(input_graph))


compress_pickle.dump((
    naive_data,
    ), f"{script_n}_naive_{n_samples}.gz")



# correct
# - gt dendrite length
# - gt grc degree
# - gt mf degree

print(f'Generating random_correct_data')
random_correct_data = []
for i in range(n_samples):
    input_graph.randomize_graph_by_grc(
        mf_dist_margin=4000,
        single_connection_per_pair=True,
        # constant_grc_degree=4,
        # constant_dendrite_length=15000,
        # always_pick_closest_rosette=True,
        preserve_mf_degree=True,
        # approximate_in_degree=True,
        # local_lengths=True,
        )
    random_correct_data.append(get_prob(input_graph))

compress_pickle.dump((
    random_correct_data,
    ), f"{script_n}_random_correct_{n_samples}.gz")


print(f'Generating random_fixed_length_data')

random_fixed_length_data = []
for i in range(n_samples):
    input_graph.randomize_graph_by_grc(
        mf_dist_margin=1000,
        single_connection_per_pair=True,
        # constant_grc_degree=4,
        constant_dendrite_length=15000,
        # always_pick_closest_rosette=True,
        preserve_mf_degree=True,
        # approximate_in_degree=True,
        # local_lengths=True,
        )
    random_fixed_length_data.append(get_prob(input_graph))

compress_pickle.dump((
    random_fixed_length_data,
    ), f"{script_n}_random_fixed_length_{n_samples}.gz")

print(f'Generating random_constant_grc_degree_data')

random_constant_grc_degree_data = []
for i in range(n_samples):
    input_graph.randomize_graph_by_grc(
        mf_dist_margin=4000,
        single_connection_per_pair=True,
        constant_grc_degree=4,
        # constant_dendrite_length=15000,
        # always_pick_closest_rosette=True,
        preserve_mf_degree=True,
        # approximate_in_degree=True,
        # local_lengths=True,
        )
    random_constant_grc_degree_data.append(get_prob(input_graph))

compress_pickle.dump((
    random_constant_grc_degree_data,
    ), f"{script_n}_random_constant_grc_degree_{n_samples}.gz")


print(f'Generating random_no_gt_mf_degree_data')

random_no_gt_mf_degree_data = []
for i in range(n_samples):
    input_graph.randomize_graph_by_grc(
        mf_dist_margin=4000,
        single_connection_per_pair=True,
        # constant_grc_degree=4,
        # constant_dendrite_length=15000,
        # always_pick_closest_rosette=True,
        # preserve_mf_degree=True,
        # approximate_in_degree=True,
        # local_lengths=True,
        )
    random_no_gt_mf_degree_data.append(get_prob(input_graph))

compress_pickle.dump((
    random_no_gt_mf_degree_data,
    ), f"{script_n}_random_no_gt_mf_degree_{n_samples}.gz")

