
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
# min_pattern_len, true_data, fake_data_list = data

import my_plot
importlib.reload(my_plot)
from my_plot import MyPlotData, my_box_plot

def to_ng_coord(coord):
    return (
        int(coord[0]/4),
        int(coord[1]/4),
        int(coord[2]/40),
        )

# def get_eucledean_dist(a, b):
#     return np.linalg.norm(
#         (a[0]-b[0], a[1]-b[1], a[2]-b[2]))

# def get_distance(u, v):
#     return get_eucledean_dist(u, v)

import compress_pickle
# input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114_restricted_z.gz')
input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114.gz')
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

observed_data = get_prob(input_graph)

import compress_pickle
compress_pickle.dump((
    observed_data,
    observed_data,
    ), f"gen_global_random_7k_204k_data_{n}.gz")








script_n = os.path.basename(__file__).split('.')[0]

importlib.reload(my_plot); my_plot.my_relplot(
    mpd_data,
    x='soma_dist',
    y='count',
    # hue='kind',
    kind='line',
    # stat="density",
    # kde=True,
    # hue_order=['Data', 'Shuffle'],
    context='paper',
    xlim=(None, 100),
    # ylim=[0, 1.01],
    height=4,
    aspect=2,
    y_axis_label='Count',
    x_axis_label='Soma Distance (um)',
    save_filename=f'{script_n}_data_hist.svg',
    )

mpd_random = MyPlotData()
mf_dist_margin = 5000
n_random = 5
n_random = 20
for i in range(n_random):
    print(i)
    input_graph.randomize_graph_by_grc(
        mf_dist_margin=mf_dist_margin,
        single_connection_per_pair=True,
        # preserve_mf_degree=True,
        # approximate_in_degree=True,
        # local_lengths=True,
        )
    mpd = get_prob(input_graph, kind='Random', trial=i)
    mpd_random.append(mpd)

# script_n = os.path.basename(__file__).split('.')[0]
# import compress_pickle
# compress_pickle.dump((
#     mpd_data,
#     mpd_random,
#     ), f"{script_n}_data.gz")

mpd_total = MyPlotData()
mpd_total.append(mpd_data)
mpd_total.append(mpd_random)

importlib.reload(my_plot); my_plot.my_relplot(
    mpd_total,
    x='soma_dist',
    y='count',
    hue='kind',
    kind='line',
    # stat="density",
    # kde=True,
    # hue_order=['Data', 'Shuffle'],
    context='paper',
    xlim=(None, 100),
    # ylim=[0, 1.01],
    height=4,
    aspect=2,
    y_axis_label='Count',
    x_axis_label='Soma Distance (um)',
    save_filename=f'{script_n}_compare_hist.svg',
    )


asdf

# importlib.reload(my_plot); my_plot.my_relplot(
#     mpd_random, x='soma_dist',
#     # hue='kind',
#     kind='line',
#     # stat="density",
#     # kde=True,
#     # hue_order=['Data', 'Shuffle'],
#     context='paper',
#     # xlim=(0, 13),
#     # ylim=[0, 1.01],
#     height=4,
#     aspect=3,
#     y_axis_label='Count',
#     x_axis_label='Soma Distance (um)',
#     # save_filename=f'{script_n}_data_hist.svg',
#     )


mf_dist_margin = 5000
# prob of random graphs
input_graph.randomize_graph_by_grc(
    mf_dist_margin=mf_dist_margin,
    single_connection_per_pair=True,
    # preserve_mf_degree=True,
    # approximate_in_degree=True,
    # local_lengths=True,
    )

get_prob(input_graph)

mpd = MyPlotData()

for i in range(100):
    input_graph.randomize_graph_by_grc(
        mf_dist_margin=mf_dist_margin,
        single_connection_per_pair=True,
        preserve_mf_degree=True,
        # approximate_in_degree=True,
        # local_lengths=True,
        )
    n = get_prob(input_graph)
    mpd.add_data_point(
        n=n)

script_n = os.path.basename(__file__).split('.')[0]
import compress_pickle
compress_pickle.dump((
    mpd,
    # random_counts,
    ), f"{script_n}_data.gz")

importlib.reload(my_plot); my_plot.my_catplot(
    mpd, y='n',
    kind='box',
    # y='ratio', y_lims=[.25, .75],
    context='paper',
    # kind='violin',
    # font_scale=1.5,
    width=4,
    aspect=1,
    y_axis_label='Count',
    x_axis_label='GrCs per Mossy Fiber',
    # save_filename=f'{script_n}_bar.svg',
    )


asdf

# random_claw_lengths = []
# for i in range(10):
#     input_graph.randomize_graph()
#     random_claw_lengths.append

script_n = os.path.basename(__file__).split('.')[0]

importlib.reload(my_plot); my_plot.my_relplot(
    mpd, x='claw_length',
    y='prob',
    context='paper',
    height=4,
    aspect=1.33,
    y_axis_label='Probability',
    x_axis_label='Distance (um)',
    save_filename=f'{script_n}.svg',
    )
