
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
radius = 200

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

total_n_pairs = 0

def get_prob(in_graph, n_share, kind, trial=0):
    global total_n_pairs
    common_pair_dist = []
    n_pairs = 0
    n_common_pairs = 0
    processed = set()
    for grc_i_id in in_graph.grcs:
        grc_i = in_graph.grcs[grc_i_id]
        x, y, z = grc_i.soma_loc
        if x < 360000 or x > 520000:
            continue
        if z < z_min*1000 or z > z_max*1000:
            continue
        rosettes_i = set([mf[1] for mf in grc_i.edges])
        for grc_j_id in in_graph.grcs:
            if grc_i_id == grc_j_id:
                continue
            if (grc_i_id, grc_j_id) in processed:
                continue
            processed.add((grc_i_id, grc_j_id))
            processed.add((grc_j_id, grc_i_id))
            grc_j = in_graph.grcs[grc_j_id]
            dist = get_eucledean_dist(grc_i.soma_loc, grc_j.soma_loc)
            dist = dist/1000
            if dist > radius:
                continue
            n_pairs += 1
            common_rosettes = set([mf[1] for mf in grc_j.edges])
            common_rosettes = common_rosettes & rosettes_i
            if len(common_rosettes) >= n_share:
                n_common_pairs += 1
                common_pair_dist.append(dist)
    mpd = MyPlotData()
    hist = defaultdict(int)
    for d in common_pair_dist:
        hist[int(d+0.5)] += 1
    total_n_pairs = n_pairs
    return hist

# mpd = get_claw_lengths(input_graph)
hist_data = defaultdict(int)
for n_share in [1, 2, 3]:
    hist = get_prob(input_graph, n_share, kind='Data')
    sum = 0
    for m, n in hist.items():
        # hist_data[n_share] += n
        sum += n
    hist_data[n_share] = sum/total_n_pairs
    # mpd_data.append(mpd)

# normalize

mpd_data = MyPlotData()
for n_share in [1, 2, 3]:
    mpd_data.add_data_point(
        n_share=n_share,
        count=hist_data[n_share],
        )


script_n = os.path.basename(__file__).split('.')[0]

importlib.reload(my_plot); my_plot.my_relplot(
    mpd_data,
    x='n_share',
    y='count',
    # hue='kind',
    kind='scatter',
    # stat="density",
    # kde=True,
    # hue_order=['Data', 'Shuffle'],
    context='paper',
    # xlim=(None, 100),
    # ylim=[0, 1.01],
    height=4,
    aspect=2,
    y_axis_label='Count',
    x_axis_label='Soma Distance (um)',
    # save_filename=f'{script_n}.svg',
    )

asdf

mpd_random = MyPlotData()
mf_dist_margin = 5000
n_random = 5
# n_random = 20
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
