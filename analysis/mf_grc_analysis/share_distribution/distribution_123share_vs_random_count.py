
import os
import sys
import importlib
import copy
from collections import defaultdict
sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')

from tools_pattern import get_eucledean_dist

script_n = os.path.basename(__file__).split('.')[0]

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
input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114_restricted_z.gz')
# input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114.gz')
grcs = [k for k in input_graph.grcs.keys()]

z_min = 10
z_max = 40
# z_min = 20000
# z_max = 30000
x_min = 360
x_max = 520
# radius = 200

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

def get_prob(in_graph ):
    n_common_pairs = 0
    processed = set()
    total_n_pairs = 0
    hist = defaultdict(int)
    n = 0
    for grc_i_id in in_graph.grcs:
        n += 1
        if n % 1000 == 0:
            print(n, end=', ')
        grc_i = in_graph.grcs[grc_i_id]
        # x, y, z = grc_i.soma_loc
        # if x < x_min*1000 or x > x_max*1000:
        #     continue
        # if z < z_min*1000 or z > z_max*1000:
        #     continue
        rosettes_i = set([mf[0] for mf in grc_i.edges])
        for grc_j_id in in_graph.grcs:
            if grc_i_id == grc_j_id:
                continue
            if (grc_i_id, grc_j_id) in processed:
                continue
            processed.add((grc_i_id, grc_j_id))
            processed.add((grc_j_id, grc_i_id))
            grc_j = in_graph.grcs[grc_j_id]
            common_rosettes = set([mf[0] for mf in grc_j.edges])
            common_rosettes = common_rosettes & rosettes_i
            hist[len(common_rosettes)] += 1
    for k in hist:
        # fix 0 datapoint plots
        if hist[k] == 0:
            hist[k] = 1
    return hist

input_observed = copy.deepcopy(input_graph)
hist_data = get_prob(input_observed)

# input_graph.randomize_graph(random_model=True)
input_graph.randomize_graph_by_grc(
    single_connection_per_pair=True,
    constant_grc_degree=4,
    constant_dendrite_length=15000,
    always_pick_closest_rosette=True,
    )
hist_random = get_prob(input_graph)


global_random_n_grcs, hist_global_random = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_analysis/share_distribution/gen_global_random_7k_204k_data_2000.gz')

import compress_pickle
compress_pickle.dump((
    hist_data,
    hist_random,
    global_random_n_grcs,
    hist_global_random,
    ), f"{script_n}_data.gz")
# normalize

# total_n_pairs = hist_data[0] + hist_data[1] + hist_data[2] + hist_data[3]
n_grcs = len(input_graph.grcs)
# global_random_n_grcs = 204000

for h in [hist_data, hist_random, hist_global_random]:
    for k in h:
        # fix 0 datapoint plots
        h[k] += 1

mpd_data = MyPlotData()
for n_share in [1, 2, 3]:
    mpd_data.add_data_point(
        n_share=n_share,
        count=hist_data[n_share]/n_grcs,
        type='Data',
        )
    mpd_data.add_data_point(
        n_share=n_share,
        count=hist_random[n_share]/n_grcs,
        type='Local Random',
        )
    mpd_data.add_data_point(
        n_share=n_share,
        # need to divide by 2 because we're sampling only 2/200 grcs
        # (or multiply others by 2)
        count=hist_global_random[n_share]/global_random_n_grcs/2,
        type='Global Random',
        )

# mpd_data.add_data_point(
#     n_share=1,
#     count=.002,
#     type='LK-Local',
#     )
# mpd_data.add_data_point(
#     n_share=2,
#     count=.0004,
#     type='LK-Local',
#     )
# mpd_data.add_data_point(
#     n_share=3,
#     count=.00005,
#     type='LK-Local',
#     )

importlib.reload(my_plot); my_plot.my_relplot(
    mpd_data,
    x='n_share',
    y='count',
    hue='type',
    # hue_order=['Data', 'Random Model'],
    kind='scatter',
    context='paper',
    # ylim=[.005, 50],
    ylim=[.005, 2000],
    xlim=[.7, 3.9],
    log_scale_y=True,
    s=70,
    # xticklabels=['', 1, '', 2, '', 3, ''],
    xticks=[1, 2, 3],
    height=4,
    aspect=.9,
    y_axis_label='# of GrC pairs with\nshared inputs / # of GrCs',
    x_axis_label='Shared Inputs',
    # save_filename=f'{script_n}.pdf',
    save_filename=f'{script_n}.svg',
    )

