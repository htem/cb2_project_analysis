
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
mpd = MyPlotData()

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
    same_mf_pair = []
    same_mf_count = 0
    num_connections = 0
    for grc_id in in_graph.grcs:
        grc = in_graph.grcs[grc_id]
        soma_loc = grc.soma_loc
        x, y, z = soma_loc
        # if x < x_min or x > x_max:
        #     continue
        # if z < z_min or z > z_max:
        #     continue
        mfs = [mf[0] for mf in grc.edges]
        same_mf_count += len(mfs) - len(set(mfs))
        num_connections += len(mfs)
        for k in set(mfs):
            mfs.remove(k)
        if len(mfs):
            same_mf_pair.append((grc_id, mfs))
    print(f'same_mf_count: {same_mf_count}')
    print(f'num_connections: {num_connections}')
    print(f'same_mf_pair: {same_mf_pair}')
    return same_mf_count

# mpd = get_claw_lengths(input_graph)
mpd = get_prob(input_graph)

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
