
import sys
import importlib
sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')

import my_plot
importlib.reload(my_plot)
from my_plot import MyPlotData, my_box_plot

def to_ng_coord(coord):
    return (
        int(coord[0]/4),
        int(coord[1]/4),
        int(coord[2]/40),
        )


import compress_pickle
# input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114_restricted_z.gz')
input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114.gz')
grcs = [k for k in input_graph.grcs.keys()]

rosette_loc_size = {}
for mf_id, mf in input_graph.mfs.items():
    mf.get_rosette_loc_capacity(rosette_loc_size)

mpd = MyPlotData()

for rosette_loc, size in rosette_loc_size.items():
    mpd.add_data_point(
        rosette_loc=rosette_loc,
        size=size
        )
    # grc = input_graph.grcs[rosette_loc]
    # soma_loc = grc.soma_loc
    # for rosette_loc in grc.mf_locs:
    #     x_dis = soma_loc[0] - rosette_loc[0]
    #     y_dis = soma_loc[1] - rosette_loc[1]
    #     z_dis = soma_loc[2] - rosette_loc[2]
    #     mpd.add_data_point(
    #         rosette_loc=rosette_loc,
    #         x_dis=x_dis,
    #         y_dis=y_dis,
    #         z_dis=z_dis,
    #         kind='All'
    #         )

# import seaborn as sns
# from matplotlib import pyplot as plt
# importlib.reload(my_plot); my_plot.my_jointplot(mpd, x="x_dis", y="y_dis")
# importlib.reload(my_plot); my_plot.my_jointplot(mpd, x="x_dis", y="z_dis", kind='kde')
# importlib.reload(my_plot); my_plot.my_relplot(
#     mpd, y='size', x='num_partners', hue='kind',
#     kind='line',
#     hue_order=['Data', 'Shuffle'],
#     context='paper',
#     xlim=(0, 13),
#     ylim=[0, 1.01],
#     height=4,
#     aspect=1.33,
#     y_axis_label='Cumulative Distribution',
#     x_axis_label='# of other similar patterns',
#     save_filename=f'fuzz_{mf_dist_margin}_{n_random}_cdf_201114.svg',
#     )

importlib.reload(my_plot); my_plot.my_displot(
    mpd,
    # y='size',
    x='size',
    # hue='kind',
    kind='hist',
    context='paper',
    # xlim=(0, 13),
    height=4,
    aspect=2,
    # y_axis_label='Cumulative Distribution',
    x_axis_label='# of GrCs',
    save_filename=f'rosette_size_all.svg',
    )








# sns.jointplot(data=mpd.to_dataframe(), x="x_dis", y="y_dis", hue='kind')
# plt.show()

# for num_claws in range(max_claws+1):
#     if num_claws == 0:
#         continue
#     mpd.add_data_point(
#         kind='Data',
#         num_claws=num_claws,
#         count=true_count[num_claws],
#         )

# for i, random_count in enumerate(random_counts):
#     for num_claws in range(max_claws+1):
#         if num_claws == 0:
#             continue
#         mpd.add_data_point(
#             kind='Shuffle',
#             num_claws=num_claws,
#             count=random_count[num_claws],
#             shuffle_i=i,
#             )


# importlib.reload(my_plot); my_plot.my_relplot(
#     mpd, y='count', x='num_claws', hue='kind',
#     kind='line',
#     # y='ratio', y_lims=[.25, .75],
#     context='paper',
#     # kind='violin',
#     # font_scale=1.5,
#     width=4,
#     aspect=1,
#     y_axis_label='Count',
#     x_axis_label='GrCs per Mossy Fiber',
#     save_filename='claws_per_mf_201109_plot.svg',
#     )
