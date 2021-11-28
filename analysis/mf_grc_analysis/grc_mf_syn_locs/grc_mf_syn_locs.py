
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

# def get_eucledean_dist(a, b):
#     return np.linalg.norm(
#         (a[0]-b[0], a[1]-b[1], a[2]-b[2]))

# def get_distance(u, v):
#     return get_eucledean_dist(u, v)

import compress_pickle
# input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114_restricted_z.gz')
input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114.gz')
grcs = [k for k in input_graph.grcs.keys()]

mpd = MyPlotData()

for grc_id in grcs:
    grc = input_graph.grcs[grc_id]
    soma_loc = grc.soma_loc
    for rosette_loc in grc.mf_locs:
        x_dis = soma_loc[0] - rosette_loc[0]
        y_dis = soma_loc[1] - rosette_loc[1]
        z_dis = soma_loc[2] - rosette_loc[2]
        mpd.add_data_point(
            grc_id=grc_id,
            x_dis=x_dis,
            y_dis=y_dis,
            z_dis=z_dis,
            kind='All'
            )

import seaborn as sns
from matplotlib import pyplot as plt
# importlib.reload(my_plot); my_plot.my_jointplot(mpd, x="x_dis", y="y_dis")
# importlib.reload(my_plot); my_plot.my_jointplot(mpd, x="x_dis", y="z_dis", kind='kde')

save_filename='all_xz.svg'
importlib.reload(my_plot); my_plot.my_jointplot(
    mpd, x="x_dis", y="z_dis",
    x_axis_label='X', y_axis_label='Z',
    save_filename=save_filename)

save_filename='all_xy.svg'
importlib.reload(my_plot); my_plot.my_jointplot(
    mpd, x="x_dis", y="y_dis",
    x_axis_label='X', y_axis_label='Y',
    save_filename=save_filename)

save_filename='all_yz.svg'
importlib.reload(my_plot); my_plot.my_jointplot(
    mpd, x="y_dis", y="z_dis",
    x_axis_label='Y', y_axis_label='Z',
    save_filename=save_filename)







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
