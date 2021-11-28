
import os
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


# randomize = True
# # randomize = False

# if randomize:
#     input_graph.randomize_graph(
#         # preserve_mf_out_degree=True,
#         mf_dist_margin=1000,
#         # grc_local=True,
#         # winner=True,
#         use_claw_vectors=True,
#         )

z_min = 15
z_max = 35
z_min = 20
z_max = 30
mpd = MyPlotData()

# miny = 999999
# maxy = 0
# minx = 999999
# maxx = 0

for mf_id, mf in input_graph.mfs.items():
    rosette_capacities = mf.get_all_mf_locs_size()
    for rosette_loc, claw_count in rosette_capacities.items():
        x, y, z = rosette_loc
        if x < 360000 or x > 520000:
            continue
        if z < z_min*1000 or z > z_max*1000:
            continue
        mpd.add_data_point(
            x=x/1000,
            y=y/1000,
            z=z/1000,
            claw_count=claw_count,
            )

script_n = os.path.basename(__file__).split('.')[0]

save_filename=f'{script_n}_{z_min}_{z_max}_xy.svg'
import seaborn as sns
importlib.reload(my_plot); my_plot.my_relplot(
    mpd,
    kind='scatter',
    x="x",
    y="y",
    xlim=(350, 525),
    aspect=2.5,
    width=8,
    size="claw_count",
    hue="claw_count",
    palette=sns.color_palette("rocket_r", as_cmap=True),
    # alpha=.9,
    save_filename=save_filename,
    y_axis_label='Y (µm)',
    font_scale=1,
    custom_legend=True,
    # x_axis_label='# of other similar patterns',
    )


# save_filename=f'data_xz_{z_min}_{z_max}.svg'
save_filename=f'{script_n}_{z_min}_{z_max}_xz.svg'
import seaborn as sns
importlib.reload(my_plot); g = my_plot.my_relplot(
    mpd,
    kind='scatter',
    x="x",
    y="z",
    xlim=(350, 525),
    aspect=2.5,
    width=8,
    size="claw_count",
    hue="claw_count",
    palette=sns.color_palette("rocket_r", as_cmap=True),
    # alpha=.9,
    y_axis_label='Z (µm)',
    x_axis_label='X (µm)',
    font_scale=1,
    custom_legend=True,
    save_filename=save_filename,
    )

# save_filename=f'data_heat_xy_{z_min}_{z_max}.png'
# importlib.reload(my_plot); my_plot.my_displot(
#     mpd,
#     # kind='scatter',
#     x="x",
#     y="y",
#     aspect=4,
#     # stat='frequency',
#     # stat='density',
#     bins=(4,1)
#     # size="claw_count",
#     # hue="claw_count",
#     # palette=sns.color_palette("rocket_r", as_cmap=True),
#     # alpha=.9,
#     # save_filename=save_filename,
#     )




asdf


import seaborn as sns
from matplotlib import pyplot as plt
# importlib.reload(my_plot); my_plot.my_jointplot(mpd, x="x_dis", y="y_dis")
# importlib.reload(my_plot); my_plot.my_jointplot(mpd, x="x_dis", y="z_dis", kind='kde')

postpend = ""
if randomize:
    postpend += "_rand_vectors"

for kind in ['scatter', 'kde']:
    for pair in [['x', 'y'], ['x', 'z'], ['y', 'z']]:
        save_filename=f'center_{pair[0]}{pair[1]}_{kind}{postpend}.png'
        print(f'Plotting {save_filename}')
        importlib.reload(my_plot); my_plot.my_jointplot(
            mpd, x=f"{pair[0]}_dis", y=f"{pair[1]}_dis",
            x_axis_label=f"{pair[0]}".upper(), y_axis_label=f"{pair[1]}".upper(),
            kind=f'{kind}',
            xlim=(-60000, 60000),
            ylim=(-60000, 60000),
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
