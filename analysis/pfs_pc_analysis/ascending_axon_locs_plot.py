
import sys
import importlib
from collections import defaultdict
# sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')

sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')
import my_plot
importlib.reload(my_plot)
from my_plot import MyPlotData, my_box_plot

show=True
if '--save' in sys.argv:
    show=False

'''Load data'''
import compress_pickle
fname = 'ascending_axon_locs.gz'
data = compress_pickle.load(fname)
soma_locs, synapse_locs = data

mpd = MyPlotData()

for grc in soma_locs:
    soma_loc = soma_locs[grc]
    synapse_loc = synapse_locs[grc]
    x_delta = synapse_loc[0] - soma_loc[0]
    if abs(x_delta/1000) > 200:
        print(f'{grc} has x displacement of {x_delta}')
    y = synapse_loc[1]
    y = 390-(y/1000)
    if y < 0:
        print(f'{grc} has ypos of {y}')
    # if y < 40:
    #     print(f'{grc} has ypos of {y}')
    if y > 180:
        print(f'{grc} has ypos of {y}')

    mpd.add_data_point(
        x_delta=x_delta/1000,
        y=y
        )

# save_filename=f'ascending_axon_locs_plot.svg'
# # import seaborn as sns
# importlib.reload(my_plot); my_plot.my_relplot(
#     mpd,
#     kind='scatter',
#     x="x_delta",
#     y="y",
#     # aspect=2.5,
#     # width=10,
#     # xlim=(280, 630),
#     # size="claw_count",
#     # hue="claw_count",
#     # palette=sns.color_palette("mako_r", as_cmap=True),
#     # alpha=.9,
#     y_axis_label='Y (um)',
#     x_axis_label='X offset (um)',
#     save_filename=save_filename,
#     show=show,
#     )

importlib.reload(my_plot); my_plot.my_jointplot(
    mpd,
    x="x_delta",
    y="y",
    y_axis_label='Distance from PCL (um)',
    x_axis_label='X offset from soma (um)',
    xlim=[-200, 200],
    # kind=f'{kind}',
    kind='scatter',
    save_filename='ascending_axon_locs_plot_joint.svg',
    show=show,
    )

importlib.reload(my_plot); my_plot.my_displot(
    mpd,
    x="x_delta",
    # y="y",
    y_axis_label='Cumulative Distribution',
    x_axis_label='X offset from soma (um)',
    xlim=[-200, 150],
    # kind=f'{kind}',
    kind='ecdf',
    # width=4,
    height=4,
    # save_filename=save_filename,
    save_filename='ascending_axon_locs_plot_cdf.svg',
    show=show,
    )

importlib.reload(my_plot); my_plot.my_displot(
    mpd,
    x="x_delta",
    # y="y",
    y_axis_label='Distribution',
    x_axis_label='X offset from soma (um)',
    xlim=[-200, 200],
    # kind=f'{kind}',
    kind='hist',
    stat='probability',
    # width=4,
    height=4,
    # save_filename=save_filename,
    save_filename='ascending_axon_locs_plot_bar.svg',
    show=show,
    )
