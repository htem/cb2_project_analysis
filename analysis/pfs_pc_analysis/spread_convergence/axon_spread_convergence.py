
import sys
import importlib
import random
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
input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114_xlim_320_560_zlim_10_40.gz')

pc_width = 160000  # empirically found across PCs
# pfs_spread = 100000
num_trials = 100
# num_trials = 2

# stretch_factor = 1.25  # stretch 240um to 300um width
stretch_factor = 1  # stretch 240um to 300um width

# get PC center
min_x = sys.maxsize
max_x = -sys.maxsize
for grc_id, grc in input_graph.grcs.items():
    x, y, z = grc.soma_loc
    min_x = min(x, min_x)
    max_x = max(x, max_x)
center_x = (min_x+max_x)/2
center_x *= stretch_factor

convergence_range = ((center_x-pc_width*0.5), (center_x+pc_width*0.5))
print(f'convergence_range: {convergence_range}')

# get grcs, simulate pf locations, filter out, then count # of unique mfs in the convergence

def simulate(graph, pfs_spread, pc_width):
    converged_grcs = []
    for grc_id, grc in graph.grcs.items():
        x, _, _ = grc.soma_loc
        x *= stretch_factor
        pf_delta = (random.random()-.5)*pfs_spread
        pf_x = x + pf_delta
        if pf_x > convergence_range[0] and pf_x < convergence_range[1]:
            converged_grcs.append(grc_id)
    converged_mfs = set()
    for grc_id in converged_grcs:
        converged_mfs |= graph.grcs[grc_id].mfs
    return len(converged_mfs), len(converged_grcs)


mpd = MyPlotData()
ref_n_grc = None
ref_n_mfs = None
for pfs_spread in range(0, 201, 10):
    # print("pfs_spread:", pfs_spread)
    pfs_spread = pfs_spread*1000  # to nm
    for n_trial in range(num_trials):
        n_mfs, n_grc = simulate(input_graph, pfs_spread, pc_width)
        if ref_n_grc is None:
            ref_n_grc = n_grc
            ref_n_mfs = n_mfs
        # print((n_mfs, n_grc))
        mpd.add_data_point(
            pfs_spread=pfs_spread/1000,
            n_grc=n_grc,
            n_mfs=n_mfs,
            n_grc_ratio=n_grc/ref_n_grc,
            n_mfs_ratio=n_mfs/ref_n_mfs,
            n_trial=n_trial,
            )


importlib.reload(my_plot); my_plot.my_relplot(
    mpd,
    x="pfs_spread",
    y="n_mfs",
    y_axis_label='# converged MFs',
    x_axis_label='pf spread (um)',
    context='paper',
    xlim=[None, 150],
    ylim=[0, None],
    height=4,
    # aspect=1,
    # kind=f'{kind}',
    kind='line',
    save_filename='axon_spread_convergence.svg',
    show=show,
    )

importlib.reload(my_plot); my_plot.my_relplot(
    mpd,
    x="pfs_spread",
    y="n_mfs",
    y_axis_label='# converged MFs',
    x_axis_label='pf spread (um)',
    context='paper',
    xlim=[None, 150],
    ylim=[0, None],
    height=4,
    # aspect=1,
    # kind=f'{kind}',
    kind='line',
    save_filename='axon_spread_convergence_pct.svg',
    show=show,
    )


asdf


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
