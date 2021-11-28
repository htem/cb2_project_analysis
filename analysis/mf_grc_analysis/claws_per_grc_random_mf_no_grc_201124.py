
import os
import sys
import importlib
from collections import defaultdict
sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')

from tools_pattern import get_eucledean_dist

'''Load data'''
import compress_pickle
input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_201114_restricted_z.gz')
# min_pattern_len, true_data, fake_data_list = data

import my_plot
importlib.reload(my_plot)
from my_plot import MyPlotData, my_box_plot

def count_per_grc(graph):
    count = defaultdict(int)
    for grc_id, grc in graph.grcs.items():
        count[len(grc.edges)] += 1
    return count

true_count = count_per_grc(input_graph)

print("Generating random graphs...")
random_counts = []
random_n = 5
random_n = 50
mf_dist_margin = 4000
for i in range(random_n):
    print(i)
    input_graph.randomize_graph_by_mf(
        mf_dist_margin=mf_dist_margin,
        single_connection_per_pair=True,
        # preserve_in_degree=True,
        # approximate_in_degree=True,
        # local_lengths=True,
        )
    random_counts.append(count_per_grc(input_graph))

script_n = os.path.basename(__file__).split('.')[0]
import compress_pickle
compress_pickle.dump((
    true_count,
    random_counts,
    ), f"{script_n}_data.gz")



mpd = MyPlotData()

max_claws = max(true_count.keys())

for num_claws in range(max_claws+1):
    if num_claws == 0:
        continue
    mpd.add_data_point(
        kind='Data',
        num_claws=num_claws,
        count=true_count[num_claws],
        )

for i, random_count in enumerate(random_counts):
    for num_claws in range(max_claws+1):
        if num_claws == 0:
            continue
        mpd.add_data_point(
            kind='Shuffle',
            num_claws=num_claws,
            count=random_count[num_claws],
            shuffle_i=i,
            )


# importlib.reload(my_plot); my_plot.my_box_plot(
#     mpd, y='ratio', y_lims=[.25, .75], context='paper')

# importlib.reload(my_plot); my_plot.my_cat_bar_plot(
#     mpd, y='count', x='num_claws', hue='kind',
#     kind='line',
#     # y='ratio', y_lims=[.25, .75], context='paper', kind='violin',
#     # font_scale=1.5,
#     width=4,
#     y_axis_label='Count',
#     # x_axis_label='Connected / Total',
#     # save_filename='pfs_pc_connection_rate_201106_plot.svg',
#     )

# importlib.reload(my_plot); my_plot.my_relplot(
#     mpd, y='count', x='num_claws', hue='kind',
#     kind='bar',
#     # y='ratio', y_lims=[.25, .75],
#     context='paper',
#     # kind='violin',
#     # font_scale=1.5,
#     width=4,
#     aspect=1,
#     y_axis_label='Count',
#     x_axis_label='GrCs per Mossy Fiber',
#     save_filename=f'{script_n}_bar_{random_n}.svg',
#     )


importlib.reload(my_plot); my_plot.my_catplot(
    mpd, y='count', x='num_claws',
    hue='kind',
    kind='bar',
    # y='ratio', y_lims=[.25, .75],
    context='paper',
    # kind='violin',
    # font_scale=1.5,
    width=4,
    aspect=1,
    y_axis_label='Count',
    x_axis_label='# Claws per GrC',
    save_filename=f'{script_n}_bar_{random_n}.svg',
    )

