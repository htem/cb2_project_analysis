
import sys
import importlib
sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')

'''Load data'''
import compress_pickle
fname = 'claws_per_mf_201109_data.gz'
data = compress_pickle.load(fname)
true_count, random_counts = data


import my_plot
importlib.reload(my_plot)
from my_plot import MyPlotData, my_box_plot

all_mpd = MyPlotData()

max_claws = max(true_count.keys())

data_mpd = MyPlotData()
for num_claws in range(max_claws+1):
    if num_claws == 0:
        continue
    data_mpd.add_data_point(
        kind='Data',
        num_claws=num_claws,
        count=true_count[num_claws],
        )

data_mpd = data_mpd.to_pdf('count', cumulative=True)
all_mpd.append(data_mpd)

for i, random_count in enumerate(random_counts):
    random_mpd = MyPlotData()
    for num_claws in range(max_claws+1):
        if num_claws == 0:
            continue
        random_mpd.add_data_point(
            kind='Shuffle',
            num_claws=num_claws,
            count=random_count[num_claws],
            shuffle_i=i,
            )
    random_mpd = random_mpd.to_pdf('count', cumulative=True)
    all_mpd.append(random_mpd)


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

importlib.reload(my_plot); my_plot.my_relplot(
    all_mpd, y='count', x='num_claws', hue='kind',
    kind='line',
    # y='ratio', y_lims=[.25, .75],
    context='paper',
    # kind='violin',
    # font_scale=1.5,
    width=4,
    aspect=1,
    y_axis_label='Cumulative Distribution',
    x_axis_label='GrCs per Mossy Fiber',
    save_filename='claws_per_mf_201109_plot_cdf.svg',
    )

