
import sys
import importlib
from collections import defaultdict
sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')

'''Load data'''
import compress_pickle
fname = 'multi_syn_ratio_201107_data.gz'
data = compress_pickle.load(fname)
total_syn_count, all_syn_count_histogram, pc_syns_histogram_list, pf_syn_count_histogram_list = data



import my_plot
importlib.reload(my_plot)
from my_plot import MyPlotData, my_box_plot

mpd = MyPlotData()

type = 'All'
for num_syn in all_syn_count_histogram:
    if num_syn == 0:
        continue
    count = all_syn_count_histogram[num_syn]
    pct = float(count) / total_syn_count
    mpd.add_data_point(
        type=type,
        num_syns_per_connection=num_syn,
        pct=pct,
        )

type = 'Per PC'
for num_syn in pc_syns_histogram_list:
    if num_syn == 0:
        continue
    count = pc_syns_histogram_list[num_syn]
    for pct in pc_syns_histogram_list[num_syn]:
        mpd.add_data_point(
            type=type,
            num_syns_per_connection=num_syn,
            pct=pct,
            )

type = 'Per pf'
for num_syn in pf_syn_count_histogram_list:
    if num_syn == 0:
        continue
    for pct in pf_syn_count_histogram_list[num_syn]:
        mpd.add_data_point(
            type=type,
            num_syns_per_connection=num_syn,
            pct=pct,
            )

importlib.reload(my_plot); my_plot.my_cat_bar_plot(
    mpd,
    x="num_syns_per_connection", y="pct", hue="type",
    hue_order=['All', 'Per PC', 'Per pf'],
    # hue_order=['All', 'Per PC'],
    # y_lims=[.25, .75],
    context='paper',
    # kind='violin',
    font_scale=1.5,
    height=4,
    y_axis_label='Normalized Frequency',
    x_axis_label='# of synapses per connection',
    save_filename='multi_syn_ratio_201107_plot.svg',
    )

