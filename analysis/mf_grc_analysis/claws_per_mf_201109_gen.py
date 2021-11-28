
import sys
import importlib
from collections import defaultdict
sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')

from tools_pattern import get_eucledean_dist

'''Load data'''
import compress_pickle
fname = '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/input_graph_200919.gz'
input_graph = compress_pickle.load(fname)
# min_pattern_len, true_data, fake_data_list = data

import my_plot
importlib.reload(my_plot)
from my_plot import MyPlotData, my_box_plot

# adapted from /n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/mf_grc_model/gen_gt_graph_200917.py

def count_per_mf(graph):
    count = defaultdict(int)
    for mf_id in graph.mfs:
        mf = graph.mfs[mf_id]
        count[len(mf.claws)] += 1
    return count

true_count = count_per_mf(input_graph)

# for k in sorted(count.keys()):
# for k in range(max(count.keys())+1):
#     print(f'{k},{count[k]}')

print("Generating random graphs...")
random_counts = []
for i in range(1000):
    print(i)
    input_graph.randomize_graph()
    random_counts.append(count_per_mf(input_graph))


import compress_pickle
compress_pickle.dump((
    true_count,
    random_counts,
    ), "claws_per_mf_201109_data.gz")

