import copy
from collections import defaultdict
import compress_pickle

from global_random_model2 import GlobalRandomModel

# input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/gen_db/mf_grc/input_graph_210611_grc_mf_limited_100_2_xlim_360000_600000.gz')
# input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/gen_db/mf_grc/input_graph_210611_grc_center_z_100_2_xlim_360000_600000_x_margin_20_z_margin_10.gz')
input_graph = compress_pickle.load('/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/gen_db/mf_grc/input_graph_210611_grc_center_z_100_2_xlim_360000_600000_x_margin_20_z_margin_5.0.gz')

gt_claw_count_hist = []
for _, grc in input_graph.grcs.items():
    # gt_claw_count_hist[grc.num_claws_gt] += 1
    gt_claw_count_hist.append(grc.num_claws_gt)
# gt_claw_count_hist = dict(gt_claw_count_hist)

def make_graph(model, n_grcs, n_mfs,
            config,
            seed=0):

    redundant_factor = 1
    n_share = 2
    redundant_factor = config.redundant_factor
    n_share = config.n_share

    if model == 'global_random':
        g = GlobalRandomModel(
            n_grcs=n_grcs,
            n_mfs=n_mfs,
            n_dendrites=gt_claw_count_hist,
            seed=seed,
            )
        model_desc = f'{model}'
        if redundant_factor > 1:
            g = g.make_redundant(config.redundant_factor, config.n_share)
            model_desc = f'{model}_redundant_{config.redundant_factor}_nshare_{config.n_share}'

    elif model == 'observed':
        g = input_graph
        model_desc = f'{model}'

    elif model == 'shuffle':
        g = copy.deepcopy(input_graph)
        g.randomize_graph_by_grc2(
            constant_dendrite_length=True,
            mf_dist_margin=5000,
            # approximate_mf_degree=True,
            seed=seed,
            )
        assert n_grcs == len(g.grcs)
        assert n_mfs == len(g.mfs)
        model_desc = f'{model}'
        if redundant_factor > 1:
            g = g.make_redundant(config.redundant_factor, config.n_share)
            model_desc = f'{model}_redundant_{config.redundant_factor}_nshare_{config.n_share}'

    else:
        assert False
    return g, model_desc


